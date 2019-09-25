import os
import argparse
import logging
import numpy as np
import pandas as pd
import random
from tqdm import tqdm, trange

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from pytorch_pretrained_bert.modeling import BertConfig, CONFIG_NAME, PreTrainedBertModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import re
import sys
sys.path.insert(0,'.')
import utils


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='Data/', type=str)
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--do_lower_case", default=True)
    parser.add_argument('--seed', type=int, default=181)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--device", default='cuda', type=str, help="cuda, cpu")
    parser.add_argument('--csvtrain', default='data_full_train.csv', help='Training set data file')
    parser.add_argument('--csvval', default='data_full_val.csv', help='Dataset val data file')
    parser.add_argument('--csvtest', default='data_full_test_qtypes.csv', help='Dataset test data file')
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=1, type=int)
    parser.add_argument('--train_max_seq_len', default=256)
    parser.add_argument('--eval_max_seq_len', default=512)
    parser.add_argument('--topk', default=5)
    parser.add_argument('--use_captions', action='store_true')
    parser.add_argument('--captionsfile', default='Captions/knowit_captions.csv')
    parser.add_argument('--numframes', type=int, default=5)
    return parser.parse_args()


class BertForMultipleChoiceFeatures(PreTrainedBertModel):
    def __init__(self, config, num_choices=2):
        super(BertForMultipleChoiceFeatures, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, feat=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if feat:
            return pooled_output
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext


class LanguageData(object):
    def __init__(self, id, q, a1, a2, a3, a4, subs, kg, label, caption):
        self.id = id
        self.question = q
        self.subtitles = subs
        self.answers = [a1, a2, a3, a4]
        self.kg = kg
        self.label = label
        self.caption = caption


def _truncate_seq_pair_inv(tokens_a, tokens_b, max_length):

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()


def get_kg_eval(thisidx, scores1, idx_q, num_r, k):

    pos_iq = np.where(idx_q == thisidx)[0]
    this_ir = list(range(num_r))
    this_scores1 = scores1[pos_iq]
    assert len(this_ir) == len(this_scores1)

    ranking = np.argsort(this_scores1).tolist()
    top_r_idx = ranking[:k]
    return top_r_idx


class InputFeatures(object):
    def __init__(self, sample_id, choices_features, label, question):
        self.example_id = sample_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'question' : question
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def read_data(args, split):

    # Load data
    if split == 'train':
        input_file = os.path.join(args.data_dir, args.csvtrain)
        filekbsplit = os.path.join(args.data_dir, 'KB/reason_idx_to_kb_train.pckl')
    elif split == 'val':
        input_file = os.path.join(args.data_dir, args.csvval)
        filekbsplit = os.path.join(args.data_dir, 'KB/reason_idx_to_kb_val.pckl')
        scores = utils.load_obj(os.path.join(args.data_dir, 'retrieval_idxq_val.pckl'))
        idxq = utils.load_obj(os.path.join(args.data_dir, 'retrieval_idxq_val.pckl'))
    elif split == 'test':
        input_file = os.path.join(args.data_dir, args.csvtest)
        filekbsplit = os.path.join(args.data_dir, 'KB/reason_idx_to_kb_test.pckl')
        scores = utils.load_obj(os.path.join(args.data_dir, 'retieval_scores_test.pckl'))
        idxq = utils.load_obj(os.path.join(args.data_dir, 'retrieval_idxq_test.pckl'))
    df = pd.read_csv(input_file, delimiter='\t')
    logger.info('Loaded file with %d samples' % len(df))
    kb = utils.load_obj(os.path.join(args.data_dir, 'KB/reason_kb_dict.pckl'))
    idx_to_kb_this = utils.load_obj(filekbsplit)
    kblist = [values for cluster, values in kb.items()]

    if args.use_captions:
        df_caps = pd.read_csv(os.path.join(args.data_dir, args.captionsfile), delimiter=',')
        framepaths = utils.get_frame_paths('', df, numframes=args.numframes)

    samples = []
    for index, row in df.iterrows():

        # Sample info
        q = row['question']
        a1 = row['answer1']
        a2 = row['answer2']
        a3 = row['answer3']
        a4 = row['answer4']
        subs = cleanhtml(row['subtitle'].replace('<br />', ' ').replace(' - ', ' '))
        label = int(df['idxCorrect'].iloc[index] - 1)

        # Captions
        caption = ''
        if args.use_captions:
            clipaths = framepaths[index]
            for path in clipaths:
                caption += df_caps[df_caps['image_files'] == path]['caption'].values[0]

        # Find knowledge to use
        if split == 'train':
            cluster_pos = idx_to_kb_this[index]
            # reasons = kb[cluster_pos]
            reason_retrieved = []
            reason_retrieved.append(kb[cluster_pos])
            for i in list(range(args.topk-1)):
                idx = random.randint(0,len(idx_to_kb_this)-1)
                cluster_rand = idx_to_kb_this[idx]
                reason_retrieved.append(kb[cluster_rand])
            reasons = ' '.join(reason_retrieved)
        else:
            idx_retrieved_reason = get_kg_eval(index, scores, idxq, len(kb), k=args.topk)
            reason_retrieved = []
            for idxr in idx_retrieved_reason:
                reason_retrieved.append(kblist[idxr])
            reasons = ' '.join(reason_retrieved)

        # Add new sample
        samples.append(LanguageData(id=index, subs=subs, q=q, kg=reasons, a1=a1, a2=a2, a3=a3, a4=a4, label=label, caption=caption))

    return samples


def convert_to_input(samples, tokenizer, max_seq_length):
    features = []
    for index, sample in enumerate(samples):
        subtitle = tokenizer.tokenize(sample.subtitles)
        question = tokenizer.tokenize(sample.question)
        kg = tokenizer.tokenize(sample.kg)
        caption = tokenizer.tokenize(sample.caption)
        choices_features = []
        for answer_index, answer in enumerate(sample.answers):
            context_tokens_choice = caption[:] + subtitle[:] + question[:]
            ending_tokens =  tokenizer.tokenize(answer) + kg[:]
            _truncate_seq_pair_inv(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = sample.label
        features.append(
            InputFeatures(
                sample_id = sample.id,
                choices_features = choices_features,
                label = label,
                question = sample.question
            )
        )

    return features


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def train(args, outdir, modelname):

    # Set GPU
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(args.device, n_gpu))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Create training directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = BertForMultipleChoiceFeatures.from_pretrained(args.bert_model,
        cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1)),num_choices=4)
    model.to(args.device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Read data
    samples_data = read_data(args, 'train')
    num_train_optimization_steps = int(len(samples_data) / args.batch_size) * args.num_train_epochs
    data_features = convert_to_input(samples_data, tokenizer, args.train_max_seq_len)
    logger.info("***** Running training of BertReasoning (language features) *****")
    logger.info("Num examples = %d", len(samples_data))
    logger.info("Batch size = %d", args.batch_size)
    logger.info("Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor(select_field(data_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(data_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(data_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in data_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    # Optimizer
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    global_step = 0
    global_loss = utils.AverageMeter()
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataloader.__len__())
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            global_loss.update(loss.item(), input_ids.shape[0])

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1


    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(outdir, modelname)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(outdir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())


def compute_embeddings(args, modeldir, modelname, split):

    # Compute embeddings only if they have not been computed yet
    filedir = os.path.join(args.data_dir, 'Features/')
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    embsfile = os.path.join(filedir, 'language_bert_%s.pckl' % split)
    if os.path.exists(embsfile):
        return

    # Load model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    output_model_file = os.path.join(modeldir, modelname)
    output_config_file = os.path.join(modeldir, CONFIG_NAME)
    config = BertConfig(output_config_file)
    model = BertForMultipleChoiceFeatures(config, num_choices=4)
    model.load_state_dict(torch.load(output_model_file))
    model.to(args.device)

    # Read data
    samples_data = read_data(args, split)
    data_features = convert_to_input(samples_data, tokenizer, args.eval_max_seq_len)
    logger.info("***** Extracting language features *****")
    logger.info("  Num samples = %d", len(samples_data))
    all_input_ids = torch.tensor(select_field(data_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(data_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(data_features, 'segment_ids'), dtype=torch.long)
    all_qa_id = torch.tensor([f.example_id for f in data_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_qa_id)

    # Get language embeddings
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    for batch_idx, inputs in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        input_ids = inputs[0].to(args.device)
        input_mask = inputs[1].to(args.device)
        segment_ids = inputs[2].to(args.device)

        #  Apply model
        with torch.no_grad():
            features = model(input_ids, segment_ids, input_mask, labels=None, feat=True)

        # Save features
        if batch_idx==0:
            feat = np.expand_dims(features.to('cpu').numpy(), axis=0)
        else:
            feat = np.concatenate((feat, np.expand_dims(features.to('cpu').numpy(), axis=0)),axis=0)

    # save
    logger.info('Saving to %s...' % embsfile)
    utils.save_obj(feat, embsfile)


def evaluate(args, modeldir, modelname):

    args.eval_batch_size = 64

    # Load model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    output_model_file = os.path.join(modeldir, modelname)
    output_config_file = os.path.join(modeldir, CONFIG_NAME)
    config = BertConfig(output_config_file)
    model = BertForMultipleChoiceFeatures(config, num_choices=4)
    model.load_state_dict(torch.load(output_model_file))
    model.to(args.device)

    # Read data
    samples_data = read_data(args, 'test')
    data_features = convert_to_input(samples_data, tokenizer, args.eval_max_seq_len)
    logger.info("***** Computing only language evaluation *****")
    logger.info("  Num samples = %d", len(samples_data))
    all_input_ids = torch.tensor(select_field(data_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(data_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(data_features, 'segment_ids'), dtype=torch.long)
    all_labels = torch.tensor([f.label for f in data_features], dtype=torch.long)
    all_indices = torch.tensor([f.example_id for f in data_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels, all_indices)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    for batch_idx, inputs in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        input_ids = inputs[0].to(args.device)
        input_mask = inputs[1].to(args.device)
        segment_ids = inputs[2].to(args.device)
        label_this = inputs[3].to(args.device)
        index_this = inputs[4]

        # Output of the model
        with torch.no_grad():
            output = model(input_ids, segment_ids, input_mask, labels=None, feat=False)
        _, predicted = torch.max(output, 1)

        # Store outpputs
        if batch_idx==0:
            out = predicted.data.cpu().numpy()
            label = label_this.cpu().numpy()
            index = index_this
        else:
            out = np.concatenate((out,predicted.data.cpu().numpy()),axis=0)
            label = np.concatenate((label,label_this.cpu().numpy()),axis=0)
            index = np.concatenate((index, index_this), axis=0)

    # Compute Accuracy
    acc = np.sum(out == label)/len(out)
    logger.info('*' *20)
    logger.info('Model in %s' %modeldir)
    df = pd.read_csv('Data/data_full_test_qtypes.csv', delimiter='\t')
    utils.accuracy_perclass(df, out, label, index)
    logger.info('Overall Accuracy\t%.03f' % acc)
    logger.info('*' * 20)


if __name__ == "__main__":

    args = get_params()

    if args.use_captions:
        train_name = 'AnswerPrediction_caption'
        modelname = 'ROCK-caption-weights.pth.tar'
    else:
        train_name = 'BertReasoning_topk%d_maxseq%d' % (args.topk, args.train_max_seq_len)
        modelname = 'pytorch_model.bin'

    outdir = os.path.join('Training/VideoReasoning/', train_name)
    if not os.path.isfile(os.path.join(outdir, modelname)):
        train(args, outdir, modelname)

    if args.use_captions:
        evaluate(args, outdir, modelname)
    else:
        compute_embeddings(args, outdir, modelname, split = 'train')
        compute_embeddings(args, outdir, modelname, split = 'val')
        compute_embeddings(args, outdir, modelname, split = 'test')
