# coding=utf-8

import argparse
import logging
import os
import random
from io import open
import pandas as pd
import utils

import numpy as np
import torch
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForMultipleChoice, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


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
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--max_seq_length", default=120, type=int)
    parser.add_argument("--workers", default=8)
    return parser.parse_args()


def select_field(features, field):
    return [
        [
            data[field]
            for data in feature.choices_features
        ]
        for feature in features
    ]


class Sample(object):
    def __init__(self, id, question, answer_0, answer_1, answer_2, answer_3, label = None):
        self.id = id
        self.question = question
        self.answers = [
            answer_0,
            answer_1,
            answer_2,
            answer_3,
        ]
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def read_samples(input_file):

    df = pd.read_csv(input_file, delimiter='\t')
    print('Loaded file with %d samples' % len(df))

    examples = [
        Sample(
            id = index,
            question = row['question'], # question
            answer_0 = row['answer1'],
            answer_1 = row['answer2'],
            answer_2 = row['answer3'],
            answer_3 = row['answer4'],
            label = int(row['idxCorrect'] - 1)
        ) for index, row in df.iterrows()
    ]

    return examples


def convert_to_features(examples, tokenizer, max_seq_length):
    # - [CLS] question [SEP] answer1 [SEP]
    # - [CLS] question [SEP] answer2 [SEP]
    # - [CLS] question [SEP] answer3 [SEP]
    # - [CLS] question [SEP] answer4 [SEP]
    features = []
    for example_index, example in enumerate(examples):

        q_tokens = tokenizer.tokenize(example.question)

        choices_features = []
        for answer_index, answer in enumerate(example.answers):

            context_tokens = q_tokens[:]
            a_tokens =  tokenizer.tokenize(answer)
            _truncate_seq_pair(context_tokens, a_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens + ["[SEP]"] + a_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens) + 2) + [1] * (len(a_tokens) + 1)
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

        label = example.label
        if example_index < 1:
            logger.info("*** Example ***")
            logger.info("id: {}".format(example.id))
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))

        features.append(
            InputFeatures(
                example_id = example.id,
                choices_features = choices_features,
                label = label
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):

    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def train(args, outdir):


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
    model = BertForMultipleChoice.from_pretrained(args.bert_model,
        cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1)),
        num_choices=4)
    model.to(args.device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Load data
    train_examples = read_samples(os.path.join(args.data_dir, args.csvtrain))
    num_train_optimization_steps = int(len(train_examples) / args.batch_size) * args.num_train_epochs
    train_features = convert_to_features(train_examples, tokenizer, args.max_seq_length)
    all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)

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
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
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


            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(outdir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(outdir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())


def get_scores(args, split='test'):

    if split == 'train':
        input_file = os.path.join(args.data_dir, args.csvtrain)
        filescores = os.path.join(args.data_dir, 'PriorScores/priorscores_answers_train.pckl')
    elif split == 'val':
        input_file = os.path.join(args.data_dir, args.csvval)
        filescores = os.path.join(args.data_dir, 'PriorScores/priorscores_answers_val.pckl')
    elif split == 'test':
        input_file = os.path.join(args.data_dir, args.csvtest)
        filescores = os.path.join(args.data_dir, 'PriorScores/priorscores_answers_test.pckl')

    # Load Model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    output_model_file = os.path.join(outdir, WEIGHTS_NAME)
    output_config_file = os.path.join(outdir, CONFIG_NAME)
    config = BertConfig(output_config_file)
    model = BertForMultipleChoice(config, num_choices=4)
    model.load_state_dict(torch.load(output_model_file))
    model.to(args.device)
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(args.device, n_gpu))
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Data
    eval_examples = read_samples(input_file)
    eval_features = convert_to_features(eval_examples, tokenizer, args.max_seq_length)
    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    all_labels = torch.tensor([example.label for example in eval_examples], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Run prediction
    logger.info("***** Compute prior scores *****")
    logger.info("Num examples = %d", len(eval_examples))
    logger.info("Batch size = %d", args.eval_batch_size)
    model.eval()
    batch_idx = 0
    for _, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        input_ids, input_mask, segment_ids, truelabel = batch
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            logits = nn.functional.softmax(logits)

        logits = logits.detach().cpu().numpy()
        if batch_idx == 0:
            scores = logits
        else:
            scores = np.concatenate((scores, logits), axis=0)
        batch_idx += 1

    if not os.path.exists(os.path.dirname(filescores)):
        os.mkdir(os.path.dirname(filescores))
    utils.save_obj(scores, filescores)
    logger.info('Prior scores for %s saved into %s' % (split, filescores))


if __name__ == "__main__":

    args = get_params()
    train_name = 'PriorScores_maxseq%d' % (args.max_seq_length)
    outdir = os.path.join('Training/PriorScores', train_name)

    if not os.path.isfile(os.path.join(outdir, 'pytorch_model.bin')):
        train(args, outdir)

    get_scores(args, 'train')
    get_scores(args, 'val')
    get_scores(args, 'test')