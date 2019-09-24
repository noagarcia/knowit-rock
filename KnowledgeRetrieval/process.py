# coding=utf-8

import argparse
import os
import random
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

import sys
sys.path.insert(0,'.')
import utils
from rank import rank
from model import BertScoring
from dataloader import RetrievalDataset

import logging
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
    parser.add_argument("--num_pairs", default=10, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--workers", default=8)
    return parser.parse_args()


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
    model = BertScoring.from_pretrained(args.bert_model,
        cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1)),)
    model.to(args.device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Load data
    trainDataObject = RetrievalDataset(args, split='train', tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(trainDataObject, batch_size=args.batch_size, shuffle=True,
                                               pin_memory=True, num_workers=args.workers)
    num_train_optimization_steps = int(trainDataObject.num_samples / args.batch_size) * args.num_train_epochs

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

            # input_ids, input_mask, segment_ids = batch
            input_ids, input_mask, segment_ids, id_q, label_ids = batch

            # loss = model(input_ids, segment_ids, input_mask)
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

            plotter.plot('loss', 'train', 'Loss', global_step, global_loss.avg)

    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(outdir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(outdir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())


def evaluate(args, outdir, split):

    # Load Model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    output_model_file = os.path.join(outdir, WEIGHTS_NAME)
    output_config_file = os.path.join(outdir, CONFIG_NAME)
    config = BertConfig(output_config_file)
    model = BertScoring(config)
    model.load_state_dict(torch.load(output_model_file))
    model.to(args.device)
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(args.device, n_gpu))
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Data
    evalDataObject = RetrievalDataset(args, split=split, tokenizer=tokenizer)
    eval_dataloader = torch.utils.data.DataLoader(evalDataObject, batch_size=args.batch_size, shuffle=False,
                                               pin_memory=True, num_workers=args.workers)

    # Run prediction
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", eval_dataloader.__len__())
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    batch_idx = 0
    for _, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        input_ids, input_mask, segment_ids, id_q, truelabel = batch
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)

        with torch.no_grad():
            scores = model(input_ids, segment_ids, input_mask)

        scores = scores.detach().cpu().numpy()

        if batch_idx == 0:
            scores1 = scores[:,0]
            scores2 = scores[:,1]
            idxq = id_q.numpy()
            labels = truelabel.numpy()
        else:
            scores1 = np.concatenate((scores1, scores[:,0]), axis=0)
            scores2 = np.concatenate((scores2, scores[:,1]), axis=0)
            idxq = np.concatenate((idxq, id_q.numpy()), axis=0)
            labels = np.concatenate((labels, truelabel.numpy()), axis=0)

        batch_idx += 1

    utils.save_obj(scores1, os.path.join(args.data_dir, 'retieval_scores_%s.pckl' % split))
    utils.save_obj(idxq, os.path.join(args.data_dir, 'retrieval_idxq_%s.pckl' % split))
    utils.save_obj(labels, os.path.join(args.data_dir, 'retrieval_labels_%s.pckl' % split))
    medR1, recall1, medR2, recall2 = rank(scores1, scores2, idxq, labels)
    logger.info('Accuracy medR {medR:.2f}\t Recall {recall}'.format(medR=medR1, recall=recall1))


if __name__ == "__main__":

    args = get_params()
    train_name = 'BertScoring_maxseq%d_%dpairs' % (args.max_seq_length, args.num_pairs)
    outdir = os.path.join('Training/KnowledgeRetrieval/', train_name)

    if not os.path.isfile(os.path.join(outdir, 'pytorch_model.bin')):
        global plotter
        plotter = utils.VisdomLinePlotter(env_name=train_name)
        train(args, outdir)

    if not os.path.exists(os.path.join(args.data_dir, 'retieval_scores_test.pckl')):
        evaluate(args, outdir, split='test')
    if not os.path.exists(os.path.join(args.data_dir, 'retieval_scores_val.pckl')):
        evaluate(args, outdir, split='val')