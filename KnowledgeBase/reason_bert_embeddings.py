import os
import argparse
import logging
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

import sys
sys.path.insert(0,'..')
import utils


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='Data/', type=str)
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--device", default='cuda', type=str, help="cuda, cpu")
    parser.add_argument('--csvtrain', default='data_full_train.csv', help='Training set data file')
    parser.add_argument('--csvval', default='data_full_val.csv', help='Dataset val data file')
    parser.add_argument('--csvtest', default='data_full_test_qtypes.csv', help='Dataset test data file')
    parser.add_argument('--embsfile', default='cache/kb_allreasons_60len.pckl')
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Total batch size for eval.")
    parser.add_argument("--max_seq_length", default=60, type=int, help="Total batch size for eval.")
    parser.add_argument("--do_lower_case", default=True, help="Set this flag if you are using an uncased model.")
    return parser.parse_args()


class BERTFeatures(object):
    def __init__(self, instance_id, tokens, input_ids, input_mask, segment_ids):
        self.instance_id = instance_id
        self.token = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def read_data(args):
    """Loads all reasons in the dataset into a list."""

    files = [args.csvtrain, args.csvval, args.csvtest]
    allreasons = []

    for f in files:
        filename = os.path.join(args.data_dir, f)
        df = pd.read_csv(filename, delimiter='\t')
        logger.info('Loaded %s with %d samples' % (f, len(df)))
        allreasons += [row['reason'] for index, row in df.iterrows()]

    return allreasons


def convert_to_features(reasons, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""

    # Each choice will correspond to a sample on which we run the
    # inference.
    # - [CLS] reason [SEP]
    features = []
    for idx, sample in enumerate(reasons):

        sample_tokens = tokenizer.tokenize(sample)
        while len(sample_tokens) > max_seq_length - 2:
            sample_tokens.pop()

        tokens = ["[CLS]"] + sample_tokens + ["[SEP]"]
        segment_ids = [0] * (len(sample_tokens) + 2)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if idx < 1:
            logger.info("*** Example ***")
            logger.info("id: {}".format(idx))
            logger.info("tokens: {}".format(' '.join(tokens)))
            logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
            logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))

        features.append(
            BERTFeatures(
                instance_id = idx,
                tokens = tokens,
                input_ids = input_ids,
                input_mask = input_mask,
                segment_ids = segment_ids
            )
        )

    return features


def compute_embeddings(args):

    # Load pre-trained model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = BertModel.from_pretrained(args.bert_model)
    model.to(args.device)
    model.eval()

    # Prepare data
    allreasons = read_data(args)
    features = convert_to_features(allreasons, tokenizer, args.max_seq_length)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_ids = torch.tensor([f.instance_id for f in features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Run prediction for full data
    logger.info("***** Extracting pre-trained BERT embeddings *****")
    logger.info("Num samples = %d", len(allreasons))
    for batch_idx, inputs in enumerate(eval_dataloader):

        # Send data to GPU
        input_ids = inputs[0].to(args.device)
        input_mask = inputs[1].to(args.device)
        segment_ids = inputs[2].to(args.device)

        #  Apply model
        with torch.no_grad():
            _, cls_output = model(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)

        # Save embeddings
        if batch_idx==0:
            embeddings = cls_output.data.cpu().numpy()
        else:
            embeddings = np.concatenate((embeddings, cls_output.data.cpu().numpy()),axis=0)


    # Save in a file
    logger.info('Embeddings shape %s' % str(embeddings.shape))
    fileout = os.path.join(args.data_dir, args.embsfile)
    if not os.path.exists(os.path.dirname(fileout)):
        os.mkdir(os.path.dirname(fileout))
    utils.save_obj(embeddings, fileout)
    logger.info('Embeddings saved into %s' % fileout)


if __name__ == "__main__":

    args = get_params()
    if not os.path.exists(args.embsfile):
        compute_embeddings(args)