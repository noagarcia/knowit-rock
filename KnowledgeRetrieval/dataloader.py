import torch.utils.data as data
import os
import torch
import numpy as np
import utils
import pandas as pd


import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class QuestionKnowledgePairs(object):
    def __init__(self, id_q, question, answer1, answer2, answer3, answer4, reason, label):
        self.id_q = id_q
        self.question = question
        self.reason = reason
        self.label = label
        self.answers = [
            answer1,
            answer2,
            answer3,
            answer4,
        ]


def truncate_seq_pair(tokens_a, tokens_b, max_length):

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class RetrievalDataset(data.Dataset):

    def __init__(self, args, split, tokenizer):

        # Params
        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length

        # Load Data
        if self.split == 'train':
            input_file = os.path.join(args.data_dir, args.csvtrain)
            filekbsplit = os.path.join(args.data_dir, 'KB/reason_idx_to_kb_train.pckl')
            filescores = os.path.join(args.data_dir, 'PriorScores/priorscores_answers_train.pckl')
        elif self.split == 'val':
            input_file = os.path.join(args.data_dir, args.csvval)
            filekbsplit = os.path.join(args.data_dir, 'KB/reason_idx_to_kb_val.pckl')
            filescores = os.path.join(args.data_dir, 'PriorScores/priorscores_answers_val.pckl')
        elif self.split == 'test':
            input_file = os.path.join(args.data_dir, args.csvtest)
            filekbsplit = os.path.join(args.data_dir, 'KB/reason_idx_to_kb_test.pckl')
            filescores = os.path.join(args.data_dir, 'PriorScores/priorscores_answers_test.pckl')

        df = pd.read_csv(input_file, delimiter='\t')
        logger.info('Loaded file with %d samples' % len(df))
        self.scores = utils.load_obj(filescores)

        # Load KB
        self.kb = utils.load_obj(os.path.join(args.data_dir, 'KB/reason_kb_dict.pckl'))
        self.idx_to_kb = utils.load_obj(os.path.join(args.data_dir, 'KB/reason_idx_to_kb.pckl'))
        self.idx_to_kb_this = utils.load_obj(filekbsplit)

        # Prepapre pairs
        self.data_pairs = self.get_samples_pais(df)
        self.num_samples = len(self.data_pairs)
        logger.info('Dataloader with %d samples' % self.num_samples)



    def get_samples_pais(self, df):

        pairs = []
        for idx_q in list(range(len(df))):

            # Question info
            question = df['question'].iloc[idx_q]
            cluster_pos = self.idx_to_kb_this[idx_q]

            # Order answer according to the score prior
            ansprior = self.scores[idx_q]
            idxsort = np.argsort(ansprior)[::-1]
            answer1 = df['answer{}'.format(idxsort[0] + 1)].iloc[idx_q]
            answer2 = df['answer{}'.format(idxsort[1] + 1)].iloc[idx_q]
            answer3 = df['answer{}'.format(idxsort[2] + 1)].iloc[idx_q]
            answer4 = df['answer{}'.format(idxsort[3] + 1)].iloc[idx_q]

            # at training time we pick num_pairs reasons for each question
            if self.split == 'train':

                # Matching knowledge
                reason_pos = self.kb[cluster_pos]
                pairs.append(QuestionKnowledgePairs(id_q=idx_q,
                                                    question=question,
                                                    reason=reason_pos,
                                                    answer1=answer1,
                                                    answer2=answer2,
                                                    answer3=answer3,
                                                    answer4=answer4,
                                                    label=1))

                # Non-matching knowledge
                all_idx = list(range(len(self.kb)))
                for _ in list(range(self.args.num_pairs - 1)):
                    id_rneg = np.random.choice(all_idx)
                    cluster_neg = self.idx_to_kb[id_rneg]
                    while cluster_neg == cluster_pos:
                        id_rneg = np.random.choice(all_idx)
                        cluster_neg = self.idx_to_kb[id_rneg]

                    reason_neg = self.kb[cluster_neg]
                    pairs.append(QuestionKnowledgePairs(id_q=idx_q,
                                                        question=question,
                                                        reason=reason_neg,
                                                        answer1=answer1,
                                                        answer2=answer2,
                                                        answer3=answer3,
                                                        answer4=answer4,
                                                        label=0))

            # at val/test time we match all questions with all reasons
            else:

                for cluster_res in self.kb:

                    if cluster_res == cluster_pos:
                        label = 1
                    else:
                        label = 0

                    reason_neg = self.kb[cluster_res]
                    pairs.append(QuestionKnowledgePairs(id_q=idx_q,
                                                        question=question,
                                                        reason=reason_neg,
                                                        answer1=answer1,
                                                        answer2=answer2,
                                                        answer3=answer3,
                                                        answer4=answer4,
                                                        label=label))
        return pairs


    def __len__(self):

        return self.num_samples


    def __getitem__(self, index):

        # Get pair data sample
        sample = self.data_pairs[index]

        # Convert to BERT input data
        question_tokens = self.tokenizer.tokenize(sample.question)
        answer_tokens = self.tokenizer.tokenize(sample.answers[0]) + self.tokenizer.tokenize(sample.answers[1]) + \
                        self.tokenizer.tokenize(sample.answers[2]) + self.tokenizer.tokenize(sample.answers[3])
        query_tokens = question_tokens + answer_tokens
        knowledge_tokens = self.tokenizer.tokenize(sample.reason)

        truncate_seq_pair(query_tokens, knowledge_tokens, self.max_seq_length - 3)
        tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + knowledge_tokens + ["[SEP]"]
        segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(knowledge_tokens) + 1)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        id_q = torch.tensor(sample.id_q, dtype=torch.long)
        label = torch.tensor(sample.label, dtype=torch.long)

        return input_ids, input_mask, segment_ids, id_q, label
