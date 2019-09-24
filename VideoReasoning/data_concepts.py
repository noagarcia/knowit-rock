import torch
import torch.utils.data as data
import pandas as pd
import csv
import numpy as np
import os
import ast

import sys
sys.path.insert(0,'.')
import utils


import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


csv.field_size_limit(sys.maxsize)


class KnowITConceptsData(data.Dataset):

    def __init__(self, args, split):

        # Params
        self.args = args
        self.split = split

        # Load Data
        if self.split == 'train':
            textfile = args.data_dir + args.csvtrain
            filebert = args.data_dir + args.bertembds_ftrain
        elif self.split == 'val':
            textfile = args.data_dir + args.csvval
            filebert = args.data_dir + args.bertembds_fval
        elif self.split == 'test':
            textfile = args.data_dir + args.csvtest
            filebert = args.data_dir + args.bertembds_ftest
        df = pd.read_csv(textfile, delimiter='\t')
        logger.info('File with %d samples' % len(df))
        self.num_samples = len(df)
        self.bert_embds = utils.load_obj(filebert)
        self.framepaths = utils.get_frame_paths('', df, numframes=args.numframes)
        self.labels = df['idxCorrect']

        # Size dataset
        self.num_samples = len(df)

        # Load concepts
        self.ind2obj, self.obj2ind = self.load_vcpt_lists()
        self.objfeat = self.load_vcps(os.path.join(args.data_dir,args.vcpsframes))

        # Image paths to image ids
        with open(os.path.join(args.data_dir, args.idsframes), mode='r') as infile:
            reader = csv.reader(infile, delimiter='\t')
            next(reader)  # skip headder
            self.paths2ids = {rows[0]: int(rows[1]) for rows in reader}


    def get_num_objects(self):
        return len(self.obj2ind)


    def load_vcps(self, datafile):

        FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features', 'concepts']
        data = {}
        with open(datafile, "r") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                featFrame = {}
                featFrame['image_id'] = int(item['image_id'])
                featFrame['concepts'] = ast.literal_eval(item['concepts'])
                data[featFrame['image_id']] = featFrame
        return data


    def load_vcpt_lists(self):

        ind2obj = []
        with open(os.path.join(self.args.data_dir, self.args.list_vcps_objs)) as f:
            for object in f.readlines():
                ind2obj.append(object.split(',')[0].lower().strip())
        ind2obj.insert(0, 'unk')
        obj2ind = {c : i for i, c in enumerate(ind2obj)}
        return ind2obj, obj2ind


    def string2list(self, s):
        s = s.replace('[', '').replace(']', '')
        li = list(s.split(" "))
        return li


    def __len__(self):
        return self.num_samples


    def __getitem__(self, index):

        # Get concepts for this clip
        framepath = self.framepaths[index]
        vcps_count = np.zeros((len(self.obj2ind)), dtype=np.float32)
        for path in framepath:
            concepts = self.objfeat[self.paths2ids[path]]['concepts']
            for concept in concepts:
                # use only last word from each sentence visual concept
                word = concept.split()[-1]
                if word in self.obj2ind:
                    idx_word = self.obj2ind[word]
                else:
                    idx_word = self.obj2ind['unk']
                vcps_count[idx_word] += 1
        vcps_count = torch.from_numpy(vcps_count)

        # Get text
        embds1 = self.bert_embds[index,0]
        embds2 = self.bert_embds[index, 1]
        embds3 = self.bert_embds[index, 2]
        embds4 = self.bert_embds[index, 3]

        # Label
        label = self.labels[index] - 1

        # Data
        return [vcps_count, embds1, embds2, embds3, embds4], [label, index]
