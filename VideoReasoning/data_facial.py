import torch
import torch.utils.data as data
import pandas as pd
import utils
import csv
import numpy as np
import os
import ast

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class KnowITFacesData(data.Dataset):

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

        # Faces per frame
        _, self.pers2ind = self.load_faces_list(os.path.join(args.data_dir, args.list_faces_names))
        self.df_faces = pd.read_csv(os.path.join(args.data_dir, args.facesframes), delimiter='\t')

        # Image paths to image ids
        with open(os.path.join(args.data_dir, args.idsframes), mode='r') as infile:
            reader = csv.reader(infile, delimiter='\t')
            next(reader)  # skip headder
            self.paths2ids = {rows[0]: int(rows[1]) for rows in reader}


    def get_num_people(self):
        return len(self.pers2ind)


    def load_faces_list(self, datafile):
        ind2pers = []
        with open(datafile) as f:
            for person in f.readlines():
                ind2pers.append(person.split(',')[0].lower().strip())
        ind2pers.insert(0, 'unk')
        pers2ind = {c: i for i, c in enumerate(ind2pers)}
        return ind2pers, pers2ind


    def __len__(self):
        return self.num_samples


    def __getitem__(self, index):

        # Get faces for this clip
        framepath = self.framepaths[index]
        pep_count = np.zeros((len(self.pers2ind)), dtype=np.float32)
        for path in framepath:
            peplist = ast.literal_eval(self.df_faces[self.df_faces['frame_path'] == path]['people'].iloc[0])
            for face in peplist:
                face = face.lower()
                if face in self.pers2ind:
                    idx_face = self.pers2ind[face]
                else:
                    idx_face = self.pers2ind['unk']
                pep_count[idx_face] += 1
        pep_count = torch.from_numpy(pep_count)

        # Get text
        embds1 = self.bert_embds[index,0]
        embds2 = self.bert_embds[index, 1]
        embds3 = self.bert_embds[index, 2]
        embds4 = self.bert_embds[index, 3]

        # Label
        label = self.labels[index] - 1

        # Output data
        return [pep_count, embds1, embds2, embds3, embds4], [label, index]

