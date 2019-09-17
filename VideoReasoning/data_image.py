import torch.utils.data as data
import pandas as pd
from PIL import Image
import utils

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class KnowITImageData(data.Dataset):

    def __init__(self, args, split, transform = None):

        # Params
        self.args = args
        self.split = split
        self.transform = transform

        # Load Data
        if self.split == 'train':
            csvfile = args.data_dir + args.csvtrain
            filebertembs = args.data_dir + args.bertembds_ftrain
        elif self.split == 'val':
            csvfile = args.data_dir + args.csvval
            filebertembs = args.data_dir + args.bertembds_fval
        elif self.split == 'test':
            csvfile = args.data_dir + args.csvtest
            filebertembs = args.data_dir + args.bertembds_ftest
        df = pd.read_csv(csvfile, delimiter='\t')
        logger.info('Data file loaded with %d samples' % len(df))
        self.bert_embds = utils.load_obj(filebertembs)
        self.framepaths = self.get_frame_paths(args.framesdir, df, numframes=args.numframes)
        self.labels = df['idxCorrect']

        # Size dataset
        self.num_samples = len(df)

    def get_frame_paths(self, basedir, df, numframes):
        scenes = df['scene'].str.split('_')
        paths = []
        for s in scenes:
            start = int(s[2])
            end = int(s[3])
            step = int((end - start) / numframes)
            frame_paths = [basedir + s[0] + '/frame_' + str(start + step * (n + 1)).zfill(4) + '.jpeg' for n in
                           list(range(numframes))]
            paths.append(frame_paths)
        return paths

    def __len__(self):
        return self.num_samples


    def __getitem__(self, index):

        # Frames
        framepath = self.framepaths[index]
        framelist = []
        for path in framepath:
            frame = Image.open(path).convert('RGB')
            if self.transform is not None:
                frame = self.transform(frame)
                framelist.append(frame)

        # Language
        embds1 = self.bert_embds[index,0]
        embds2 = self.bert_embds[index, 1]
        embds3 = self.bert_embds[index, 2]
        embds4 = self.bert_embds[index, 3]

        # Label
        label = self.labels[index] - 1

        # Data
        return [*framelist, embds1, embds2, embds3, embds4], [label, index]