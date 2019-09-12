import os
import argparse
import logging
import numpy as np
import pandas as pd
import random
import sklearn.metrics

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
    parser.add_argument('--csvtrain', default='data_full_train.csv', help='Training set data file')
    parser.add_argument('--csvval', default='data_full_val.csv', help='Dataset val data file')
    parser.add_argument('--csvtest', default='data_full_test_qtypes.csv', help='Dataset test data file')
    parser.add_argument('--embsfile', default='cache/kb_allreasons_60len.pckl')
    return parser


def read_data(args):
    """Loads all reasons in the dataset into a list."""

    files = [args.csvtrain, args.csvval, args.csvtest]
    allreasons = []

    for f in files:
        filename = os.path.join(args.data_dir, f)
        df = pd.read_csv(filename, delimiter='\t')
        allreasons += [row['reason'] for index, row in df.iterrows()]

    return allreasons


def show_examples(args):

    # Load reason instances and embeddings
    fileembds = os.path.join(args.data_dir, args.embsfile)
    embeddings = utils.load_obj(fileembds)
    allreasons = read_data(args)
    allinds = list(range(len(allreasons)))
    assert len(allreasons) == embeddings.shape[0]
    numReasons = len(allreasons)

    # Get a random instance, compute scores and show most similar
    thisidx = random.sample(allinds, 1)[0]
    thisreason = allreasons[thisidx]
    print('-' * 25)
    print("REASON: {}".format(thisidx))
    print(thisreason)
    print('.')

    # Compute scores and sort
    allscores = sklearn.metrics.pairwise.cosine_similarity(embeddings)
    thisscores = allscores[thisidx,:]
    ranking = np.argsort(thisscores)[::-1].tolist()
    sortedscores = np.sort(thisscores)[::-1].tolist()

    # show top 5
    numshow = 10
    print("MATCHES")
    for k in list(range(numshow)):
        kidx = ranking[k]
        score = sortedscores[k]
        reason = allreasons[kidx]
        print("sample %d, score %.03f: %s" % (kidx, score, reason))

    return allscores


def connected_components(connGraph):

    result = []
    nodes = set(connGraph.keys())
    while nodes:
        n = nodes.pop()

        # Set to contain nodes in this connected group
        group = {n}

        # Find neighbours, add neighbours to queue and remove from global set.
        queue = [n]
        while queue:
            n = queue.pop(0)
            neighbors = set(connGraph[n])
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)

        # Add the group to the list of groups.
        result.append(group)

    return result


def compute_scores(embeddings):

    allscores = sklearn.metrics.pairwise.cosine_similarity(embeddings)
    return allscores


def find_connected_samples(allscores):

    numReasons = allscores.shape[0]

    # remove scores lower than threshold
    th = 0.998
    allscores[allscores < th] = 0

    # create a graph with the scores as edges
    connGraph = {}
    for idxSample in list(range(numReasons)):
        thisEdges = allscores[idxSample,:]
        connGraph[idxSample] = list(np.nonzero(thisEdges)[0])

    # Get all the connected graphs
    components = connected_components(connGraph)
    return components


def process(args):

    logger.info('Loading data...')
    fileembds = os.path.join(args.data_dir, args.embsfile)
    embeddings = utils.load_obj(fileembds)
    allreasons = read_data(args)
    numReasons = len(allreasons)

    logger.info('Computing scores...')
    allscores = compute_scores(embeddings)

    logger.info('Finding components...')
    components = find_connected_samples(allscores)

    # Dictionary from index of group to reason sentence
    logger.info('Creating kb dictionary...')
    kb = {}
    for i, c in enumerate(components):
        idxReason = list(c)[0]
        reason = allreasons[idxReason]
        kb['group%d' % i] = reason

    # Dictionary from index sample to index of group
    logger.info('Creating kb indexing...')
    idx_to_kb = ['group%d' %gidx for i in list(range(numReasons)) for gidx, c in enumerate(components) if i in list(c)]

    # Dictionaries for each split
    dftrain = pd.read_csv(os.path.join(args.data_dir, args.csvtrain), delimiter='\t')
    num_reason_train = len(dftrain)
    idx_to_kb_train = idx_to_kb[0:num_reason_train]

    dfval = pd.read_csv(os.path.join(args.data_dir, args.csvval), delimiter='\t')
    num_reason_val = len(dfval)
    idx_to_kb_val = idx_to_kb[num_reason_train:num_reason_train+num_reason_val]

    dftest = pd.read_csv(os.path.join(args.data_dir, args.csvtest), delimiter='\t')
    num_reason_test = len(dftest)
    idx_to_kb_test = idx_to_kb[num_reason_train+num_reason_val:num_reason_train+num_reason_val+num_reason_test]

    # save
    filedir = os.path.join(args.data_dir, 'KB/')
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    logger.info('Saving to %s...' % filedir)

    utils.save_obj(kb, os.path.join(filedir, 'reason_kb_dict.pckl'))
    utils.save_obj(idx_to_kb, os.path.join(filedir, 'reason_idx_to_kb.pckl'))
    utils.save_obj(idx_to_kb_train, os.path.join(filedir, 'reason_idx_to_kb_train.pckl'))
    utils.save_obj(idx_to_kb_val, os.path.join(filedir, 'reason_idx_to_kb_val.pckl'))
    utils.save_obj(idx_to_kb_test, os.path.join(filedir, 'reason_idx_to_kb_test.pckl'))
    logger.info('Done!')


if __name__ == "__main__":

    np.set_printoptions(precision=3,linewidth=200)
    parser = get_params()
    args, unknown = parser.parse_known_args()

    process(args)