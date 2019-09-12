import numpy as np


def rank(scores1, scores2, idx_q, labels):

    # Accuracy variables
    med_rank1 = []
    recall1 = {1: 0.0, 5: 0.0, 10: 0.0, 100: 0.0}

    med_rank2 = []
    recall2 = {1: 0.0, 5: 0.0, 10: 0.0, 100: 0.0}

    unique_idxs = np.unique(idx_q)
    N = len(unique_idxs)

    for iq in unique_idxs:

        pos_iq = np.where(idx_q  == iq)
        this_labels = labels[pos_iq]
        ir = np.where(this_labels == 1)[0]
        this_scores1 = scores1[pos_iq]
        this_scores2 = scores2[pos_iq]

        ranking1 = np.argsort(this_scores1).tolist()
        ranking2 = np.argsort(this_scores2)[::-1].tolist()

        # position of idx in ranking
        pos1 = ranking1.index(ir)
        if (pos1 + 1) == 1:
            recall1[1] += 1
        if (pos1 + 1) <= 5:
            recall1[5] += 1
        if (pos1 + 1) <= 10:
            recall1[10] += 1
        if (pos1 + 1) <= 100:
            recall1[100] += 1

        # store the position
        med_rank1.append(pos1 + 1)

        pos2 = ranking2.index(ir)
        if (pos2 + 1) == 1:
            recall2[1] += 1
        if (pos2 + 1) <= 5:
            recall2[5] += 1
        if (pos2 + 1) <= 10:
            recall2[10] += 1
        if (pos2 + 1) <= 100:
            recall2[100] += 1

        # store the position
        med_rank2.append(pos1 + 1)

    for i in recall1.keys():
        recall1[i] = recall1[i] / N

    for i in recall2.keys():
        recall2[i] = recall2[i] / N

    return np.median(med_rank1), recall1, np.median(med_rank2), recall2