import numpy as np

def dice_coefficients(label1, label2, labels=None):
    if labels is None:
        labels = np.unique(np.hstack((np.unique(label1), np.unique(label2))))
    dice_coefs = []
    for label in labels:
        match1 = (label1 == label)
        match2 = (label2 == label)
        denominator = 0.5 * (np.sum(match1.astype(np.float)) + np.sum(match2.astype(np.float)))
        numerator = np.sum(np.logical_and(match1, match2).astype(np.float))
        if denominator == 0:
            dice_coefs.append(0.)
        else:
            dice_coefs.append(numerator / denominator)
    return dice_coefs   


