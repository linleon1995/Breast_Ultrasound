import numpy as np

# class evaluator():
#     def __init__(pred, target):
#         # TODO: check input shape
#         self.tp = ((prediction.data == 1) & (labels.data == 1)).cpu().sum()


def precision(ref=None, test=None, **metrics_kwargs):
    total_cm = metrics_kwargs.pop("total_cm")
    sum_over_row = np.sum(total_cm, axis=0).astype(float)
    cm_diag = np.diagonal(total_cm).astype(float)
    precision = cm_diag / sum_over_row
    return precision

def recall(ref=None, test=None, **metrics_kwargs):
    total_cm = metrics_kwargs.pop("total_cm")
    sum_over_col = np.sum(total_cm, axis=1).astype(float)
    cm_diag = np.diagonal(total_cm).astype(float)
    precision = cm_diag / sum_over_col
    return precision

def F1_score(prediction, label):
    pass

def Dice_score(prediction, label):
    pass

def sensitivity(prediction, label):
    pass

def specificity(prediction, label):
    pass

# TODO: w/ label and w/o label
# TODO: multi-classes example