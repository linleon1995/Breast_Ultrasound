import numpy as np
import torch

# TODO: check correctness and multi-class solution
def dsc(prediction, label):
    tp = ((prediction.data == 1) & (label.data == 1)).sum()
    fn = ((prediction.data == 0) & (label.data == 1)).sum()
    fp = ((prediction.data == 1) & (label.data == 0)).sum()
    denominator = 2*tp + fp +fn
    if denominator == 0:
        return 1
    else:
        return 2*tp / denominator

def sensitivity(prediction, label):
    pass

def specificity(prediction, label):
    pass

# TODO: w/ label and w/o label
# TODO: multi-classes example


def precision(tp, fp):
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fn):
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0

# TODO: property for all avaiable metrics
class SegmentationMetrics():
    def __init__(self, metrics=None):
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = ['precision', 'recall', 'accuracy', 'f1']
    
    def __call__(self, label, pred):
        self.label = label
        self.pred = pred
        self.tp, self.fp, self.fn, self.fp = self.confusion_matrix()
        eval_result = {}
        for m in self.metrics:
            if m == 'precision':
                eval_result[m] = precision(self.tp, self.fp)
            elif m == 'recall':
                eval_result[m] = recall(self.tp, self.fn)
            elif m == 'accuracy':
                eval_result[m] = accuracy(self.tp, self.fp, self.fn)
            elif m == 'f1':
                eval_result[m] = f1(self.tp, self.fp, self.fn)
        return eval_result

    def confusion_matrix(self):
        tp = ((self.pred.data == 1) & (self.label.data == 1)).sum()
        tn = ((self.pred.data == 0) & (self.label.data == 0)).sum()
        fn = ((self.pred.data == 0) & (self.label.data == 1)).sum()
        fp = ((self.pred.data == 1) & (self.label.data == 0)).sum()
        return (tp, tn, fn, fp)

