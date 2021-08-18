import importlib
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
# TODO: input output type --> numpy or tensor
class SegmentationMetrics():
    def __init__(self, metrics=None):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = ['precision', 'recall', 'accuracy', 'f1']
        
    def __call__(self, label, pred):
        self.label = label
        self.pred = pred
        self.tp, self.fp, self.fn, self.fp = self.confusion_matrix()
        self.total_tp += self.tp
        self.total_fp += self.fp
        self.total_fn += self.fn
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
        # # print(self.pred==1 & self.label==0)
        # print(np.sum(self.label))
        # print(np.sum(self.pred))
        # tp = np.sum(self.pred * self.label)
        # print(tp)
        # print('p', tp / np.sum(self.pred), 'r', tp / np.sum(self.label))
        # # tn = np.sum(np.int32(self.pred==0 & self.label==0))
        # # fp = np.sum(np.int32(self.pred==0 & self.label==1))
        # # fn = np.sum(np.int32(self.pred==1 & self.label==0))

        tp = ((self.pred.data == 1) & (self.label.data == 1)).sum()
        tn = ((self.pred.data == 0) & (self.label.data == 0)).sum()
        fn = ((self.pred.data == 0) & (self.label.data == 1)).sum()
        fp = ((self.pred.data == 1) & (self.label.data == 0)).sum()
        return (tp, tn, fn, fp)


def get_evaluation_metric(config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    def _metric_class(class_name):
        m = importlib.import_module('utils.metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)

    