import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score,
                             top_k_accuracy_score,
                             recall_score,
                             precision_score,
                             f1_score,
                             average_precision_score,
                             multilabel_confusion_matrix)


def compute_class_metrics(pred, target, top_k=5, num_classes=None):
    if pred.shape == 2:
        if num_classes is None:
            num_classes = pred.size(1)
        labels = range(num_classes)
        pred_scores = torch.softmax(pred.detach(), dim=1).cpu().numpy()
        pred_class = torch.argmax(pred.detach(), dim=1).cpu().numpy()
        one_hot_target = F.one_hot(target.cpu(), num_classes).numpy()
    else:
        pred_class = pred
        labels = None

    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    metrics_dict = {}

    metrics_dict['acc'] = accuracy_score(target, pred_class)
    metrics_dict['rec'] = recall_score(target, pred_class, average='micro')
    metrics_dict['prec'] = precision_score(target, pred_class, average='micro')
    metrics_dict['f1'] = f1_score(target, pred_class, average='micro')

    if labels is not None:
        metrics_dict['acc_top'] = top_k_accuracy_score(target, pred_scores,
                                                       k=top_k,
                                                       labels=labels)
        metrics_dict['avg_prec'] = average_precision_score(one_hot_target,
                                                           pred_scores,
                                                           average='micro')

    return metrics_dict
