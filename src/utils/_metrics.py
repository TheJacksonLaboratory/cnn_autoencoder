import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score,
                             top_k_accuracy_score,
                             recall_score,
                             precision_score,
                             f1_score,
                             average_precision_score,
                             roc_auc_score)


def compute_class_metrics(pred, target, top_k=5, num_classes=None,
                          seg_threshold=0.90):
    if num_classes is None:
        num_classes = pred.size(1)

    top_k = min(top_k, num_classes)

    if pred.ndim == 4:
        target = target.cpu().permute(0, 2, 3, 1).reshape(-1, num_classes)
        pred = pred.cpu().detach().permute(0, 2, 3, 1).reshape(-1, num_classes)

    if num_classes > 1:
        labels = range(num_classes)
        pred_scores = torch.softmax(pred.detach(), dim=1).cpu()
        pred_class_top = torch.topk(pred_scores, k=top_k)[1].numpy()
        pred_scores = pred_scores.numpy()
        pred_class = torch.argmax(pred.detach(), dim=1).cpu().numpy()
        one_hot_target = F.one_hot(target.cpu(), num_classes).numpy()

    else:
        pred_scores = torch.sigmoid(pred.detach()).cpu().numpy()
        pred_class = pred_scores > seg_threshold

    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    metrics_dict = {}

    if num_classes > 1:
        metrics_dict['acc_top'] = top_k_accuracy_score(target, pred_scores,
                                                       k=top_k,
                                                       labels=labels)
        metrics_dict['avg_prec'] = average_precision_score(one_hot_target,
                                                           pred_scores,
                                                           average='micro')

        tp = (pred_class == target).sum()
        tp_top = (pred_class_top == target.reshape(-1, 1)).any(axis=1).sum()

        metrics_dict['tp'] = tp
        metrics_dict['tp_top'] = tp_top

        metrics_dict['tn'] = 0
        metrics_dict['fp'] = target.size - tp
        metrics_dict['fn'] = target.size - tp

        metrics_dict['p'] = target.size
        metrics_dict['n'] = 0

        metrics_dict['acc'] = accuracy_score(target, pred_class)
        metrics_dict['rec'] = recall_score(target, pred_class, average='micro')
        metrics_dict['prec'] = precision_score(target, pred_class,
                                               average='micro')
        metrics_dict['f1'] = f1_score(target, pred_class, average='micro')

    else:
        target = target > 0.5

        tp = np.bitwise_and(pred_class, target).sum()
        metrics_dict['tp'] = tp
        metrics_dict['tp_top'] = tp

        metrics_dict['tn'] = np.bitwise_and((1 - pred_class),
                                            (1 - target)).sum()
        metrics_dict['fp'] = np.bitwise_and(pred_class, (1 - target)).sum()
        metrics_dict['fn'] = np.bitwise_and((1 - pred_class), target).sum()

        metrics_dict['p'] = target.sum()
        metrics_dict['n'] = target.size - metrics_dict['p']

        if metrics_dict['p'] > 0:
            metrics_dict['auc'] = roc_auc_score(target, pred_scores[:, 0])
        else:
            metrics_dict['auc'] = float('nan')

        metrics_dict['acc'] = accuracy_score(target, pred_class)
        metrics_dict['rec'] = recall_score(target, pred_class,
                                           average='binary',
                                           zero_division=0)
        metrics_dict['prec'] = precision_score(target, pred_class,
                                               average='binary',
                                               zero_division=0)
        metrics_dict['f1'] = f1_score(target, pred_class, average='binary',
                                      zero_division=0)

    return metrics_dict
