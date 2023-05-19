import numpy as np
import torch
import torch.nn.functional as F
import dask.array as da

from sklearn.metrics import (accuracy_score,
                             top_k_accuracy_score,
                             recall_score,
                             precision_score,
                             f1_score,
                             average_precision_score,
                             roc_auc_score,
                             roc_curve,
                             auc)


def compute_class_metrics_dask(pred_class, target, num_classes,
                               pred_class_top=None):
    metrics_dict = {}

    if num_classes > 1:
        p = target.size
        n = 0

        tp = np.sum(pred_class == target).compute()
        tn = 0

        fp = p - tp
        fn = 0

        tp_top = np.sum(np.any(pred_class_top
                               == target.reshape(-1, 1), axis=1)).compute()
        tn_top = 0

    else:
        target = target > 0.5

        tp = np.bitwise_and(pred_class, target).sum().compute()
        tp_top = tp

        tn = np.sum(np.bitwise_and((1 - pred_class), (1 - target))).compute()
        tn_top = tn

        fp = np.sum(np.bitwise_and(pred_class, (1 - target))).compute()
        fn = np.sum(np.bitwise_and((1 - pred_class), target)).compute()

        p = np.sum(target).compute()
        n = target.size - p

    if (tp + tn + fp + fn) > 0:
        acc = (tp + tn) / (tp + tn + fp + fn)
        acc_top = (tp_top + tn_top) / (tp + tn + fp + fn)
    else:
        acc = float('nan')
        acc_top = float('nan')

    prec = (tp / (tp + fp)) if (tp + fp) > 0 else float('nan')
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else float('nan')
    f1 = (2 * tp / (2 * tp + fp + fn))

    metrics_dict['tp'] = tp
    metrics_dict['tp_top'] = tp_top
    metrics_dict['tn'] = tn
    metrics_dict['fp'] = fp
    metrics_dict['fn'] = fn

    metrics_dict['p'] = p
    metrics_dict['n'] = n

    metrics_dict['acc'] = acc
    metrics_dict['top_acc'] = acc_top
    metrics_dict['prec'] = prec
    metrics_dict['rec'] = rec
    metrics_dict['f1'] = f1

    return metrics_dict


def compute_class_metrics(pred_class, target, top_k, num_classes,
                          labels=None,
                          pred_scores=None,
                          pred_class_top=None,
                          one_hot_target=None):
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

        if (metrics_dict['p'] > 0
          and metrics_dict['n'] > 0
          and target.shape[0] > 1):
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


def compute_metrics_per_image(pred, target, top_k=5, num_classes=None,
                              seg_threshold=0.90):
    if num_classes is None:
        num_classes = pred.size(1)

    top_k = min(top_k, num_classes)

    if pred.ndim == 4:
        if target.size(1) > 1 and num_classes == 1:
            target = target[:, 1:]

        target = target.cpu().permute(0, 2, 3, 1).reshape(-1, num_classes)

        pred = pred.cpu().detach().permute(0, 2, 3, 1).reshape(-1, num_classes)

    if num_classes > 1:
        labels = range(num_classes)
        pred_scores = torch.softmax(pred.detach(), dim=1).cpu()
        pred_class_top = torch.topk(pred_scores, k=top_k)[1]
        pred_class = torch.argmax(pred.detach(), dim=1).cpu()
        one_hot_target = F.one_hot(target.cpu(), num_classes)

    else:
        labels = None
        pred_scores = torch.sigmoid(pred.detach()).cpu()
        pred_class_top = None
        pred_class = pred_scores > seg_threshold
        one_hot_target = None

    # Cast pytorch tensors into numpy arrays
    pred_scores = pred_scores.numpy()

    if pred_class_top is not None:
        pred_class_top = pred_class_top.numpy()

    if pred_class is not None:
        pred_class = pred_class.numpy()

    if one_hot_target is not None:
        one_hot_target = one_hot_target.numpy()

    target = target.cpu().numpy()

    return compute_class_metrics(pred_class, target, top_k, num_classes,
                                 labels,
                                 pred_scores,
                                 pred_class_top,
                                 one_hot_target)


def compute_roc_curve(pred_scores, target):
    fpr, tpr, thrsh = roc_curve(target, pred_scores, pos_label=None,
                            sample_weight=None,
                            drop_intermediate=True)

    roc_auc = auc(fpr, tpr)

    fpr = fpr.astype(np.float32)
    tpr = tpr.astype(np.float32)
    thrsh = thrsh.astype(np.float32)

    return fpr, tpr, thrsh, roc_auc
