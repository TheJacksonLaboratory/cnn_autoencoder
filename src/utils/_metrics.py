from functools import reduce
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score,
                             top_k_accuracy_score,
                             recall_score,
                             precision_score,
                             f1_score,
                             average_precision_score)


def compute_class_metrics(pred, target, top_k=5, num_classes=None):
    pred_scores = torch.softmax(pred.detach(), dim=1).cpu().numpy()
    pred_class = torch.argmax(pred.detach(), dim=1).cpu().numpy()
    one_hot_target = F.one_hot(target.cpu(), num_classes).numpy()

    target = target.cpu().numpy()
    labels = range(num_classes)

    acc = accuracy_score(target, pred_class)
    acc_top = top_k_accuracy_score(target, pred_scores, k=top_k, labels=labels)
    rec = recall_score(target, pred_class, average='micro')
    prec = precision_score(target, pred_class, average='micro')
    f1 = f1_score(target, pred_class, average='micro')
    avg_prec = average_precision_score(one_hot_target, pred_scores,
                                       average='micro')

    return dict(acc=acc, acc_top=acc_top, rec=rec, prec=prec, f1=f1,
                avg_prec=avg_prec)
