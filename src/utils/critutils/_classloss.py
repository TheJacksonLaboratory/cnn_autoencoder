import torch
from torch.nn import (CrossEntropyLoss,
                      BCELoss,
                      BCEWithLogitsLoss,
                      L1Loss,
                      MSELoss)


class ClassLoss(object):
    def __call__(self, pred, t, **kwargs):
        return dict(class_error=self._loss_impl(pred, t.to(pred.device)),
                    aux_class_error=0)


class WeightedClassLoss(object):
    def __call__(self, pred, t, **kwargs):
        loss = self._loss_impl(pred, t[:, 1:].to(pred.device))
        loss = torch.mean(t[:, :1].to(pred.device) * loss)
        return dict(class_error=loss,
                    aux_class_error=0)


class ClassLossWithAux(object):
    def __call__(self, pred, t, aux_pred, **kwargs):
        class_error = self._loss_impl(pred, t.to(pred.device))
        if aux_pred is not None:
            aux_class_error = self._aux_loss(aux_pred, t.to(pred.device))
        else:
            aux_class_error = 0

        return dict(class_error=class_error, aux_class_error=aux_class_error)


class CEClassLoss(ClassLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None,
                 reduction='mean',
                 label_smoothing=0.0,
                 gpu=False,
                 **kwargs):
        self._loss = CrossEntropyLoss(weight, size_average, ignore_index,
                                      reduce,
                                      reduction,
                                      label_smoothing)
        if gpu:
            self._loss.cuda()

    def _loss_impl(self, pred, t):
        return self._loss(pred, t)


class BCEClassLoss(ClassLoss):
    def __init__(self, weight=None, size_average=None, reduce=None,
                 reduction='mean',
                 gpu=False,
                 **kwargs):
        self._loss = BCELoss(weight, size_average, reduce, reduction)
        if gpu:
            self._loss.cuda()

    def _loss_impl(self, pred, t):
        if t.ndim != pred.ndim:
            t = t.expand_as(pred)

        return self._loss(pred, t)


class BCELogClassLoss(ClassLoss):
    def __init__(self, weight=None, size_average=None, reduce=None,
                 reduction='mean',
                 pos_weight=None,
                 gpu=False,
                 **kwargs):
        self._loss = BCEWithLogitsLoss(weight, size_average, reduce, reduction,
                                       pos_weight)
        if gpu:
            self._loss.cuda()

    def _loss_impl(self, pred, t):
        if t.ndim != pred.ndim:
            t = t.expand_as(pred)

        return self._loss(pred, t)


class BCEWeightedClassLoss(WeightedClassLoss):
    def __init__(self, weight=None, size_average=None, reduce=None,
                 reduction='none',
                 gpu=False,
                 **kwargs):
        self._loss = BCELoss(weight, size_average, reduce, reduction)
        if gpu:
            self._loss.cuda()

    def _loss_impl(self, pred, t):
        if t.ndim != pred.ndim:
            t = t.expand_as(pred)

        return self._loss(pred, t)


class BCELogWeightedClassLoss(WeightedClassLoss):
    def __init__(self, weight=None, size_average=None, reduce=None,
                 reduction='none',
                 pos_weight=None,
                 gpu=False,
                 **kwargs):
        self._loss = BCEWithLogitsLoss(weight, size_average, reduce, reduction,
                                       pos_weight)
        if gpu:
            self._loss.cuda()

    def _loss_impl(self, pred, t):
        if t.ndim != pred.ndim:
            t = t.expand_as(pred)

        return self._loss(pred, t)


class CEClassLossWithAux(ClassLossWithAux):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None,
                 reduction='mean',
                 label_smoothing=0.0,
                 gpu=False,
                 **kwargs):
        self._loss = CrossEntropyLoss(weight, size_average, ignore_index,
                                      reduce,
                                      reduction,
                                      label_smoothing)
        self._aux_loss = CrossEntropyLoss(weight, size_average, ignore_index,
                                          reduce,
                                          reduction,
                                          label_smoothing)
        if gpu:
            self._loss.cuda()
            self._aux_loss.cuda()

    def _loss_impl(self, pred, t):
        return self._loss(pred, t)


class BCEClassLossWithAux(ClassLossWithAux):
    def __init__(self, weight=None, size_average=None, reduce=None,
                 reduction='mean',
                 pos_weight=None,
                 gpu=False,
                 **kwargs):
        self._loss = BCEWithLogitsLoss(weight, size_average, reduce, reduction,
                                       pos_weight)
        self._aux_loss = BCEWithLogitsLoss(weight, size_average, reduce,
                                           reduction,
                                           pos_weight)
        if gpu:
            self._loss.cuda()
            self._aux_loss.cuda()

    def _loss_impl(self, pred, t):
        if t.ndim != pred.ndim:
            t = t.expand_as(pred)

        return self._loss(pred, t)


class L1ClassLoss(ClassLoss):
    def __init__(self, gpu=False, **kwargs):
        self._loss = L1Loss()
        if gpu:
            self._loss.cuda()

    def _loss_impl(self, pred, t):
        return self._loss(pred, t)


class L2ClassLoss(ClassLoss):
    def __init__(self, gpu=False, **kwargs):
        self._loss = MSELoss()
        if gpu:
            self._loss.cuda()

    def _loss_impl(self, pred, t):
        return self._loss(pred, t)


class HingeClassLoss(ClassLoss):
    def __init__(self, **kwargs):
        pass

    def _loss_impl(self, pred, t):
        norm_target = 2 * t - 1
        norm_pred = 2 * pred - 1
        return torch.mean(torch.clamp(1 - norm_target * norm_pred,
                                      min=0))


CLASSLOSS_LIST = {
    "L1Loss": L1ClassLoss,
    "L2Loss": L2ClassLoss,
    "HingeLoss": HingeClassLoss,
    "CELoss": CEClassLoss,
    "BCELoss": BCEClassLoss,
    "BCELogLoss": BCELogClassLoss,
    "WeightedBCELoss": BCEWeightedClassLoss,
    "WeightedBCELogLoss": BCELogWeightedClassLoss,
    "CELossWithAux": CEClassLossWithAux,
    "BCELossWithAux": BCEClassLossWithAux,
}
