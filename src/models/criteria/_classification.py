from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss


class ClassLoss(object):
    def __call__(self, pred, t, **kwargs):
        return dict(class_error=self._loss(pred, t),
                    aux_class_error=0)


class ClassLossWithAux(object):
    def __call__(self, pred, t, aux_pred, **kwargs):
        return dict(class_error=self._loss(pred, t),
                    aux_class_error=self._aux_loss(aux_pred, t))


class CEClassLoss(ClassLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None,
                 reduction='mean',
                 label_smoothing=0.0,
                 **kwargs):
        self._loss = CrossEntropyLoss(weight, size_average, ignore_index,
                                      reduce,
                                      reduction,
                                      label_smoothing)


class BCEClassLoss(ClassLoss):
    def __init__(self, weight=None, size_average=None, reduce=None,
                 reduction='mean',
                 pos_weight=None,
                 **kwargs):
        self._loss = BCEWithLogitsLoss(weight, size_average, reduce, reduction,
                                       pos_weight)


class CEClassLossWithAux(ClassLossWithAux):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None,
                 reduction='mean',
                 label_smoothing=0.0,
                 **kwargs):
        self._loss = CrossEntropyLoss(weight, size_average, ignore_index,
                                      reduce,
                                      reduction,
                                      label_smoothing)
        self._aux_loss = CrossEntropyLoss(weight, size_average, ignore_index,
                                          reduce,
                                          reduction,
                                          label_smoothing)


class BCEClassLossWithAux(ClassLossWithAux):
    def __init__(self, weight=None, size_average=None, reduce=None,
                 reduction='mean',
                 pos_weight=None,
                 **kwargs):
        self._loss = BCEWithLogitsLoss(weight, size_average, reduce, reduction,
                                       pos_weight)
        self._aux_loss = BCEWithLogitsLoss(weight, size_average, reduce,
                                           reduction,
                                           pos_weight)


CLASSLOSS_LIST = {
    "CELoss": CEClassLoss,
    "BCELoss": BCEClassLoss,
    "CELossWithAux": CEClassLossWithAux,
    "BCELossWithAux": BCEClassLossWithAux,
}