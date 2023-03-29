from ._ratedist import *
from ._classification import *


class GeneralLoss(nn.Module):
    def __init__(self, dist_loss_type="MSE", rate_loss_type="Rate",
                 penalty_loss_type=None,
                 class_loss_type=None,
                 distortion_lambda=0.1,
                 penalty_beta=0.001,
                 class_error_mu=1.0,
                 class_error_aux_mu=1.0,
                 **kwargs):
        super(GeneralLoss, self).__init__()

        if dist_loss_type is not None:
            assert dist_loss_type in DIST_LOSS_LIST
            self.dist_loss = DIST_LOSS_LIST[dist_loss_type](**kwargs)
            self._multiplier = 255 ** 2 if 'MSE' in dist_loss_type else 1

            if not isinstance(distortion_lambda, list):
                distortion_lambda = [distortion_lambda]

            self._distortion_lambda = distortion_lambda
            self.dist_loss_func = self._dist_loss
        else:
            self.dist_loss_func = self._empty_task_loss

        if rate_loss_type is not None:
            assert rate_loss_type in RATE_LOSS_LIST
            self.rate_loss = RATE_LOSS_LIST[rate_loss_type](**kwargs)
            self.rate_loss_func = self._rate_loss
        else:
            self.rate_loss_func = self._empty_task_loss

        if (penalty_loss_type is not None
          and penalty_loss_type.lower() != "none"):
            assert penalty_loss_type in PENALTY_LOSS_LIST
            self.penalty_loss = PENALTY_LOSS_LIST[dist_loss_type](**kwargs)
            self._penalty_beta = penalty_beta
            self.penalty_loss_func = self._penalty_loss
        else:
            self.penalty_loss_func = self._empty_task_loss

        if class_loss_type is not None and class_loss_type.lower() != "none":
            assert class_loss_type in CLASSLOSS_LIST
            self.class_loss = CLASSLOSS_LIST[class_loss_type](**kwargs)
            self._class_error_mu = class_error_mu
            self._class_error_aux_mu = class_error_aux_mu
            self.class_loss_func = self._class_loss
        else:
            self.class_loss_func = self._empty_task_loss

    def _dist_loss(self, loss_dict, inputs, outputs, **kwargs):
        loss_dict.update(self.dist_loss(x=inputs, x_r=outputs['x_r'],
                                        **kwargs))

        loss_dict['dist'] = [self._multiplier * d for d in loss_dict['dist']]

        loss_dict['dist_loss'] = reduce(lambda d1, d2: d1 + d2,
                                        map(lambda wd: wd[0] * wd[1],
                                            zip(loss_dict['dist'],
                                                self._distortion_lambda)))

        loss_dict['loss'] += loss_dict['dist_loss']

    def _rate_loss(self, loss_dict, inputs, outputs, net, **kwargs):
        loss_dict.update(self.rate_loss(x=inputs, p_y=outputs['p_y'],
                                        **kwargs))
        loss_dict['entropy_loss'] = net['fact_ent'].module.loss()

        loss_dict['loss'] += loss_dict['rate_loss']

    def _penalty_loss(self, loss_dict, inputs, outputs, net, **kwargs):
        loss_dict.update(self.penalty_loss(x=inputs, y=outputs['y'], 
                                           net=net['decoder'],
                                           **kwargs))

        loss_dict['loss'] += self._penalty_beta * loss_dict['weighted_penalty']

    def _class_loss(self, loss_dict, targets, outputs, **kwargs):
        if outputs.get('t_pred', None) is not None:
            pred = outputs['t_pred']
            aux_pred = outputs.get('t_aux_pred', None)
        else:
            pred = outputs['s_pred']
            aux_pred = outputs.get('s_aux_pred', None)

        loss_dict.update(self.class_loss(pred=pred, aux_pred=aux_pred,
                                         t=targets,
                                         **kwargs))

        loss_dict['loss'] += self._class_error_mu * loss_dict['class_error']
        loss_dict['loss'] += (self._class_error_aux_mu
                              * loss_dict['aux_class_error'])

    def _empty_task_loss(self, *args, **kwargs):
        pass

    def forward(self, inputs, outputs, targets=None, net=None, **kwargs):
        loss_dict = {'loss': 0, 
                     'channel_e': torch.LongTensor([-1])}

        self.dist_loss_func(loss_dict, inputs, outputs, **kwargs)
        self.rate_loss_func(loss_dict, inputs, outputs, net, **kwargs)
        self.penalty_loss_func(loss_dict, inputs, outputs, net, **kwargs)
        self.class_loss_func(loss_dict, targets, outputs, **kwargs)

        return loss_dict


def setup_loss(criterion, **kwargs):
    if 'rate' in criterion.lower():
        rate_loss_type = 'Rate'
    else:
        rate_loss_type = None

    if 'mse' in criterion.lower():
        dist_loss_type = 'MSE'

    elif 'msssim' in criterion.lower() or 'ms-ssim' in criterion.lower():
        dist_loss_type = 'MSSSIM'
    else:
        dist_loss_type = None

    if 'multiscale' in criterion.lower():
        dist_loss_type = 'Multiscale' + dist_loss_type

    if 'penaltya' in criterion.lower() or 'pa' in criterion.lower():
        penalty_loss_type = 'PenaltyA'
    elif 'penaltyb' in criterion.lower() or 'pb' in criterion.lower():
        penalty_loss_type = 'PenaltyB'
    else:
        penalty_loss_type = 'none'

    if 'bce' in criterion.lower() or 'binarycrossentropy' in criterion.lower():
        class_loss_type = 'BCELoss'
    elif 'ce' in criterion.lower() or 'crossentropy' in criterion.lower():
        class_loss_type = 'CELoss'
    else:
        class_loss_type = None

    if 'weighted' in criterion.lower():
        class_loss_type = 'Weighted' + class_loss_type

    if class_loss_type is not None and 'aux' in criterion.lower():
        class_loss_type += 'WithAux'

    return GeneralLoss(dist_loss_type, rate_loss_type, penalty_loss_type,
                       class_loss_type,
                       **kwargs)
