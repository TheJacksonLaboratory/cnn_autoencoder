from ._ratedist import *
from ._classification import *


class GeneralLoss(nn.Module):
    def __init__(self, dist_loss_type="MSE", penalty_loss_type=None,
                 class_loss_type=None,
                 distortion_lambda=0.1,
                 penalty_beta=0.001,
                 class_error_mu=1.0,
                 class_error_aux_mu=1.0,
                 **kwargs):
        super(GeneralLoss, self).__init__()

        self.rate_loss = RateLoss()

        assert dist_loss_type in DIST_LOSS_LIST
        self.dist_loss = DIST_LOSS_LIST[dist_loss_type](**kwargs)
        self._multiplier = 255 ** 2 if 'MSE' in dist_loss_type else 1

        if not isinstance(distortion_lambda, list):
            distortion_lambda = [distortion_lambda]

        self._distortion_lambda = distortion_lambda

        if (penalty_loss_type is not None
          and penalty_loss_type.lower() != "none"):
            assert penalty_loss_type in PENALTY_LOSS_LIST
            self.penalty_loss = PENALTY_LOSS_LIST[dist_loss_type](**kwargs)
            self._penalty_beta = penalty_beta

        else:
            self.penalty_loss = lambda x, y, net: dict(weighted_penalty=0,
                                                       penalty=0,
                                                       energy=0,
                                                       channel_e=0)
            self._penalty_beta = 0

        if class_loss_type is not None and class_loss_type.lower() != "none":
            assert class_loss_type in CLASSLOSS_LIST
            self.class_loss = CLASSLOSS_LIST[class_loss_type](**kwargs)
            self._class_error_mu = class_error_mu
            self._class_error_aux_mu = class_error_aux_mu

        else:
            self.class_loss = lambda pred, t, aux_pred: dict(class_error=0,
                                                             aux_class_error=0)
            self._class_error_mu = 0
            self._class_error_aux_mu = 0

    def forward(self, input, output, target=None, net=None, **kwargs):
        dist_dict = self.dist_loss(x=input, x_r=output['x_r'], **kwargs)

        dist_dict['dist_loss'] = [self._multiplier * d
                                  for d in dist_dict['dist']]

        dist_dict['dist_loss'] = reduce(lambda d1, d2: d1 + d2,
                                        map(lambda wd: wd[0] * wd[1],
                                            zip(dist_dict['dist'],
                                                self._distortion_lambda)))

        dist_dict.update(self.rate_loss(x=input, p_y=output['p_y'], **kwargs))

        dist_dict['entropy_loss'] = net['fact_ent'].module.loss()

        dist_dict.update(self.penalty_loss(x=input, y=output['y'],
                                           net=net['decoder'],
                                           **kwargs))

        dist_dict.update(self.class_loss(pred=output['t_pred'],
                                         aux_pred=output['t_aux_pred'],
                                         t=target, **kwargs))

        dist_dict['loss'] = (
            dist_dict['dist_loss'] + dist_dict['rate_loss']
            + self._penalty_beta * dist_dict['weighted_penalty']
            + self._class_error_mu * dist_dict['class_error']
            + self._class_error_aux_mu * dist_dict['aux_class_error'])

        return dist_dict


def setup_loss(criterion, **kwargs):
    if 'mse' in criterion.lower():
        dist_loss_type = 'MSE'

    elif 'msssim' in criterion.lower() or 'ms-ssim' in criterion.lower():
        dist_loss_type = 'MSSSIM'

    else:
        raise ValueError("Criterion %s not implemented" % criterion)

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

    if class_loss_type is not None and 'aux' in criterion.lower():
        class_loss_type += 'WithAux'

    return GeneralLoss(dist_loss_type, penalty_loss_type, class_loss_type,
                         **kwargs)
