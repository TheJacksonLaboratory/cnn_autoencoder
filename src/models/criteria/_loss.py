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
        else:
            self.dist_loss = None

        if rate_loss_type is not None:
            assert rate_loss_type in RATE_LOSS_LIST
            self.rate_loss = RATE_LOSS_LIST[rate_loss_type](**kwargs)
        else:
            self.rate_loss = None

        if (penalty_loss_type is not None
          and penalty_loss_type.lower() != "none"):
            assert penalty_loss_type in PENALTY_LOSS_LIST
            self.penalty_loss = PENALTY_LOSS_LIST[dist_loss_type](**kwargs)
            self._penalty_beta = penalty_beta

        else:
            self.penalty_loss = None

        if class_loss_type is not None and class_loss_type.lower() != "none":
            assert class_loss_type in CLASSLOSS_LIST
            self.class_loss = CLASSLOSS_LIST[class_loss_type](**kwargs)
            self._class_error_mu = class_error_mu
            self._class_error_aux_mu = class_error_aux_mu

        else:
            self.class_loss = None

        assert (self.dist_loss is not None
                or self.rate_loss is not None
                or self.penalty_loss is not None
                or self.class_loss is not None), "No active criterion was given"

    def forward(self, input, output, target=None, net=None, **kwargs):
        dist_dict = {'loss': 0}

        if self.dist_loss is not None:
            dist_dict.update(self.dist_loss(x=input, x_r=output['x_r'], 
                                            **kwargs))
            dist_dict['dist'] = [self._multiplier * d
                                 for d in dist_dict['dist']]

            dist_dict['dist_loss'] = reduce(lambda d1, d2: d1 + d2,
                                            map(lambda wd: wd[0] * wd[1],
                                                zip(dist_dict['dist'],
                                                    self._distortion_lambda)))
            dist_dict['loss'] += dist_dict['dist_loss']

        if self.rate_loss is not None:
            dist_dict.update(self.rate_loss(x=input, p_y=output['p_y'],
                                            **kwargs))

            dist_dict['entropy_loss'] = net['fact_ent'].module.loss()

            dist_dict['loss'] += dist_dict['rate_loss']

        if self.penalty_loss is not None:
            dist_dict.update(self.penalty_loss(x=input, y=output['y'],
                                    net=net['decoder'],
                                    **kwargs))

            dist_dict['loss'] += (self._penalty_beta
                                  * dist_dict['weighted_penalty'])
        else:
            dist_dict['channel_e'] = torch.LongTensor([-1])

        if self.class_loss is not None:
            dist_dict.update(self.class_loss(pred=output['t_pred'],
                                             aux_pred=output['t_aux_pred'],
                                             t=target,
                                             **kwargs))

            dist_dict['loss'] += (self._class_error_mu
                                  * dist_dict['class_error']
                                  + self._class_error_aux_mu
                                  * dist_dict['aux_class_error'])

        return dist_dict


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

    if class_loss_type is not None and 'aux' in criterion.lower():
        class_loss_type += 'WithAux'

    return GeneralLoss(dist_loss_type, rate_loss_type, penalty_loss_type,
                       class_loss_type,
                       **kwargs)
