from ._ratedistloss import *
from ._classloss import *


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

class StoppingCriterion(object):
    def __init__(self, max_iterations, **kwargs):
        self._max_iterations = max_iterations
        self._curr_iteration = 0

    def update(self, **kwargs):
        self._curr_iteration += 1

    def check(self):
        return self._curr_iteration <= self._max_iterations

    def reset(self):
        self._curr_iteration = 0

    def __repr__(self):
        decision = self.check()
        repr = 'StoppingCriterion(max-iterations: %d, current-iterations: %d, decision: %s)' % (self._max_iterations, self._curr_iteration, 'Continue' if decision else 'Stop')
        return repr


class EarlyStoppingPatience(StoppingCriterion):
    def __init__(self, early_patience=5, early_warmup=0, target='min', initial=None, **kwargs):
        super(EarlyStoppingPatience, self).__init__(**kwargs)

        self._bad_epochs = 0
        self._patience = early_patience
        self._warmup = early_warmup

        self._target = target
        self._initial = initial

        if self._target == 'min':
            self._best_metric = float('inf') if self._initial is None else self._initial
            self._metric_sign = 1
        else:
            self._best_metric = 0 if self._initial is None else self._initial
            self._metric_sign = -1

    def update(self, metric=None, **kwargs):
        super(EarlyStoppingPatience, self).update(**kwargs)

        if metric is None or self._curr_iteration < self._warmup:
            return

        if self._best_metric >= (self._metric_sign * metric):
            self._bad_epochs = 0
            self._best_metric = self._metric_sign * metric
        else:
            self._bad_epochs += 1

    def check(self):
        parent_decision = super().check()
        decision = self._bad_epochs < self._patience
        return parent_decision & decision

    def reset(self):
        super().reset()
        if self._target == 'min':
            self._best_metric = float('inf') if self._initial is None else self._initial
        else:
            self._best_metric = 0 if self._initial is None else self._initial

    def __repr__(self):
        repr = super(EarlyStoppingPatience, self).__repr__()
        decision = self.check()
        repr += '; EarlyStoppingPatience(target: %s, patience: %d, warmup: %d, bad-epochs: %d, best metric: %.4f, decision: %s)' % (self._target, self._patience, self._warmup, self._bad_epochs, self._best_metric, 'Continue' if decision else 'Stop')
        return repr


class EarlyStoppingTarget(StoppingCriterion):
    """ Keep training while the inequality holds.
    """
    def __init__(self, target, comparison='l', **kwargs):
        super(EarlyStoppingTarget, self).__init__(**kwargs)
        self._target = target
        self._comparison = comparison
        self._last_metric = -1

    def update(self, metric=None, **kwargs):
        super(EarlyStoppingTarget, self).update(**kwargs)
        self._last_metric = metric

    def reset(self):
        super().reset()
        self._last_metric = -1

    def check(self):
        parent_decision = super(EarlyStoppingTarget, self).check()

        # If the criterion is met, the training is stopped
        if self._comparison == 'l':
            decision = self._last_metric < self._target
        elif self._comparison == 'le':
            decision = self._last_metric <= self._target
        elif self._comparison == 'g':
            decision = self._last_metric > self._target
        elif self._comparison == 'ge':
            decision = self._last_metric >= self._target

        return parent_decision & decision

    def __repr__(self):
        repr = super(EarlyStoppingTarget, self).__repr__()
        decision = self.check()
        repr += '; EarlyStoppingTarget(comparison: %s, target: %s, last-metric: %.4f, decision: %s)' % (self._comparison, self._target, self._last_metric, 'Continue' if decision else 'Stop')
        return repr


def setup_stopping_criteria(steps, criterion, energy_limit=0.7,
                            sub_iter_steps=100,
                            **kwargs):
    stopping_criteria = {
        'early_stopping': EarlyStoppingPatience(max_iterations=steps, **kwargs)
    }

    if 'PA' in criterion:
        if energy_limit is None:
            energy_limit = 0.7

        stopping_criteria['penalty'] = EarlyStoppingTarget(
            max_iterations=sub_iter_steps,
            target=energy_limit,
            comparison = 'le',
            **kwargs)

    elif 'PB' in criterion:
        if energy_limit is None:
            energy_limit = 0.001

        stopping_criteria['penalty'] = EarlyStoppingTarget(
            max_iterations=sub_iter_steps,
            target=energy_limit,
            comparison='ge',
            **kwargs)

    return stopping_criteria


def setup_criteria(args, checkpoint=None):
    """Setup a loss function for the neural network optimization, and training
    stopping criteria.

    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. All the parameters are
        passed directly to the criteria constructors.
    checkpoint: Path or None
        Path to a pretrained model. Only used when the Penalty B is active, 
        to extract the channel index with highest energy.
    Returns
    -------
    criterion : nn.Module
        The loss function that is used as target to optimize the parameters of
        the nerual network.

    stopping_criteria : list[StoppingCriterion]
        A list of stopping criteria. The first element is always set to stop
        the training after a fixed number of iterations.
        Depending on the criterion used, additional stopping criteria is set.
    """

    # Early stopping criterion:
    if 'PB' in args.criterion:
        args.channel_e = 0
        if checkpoint is not None:
            checkpoint_state = torch.load(checkpoint, map_location='cpu')
            args.channel_e = int(checkpoint_state.get('channel_e', 0))

    stopping_criteria = setup_stopping_criteria(**args.__dict__)

    criterion = setup_loss(**args.__dict__)

    return criterion, stopping_criteria