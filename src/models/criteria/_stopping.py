
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
