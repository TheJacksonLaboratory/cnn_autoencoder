import logging 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RateDistorsion(nn.Module):
    def __init__(self, distorsion_lambda=0.01, **kwargs):
        super(RateDistorsion, self).__init__()
        self._distorsion_lambda = distorsion_lambda

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Distortion
        dist = F.mse_loss(x_r, x.to(x_r.device))
        
        # Rate of compression:
        rate = torch.mean(torch.sum(-torch.log2(p_y), dim=1)) / (p_y.size(0) * p_y.size(2) * p_y.size(3))
        return self._distorsion_lambda * dist + rate, None


class RateDistorsionPenaltyA(RateDistorsion):
    def __init__(self, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateDistorsionPenaltyA, self).__init__(distorsion_lambda)
        self._penalty_beta = penalty_beta

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Compute A, the approximation to the variance introduced during the analysis track
        with torch.no_grad():
            x_mean = torch.mean(x, dim=1)
            x_var = torch.var(x_mean, dim=(1, 2)).unsqueeze(dim=1) + 1e-10
        
        A = torch.var(y, dim=(2, 3)) / x_var.to(y.device)
        A = A / torch.sum(A, dim=1).unsqueeze(dim=1)

        # Compute the maximum energy consentrated among the layers
        with torch.no_grad():
            max_energy = A.max(dim=1)[0]
        
        P_A = torch.sum(-A * torch.log2(A + 1e-10), dim=1)

        # Distortion and rate of compression loss:
        dist_rate_loss, _ = super(RateDistorsionPenaltyA, self).forward(x=x, y=None, x_r=x_r, p_y=p_y, net=None)

        return dist_rate_loss + self._penalty_beta * torch.mean(P_A), torch.mean(max_energy)


class RateDistorsionPenaltyB(RateDistorsion):
    def __init__(self, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateDistorsionPenaltyB, self).__init__(distorsion_lambda)
        self._penalty_beta = penalty_beta

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Compute B, the approximation to the variance introduced during the quntization and synthesis track
        _, K, H, W = y.size()

        with torch.no_grad():
            x_mean = torch.mean(x, dim=1)
            x_var = torch.var(x_mean, dim=(1, 2)).unsqueeze(dim=1) + 1e-10
        
            A = torch.var(y, dim=(2, 3)) / x_var.to(y.device)
            A = A / torch.sum(A, dim=1).unsqueeze(dim=1)

            # Select the maximum energy channel
            max_energy_channel = A.argmax(dim=1)
        
        fake_codes = torch.cat([torch.zeros(1, K, 2, 2).index_fill_(1, torch.tensor([k]), 1) for k in range(K)], dim=0)
        fake_rec = net(fake_codes, synthesize_only=True)

        B = torch.var(fake_rec, dim=(1, 2, 3))
        B = B / torch.sum(B)

        # P_B = F.max_pool1d(B.unsqueeze(dim=0).unsqueeze(dim=1), kernel_size=K, stride=K).squeeze()
        P_B = B[max_energy_channel]
        
        # Distortion and rate of compression loss:
        dist_rate_loss, _ = super(RateDistorsionPenaltyB, self).forward(x=x, y=None, x_r=x_r, p_y=p_y, net=None)

        return dist_rate_loss + self._penalty_beta * torch.mean(P_B), P_B.detach().mean()


class StoppingCriterion(object):
    def __init__(self, max_iterations, **kwargs):
        self._max_iterations = max_iterations
        self._curr_iteration = 0

    def update(self, iteration, **kwargs):        
        if iteration is None:
            return
        self._curr_iteration = iteration

    def check(self):
        return self._curr_iteration <= self._max_iterations
    
    def __repr__(self):
        decision = self.check()
        repr = 'StoppingCriterion(max-iterations: %d, current-iterations: %d, decision: %s)' % (self._max_iterations, self._curr_iteration, 'Continue' if decision else 'Stop')
        return repr


class EarlyStoppingPatience(StoppingCriterion):
    def __init__(self, patience=5, warmup=0, target='min', initial=None, **kwargs):
        super(EarlyStoppingPatience, self).__init__(**kwargs)

        self._bad_epochs = 0
        self._patience = patience
        self._warmup = warmup

        self._target = target

        if self._target=='min':
            self._best_metric = float('inf') if initial is None else initial
            self._metric_sign = 1
        else:
            self._best_metric = 0 if initial is None else initial
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
