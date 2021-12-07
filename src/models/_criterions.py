import logging 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RateDistorsion(nn.Module):
    def __init__(self, distorsion_lambda=0.01, **kwargs):
        super(RateDistorsion, self).__init__()
        self._distorsion_lambda = distorsion_lambda

    def forward(self, x=None, y=None, x_r=None, p_y=None, synth_net=None):
        # Distorsion
        dist = F.mse_loss(x_r, x)
        
        # Rate of compression:
        rate = torch.sum(-p_y * torch.log2(p_y + 1e-10), dim=1)

        return self._distorsion_lambda * dist + torch.mean(rate), None
    

class RateDistorsionPenaltyA(nn.Module):
    def __init__(self, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateDistorsionPenaltyA, self).__init__()
        self._distorsion_lambda = distorsion_lambda
        self._penalty_beta = penalty_beta

    def forward(self, x=None, y=None, x_r=None, p_y=None, synth_net=None):
        # Compute A, the approximation to the variance introduced during the analysis track
        with torch.no_grad():
            x_mean = torch.mean(x, dim=1)
            x_var = torch.var(x_mean, dim=(1, 2)).unsqueeze(dim=1) + 1e-10
        
        A = torch.var(y, dim=(2, 3)) / x_var
        A = A / torch.sum(A, dim=1).unsqueeze(dim=1)

        # Compute the maximum energy consentrated among the layers
        with torch.no_grad():
            max_energy = A.max()
        
        P_A = torch.sum(-A * torch.log2(A + 1e-10), dim=1)

        # Distorsion
        dist = F.mse_loss(x_r, x)
        
        # Rate of compression:
        rate = torch.sum(-p_y * torch.log2(p_y + 1e-10), dim=1)

        return self._distorsion_lambda * dist + torch.mean(rate) + self._penalty_beta * torch.mean(P_A), max_energy


class RateDistorsionPenaltyB(nn.Module):
    def __init__(self, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateDistorsionPenaltyB, self).__init__()
        self._distorsion_lambda = distorsion_lambda
        self._penalty_beta = penalty_beta

    def forward(self, x=None, y=None, x_r=None, p_y=None, synth_net=None):
        # Compute B, the approximation to the variance introduced during the quntization and synthesis track
        _, K, H, W = y.size()

        fake_codes = torch.cat([torch.zeros(1, K, H, W).index_fill_(1, torch.tensor([k]), 1) for k in range(K)], dim=0)        
        fake_rec = synth_net(fake_codes)

        B = torch.var(fake_rec, dim=(2, 3))
        max_e = torch.softmax(B, dim=1)

        # Compute the hard max energy
        with torch.no_grad():
            max_energy = torch.max(B / torch.sum(B, dim=1).unsqueeze(dim=1))

        P_B = torch.sum(max_e * B, dim=1)

        # Distorsion
        dist = F.mse_loss(x_r, x)
        
        # Rate of compression:
        rate = torch.sum(-p_y * torch.log2(p_y + 1e-10), dim=1)
        return self._distorsion_lambda * dist + torch.mean(rate) + self._penalty_beta * torch.mean(P_B), max_energy


class StoppingCriterion(object):
    def __init__(self, max_iterations, **kwargs):
        self._max_iterations = max_iterations
        self._curr_iteration = None
        self._keep_training = True

    def update(self, iteration, **kwargs):
        self._curr_iteration = iteration
        self._keep_training = self._curr_iteration <= self._max_iterations

    def check(self):
        return self._keep_training

    def __repr__(self):
        repr = 'StoppingCriterion(max-iterations: %d, current-iterations: %d)' % (self._max_iterations, self._curr_iteration)
        return repr


class EarlyStoppingPatience(StoppingCriterion):
    def __init__(self, patience=5, warmup=0, mode='min', initial=None, **kwargs):
        super(EarlyStoppingPatience, self).__init__(**kwargs)

        self._bad_epochs = 0
        self._patience = patience
        self._warmup = warmup

        self._mode = mode

        if self._mode=='min':
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

        self._keep_training &= self._bad_epochs > self._patience

    def __repr__(self):
        repr = 'EarlyStoppingPatience(mode: %s, patience: %d, warmup: %d, bad-epochs: %d, best metric: %.4f)' % (self._mode, self._patience, self._warmup, self._bad_epochs, self._best_metric)
        return repr


class EarlyStoppingTarget(StoppingCriterion):
    def __init__(self, target, mode='l', **kwargs):
        super(EarlyStoppingTarget, self).__init__(**kwargs)
        self._target = target
        self._mode = mode

    def update(self, metric=None, **kwargs):
        super(EarlyStoppingTarget, self).update(**kwargs)

        # If the criterion is met, the training is stopped
        if self._mode == 'l':
            res = metric >= self._target
        elif self._mode == 'le':
            res = metric > self._target
        elif self._mode == 'g':
            res = metric <= self._target
        elif self._mode == 'ge':
            res = metric < self._target
        
        self._keep_training &= res
    
    def __repr__(self):
        repr = 'EarlyStoppingTarget(mode: %s, target: %s)' % (self._mode, self._target)
        return repr
