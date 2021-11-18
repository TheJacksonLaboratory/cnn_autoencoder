import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RateDistorsion(nn.Module):
    def __init__(self, distorsion_lambda=0.01):
        super(RateDistorsion, self).__init__()

        if not isinstance(distorsion_lambda, torch.Tensor):
            distorsion_lambda = torch.FloatTensor([distorsion_lambda])
        
        self._distorsion_lambda = distorsion_lambda

    def forward(self, x, x_r, p_y):
        # Distorsion
        dist = F.mse_loss(x_r, x)
        
        # Rate of compression:
        rate = torch.mean(-torch.log2(p_y + 1e-10))

        return self._distorsion_lambda * dist + rate
    

class RateDistorsionPenalty(nn.Module):
    def __init__(self, distorsion_lambda=0.01, penalty_beta=0.001):
        super(RateDistorsionPenalty, self).__init__()

        if not isinstance(distorsion_lambda, torch.Tensor):
            distorsion_lambda = torch.FloatTensor([distorsion_lambda])

        if not isinstance(penalty_beta, torch.Tensor):
            penalty_beta = torch.FloatTensor([penalty_beta])
        
        self._distorsion_lambda = distorsion_lambda
        self._penalty_beta = penalty_beta


    def forward(self, x, x_r, p_y, AB):
        # Distorsion
        dist = F.mse_loss(x_r, x)
        
        # Rate of compression:
        rate = torch.mean(-torch.log2(p_y + 1e-10))

        return self._distorsion_lambda * dist + rate + self._penalty_beta * AB
    