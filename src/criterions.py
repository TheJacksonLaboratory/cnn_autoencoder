import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RateDistorsion(nn.Module):
    def __init__(self, distorsion_lambda=0.01, **kwargs):
        super(RateDistorsion, self).__init__()
        self._distorsion_lambda = distorsion_lambda

    def forward(self, x, x_r, p_y):
        # Distorsion
        dist = F.mse_loss(x_r, x)
        
        # Rate of compression:
        rate = torch.mean(-torch.log2(p_y + 1e-10))

        return self._distorsion_lambda * dist + rate
    

class RateDistorsionPenaltyA(nn.Module):
    def __init__(self, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateDistorsionPenaltyA, self).__init__()
        self._distorsion_lambda = distorsion_lambda
        self._penalty_beta = penalty_beta

    def forward(self, x, y, x_r, p_y):
        # Compute A, the approximation to the variance introduced during the analysis track
        A = torch.sum(y ** 2, dim=(3, 4)) / (torch.sum(x ** 2, dim=(1, 2, 3)) + 1e-10)
        P_A = torch.sum(-A * torch.log2(A), dim=1)

        # Distorsion
        dist = F.mse_loss(x_r, x)
        
        # Rate of compression:
        rate = torch.mean(-torch.log2(p_y + 1e-10))

        return self._distorsion_lambda * dist + rate + self._penalty_beta * torch.mean(P_A)
    

class RateDistorsionPenaltyB(nn.Module):
    def __init__(self, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateDistorsionPenaltyB, self).__init__()
        self._distorsion_lambda = distorsion_lambda
        self._penalty_beta = penalty_beta

    def forward(self, x, x_r, p_y, y, synth_net):
        # Compute B, the approximation to the variance introduced during the quntization and synthesis track
        _, K, H, W = y.size()

        fake_codes = torch.cat([torch.zeros(1, K, H, W).index_fill_(1, torch.tensor([k]), 1) for k in range(K)], dim=0)        
        fake_rec = synth_net(fake_codes)

        B = torch.sum(fake_rec ** 2, dim=(2, 3))
        max_e = torch.softmax(B, dim=1)
        P_B = torch.sum(max_e * B, dim=1)

        # Distorsion
        dist = F.mse_loss(x_r, x)
        
        # Rate of compression:
        rate = torch.mean(-torch.log2(p_y + 1e-10))

        return self._distorsion_lambda * dist + rate + self._penalty_beta * torch.mean(P_B)
    