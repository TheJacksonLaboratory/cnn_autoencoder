from functools import reduce

import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F


class RateDistortion(nn.Module):
    def __init__(self, distorsion_lambda=0.01, normalize=False, **kwargs):
        super(RateDistortion, self).__init__()
        if isinstance(distorsion_lambda, list) and len(distorsion_lambda) == 1:
            self._distorsion_lambda = distorsion_lambda[0]
        else:
            self._distorsion_lambda = distorsion_lambda

        if normalize:
            self._min_range = -1.0
            self._max_range = 1.0
            self._range_scale = 0.5
            self._range_offset = 0.5
        else:
            self._min_range = 0.0
            self._max_range = 1.0
            self._range_scale = 1.0
            self._range_offset = 0.0

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        dist, rate = self.compute_distortion(x, y, x_r, p_y, net)
        return self._distorsion_lambda * dist + rate, None

    def compute_distortion(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Distortion
        dist = 255.0 ** 2 * F.mse_loss(
            self._range_scale * x_r + self._range_offset,
            self._range_scale * x.to(x_r.device) + self._range_offset)

        # Rate of compression:
        rate = (torch.sum(-torch.log2(p_y))
                / (x.size(0) * x.size(2) * x.size(3)))

        return dist, rate


class RateDistortionPyramid(nn.Module):
    def __init__(self, distorsion_lambda=0.01, normalize=False,
                 **kwargs):
        super(RateDistortionPyramid, self).__init__()
        if not isinstance(distorsion_lambda, list):
            distorsion_lambda = [distorsion_lambda]

        self._distorsion_lambda = distorsion_lambda

        self._pyramid_downsample_kernel = torch.tensor(
            [[[[1, 4, 6, 4, 1],
               [4, 16, 24, 16, 4],
               [6, 24, 36, 24, 6],
               [4, 16, 24, 16, 4],
               [1, 4, 6, 4, 1]
               ]]], requires_grad=False) / 256.0

        if normalize:
            self._min_range = -1.0
            self._max_range = 1.0
            self._range_scale = 0.5
            self._range_offset = 0.5
        else:
            self._min_range = 0.0
            self._max_range = 1.0
            self._range_scale = 1.0
            self._range_offset = 0.0

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        if isinstance(self._distorsion_lambda, float):
            distorsion_lambda = [self._distorsion_lambda] * len(x_r)
        else:
            distorsion_lambda = self._distorsion_lambda

        dist, rate = self.compute_distortion(x, y, x_r, p_y, net)
        dist = reduce(lambda d1, d2: d1+d2, map(lambda dl: dl[0] * dl[1],
                      zip(dist, distorsion_lambda)), 0)

        return dist + rate, None

    def compute_distortion(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Distortion
        dist = []
        x_org = x.clone().to(x_r[0].device)
        for s, x_r_s in enumerate(x_r):
            dist.append(255.0 **2 * F.mse_loss(
                self._range_scale * x_r_s + self._range_offset,
                self._range_scale * x_org + self._range_offset))

            if s < len(x_r) - 1:
                with torch.no_grad():
                    x_org = F.conv2d(x_org,
                                     self._pyramid_downsample_kernel.repeat(
                                        x.size(1), 1, 1, 1).to(x_r_s.device),
                                     padding=2,
                                     groups=x.size(1))
                    x_org = F.interpolate(x_org, scale_factor=0.5,
                                          mode='bilinear',
                                          align_corners=False)

        # Rate of compression:
        rate = (torch.sum(-torch.log2(p_y))
                / (x.size(0) * x.size(2) * x.size(3)))
        return dist, rate


class MultiScaleSSIM(nn.Module):
    def __init__(self, patch_size, distorsion_lambda=0.01, normalize=False,
                 **kwargs):
        super(MultiScaleSSIM, self).__init__()
        if isinstance(distorsion_lambda, list):
            distorsion_lambda = distorsion_lambda[0]
        self._distorsion_lambda = distorsion_lambda

        self.padding = nn.ZeroPad2d((11 - patch_size // 2 ** 4) * 2 ** 3)

        self.msssim = torchmetrics.MultiScaleStructuralSimilarityIndexMeasure(
            kernel_size=(11, 11),
            sigma=(1.5, 1.5),
            reduction='elementwise_mean',
            k1=0.01,
            k2=0.03,
            data_range=255,
            betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
            normalize='relu')

        if normalize:
            self._min_range = -1.0
            self._max_range = 1.0
            self._range_scale = 0.5
            self._range_offset = 0.5
        else:
            self._min_range = 0.0
            self._max_range = 1.0
            self._range_scale = 1.0
            self._range_offset = 0.0

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        ms_ssim, rate = self.compute_distortion(x, y, x_r, p_y, net)
        return self._distorsion_lambda * ms_ssim + rate, None

    def compute_distortion(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Distortion
        ms_ssim = 1. - self.msssim.to(x_r.device)(
            self._range_scale * self.padding(x_r) + self._range_offset,
            self._range_scale * self.padding(x.to(x_r.device))
            + self._range_offset)

        # Rate of compression:
        rate = (torch.sum(-torch.log2(p_y))
                / (x.size(0) * x.size(2) * x.size(3)))

        return ms_ssim, rate


class MultiScaleSSIMPyramid(nn.Module):
    def __init__(self, patch_size, distorsion_lambda=0.01, normalize=False,
                 **kwargs):
        super(MultiScaleSSIMPyramid, self).__init__()
        if not isinstance(distorsion_lambda, list):
            distorsion_lambda = [distorsion_lambda]

        self.padding = []
        self.msssim_pyr = []
        for s in range(len(distorsion_lambda)):
            if ((11 - 2 * s) - patch_size // 2 ** (s + 4)) > 0:
                self.padding.append(nn.ZeroPad2d(
                    ((11 - 2 * s) - patch_size // 2 ** (s + 4)) * 2 ** 3))
            else:
                self.padding.append(nn.Identity())
            self.msssim_pyr.append(
                torchmetrics.MultiScaleStructuralSimilarityIndexMeasure(
                    kernel_size=(11 - 2 * s,
                                 11 - 2 * s),
                    sigma=(1.5 / 2 ** s, 1.5 / 2 ** s),
                    reduction='elementwise_mean',
                    k1=0.01,
                    k2=0.03,
                    data_range=255,
                    betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
                    normalize='relu'))

        if normalize:
            self._min_range = -1.0
            self._max_range = 1.0
            self._range_scale = 0.5
            self._range_offset = 0.5
        else:
            self._min_range = 0.0
            self._max_range = 1.0
            self._range_scale = 1.0
            self._range_offset = 0.0

        self._distorsion_lambda = distorsion_lambda

        self._pyramid_downsample_kernel = torch.tensor(
            [[[[1, 4, 6, 4, 1],
               [4, 16, 24, 16, 4],
               [6, 24, 36, 24, 6],
               [4, 16, 24, 16, 4],
               [1, 4, 6, 4, 1]
            ]]], requires_grad=False) / 256.0

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        if isinstance(self._distorsion_lambda, float):
            distorsion_lambda = [self._distorsion_lambda] * len(x_r)
        else:
            distorsion_lambda = self._distorsion_lambda

        ms_ssim_pyr, rate = self.compute_distortion(x, y, x_r, p_y, net)
        ms_ssim = reduce(lambda d1, d2: d1+d2,
                         map(lambda dl: dl[0] * dl[1],
                             zip(ms_ssim_pyr, distorsion_lambda)), 0)

        return ms_ssim + rate, None

    def compute_distortion(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Distortion
        ms_ssim_pyr = []
        x_org = x.clone().to(x_r[0].device)
        for s, (x_r_s, pad_fn, msssim_s) in enumerate(zip(x_r, self.padding,
                                                          self.msssim_pyr)):
            ms_ssim_pyr.append(1. - msssim_s.to(x_r_s.device)(
                self._range_scale * pad_fn(x_r_s) + self._range_offset,
                self._range_scale * pad_fn(x_org) + self._range_offset))
            if s < len(x_r) - 1:
                with torch.no_grad():
                    x_org = F.conv2d(
                        x_org,
                        self._pyramid_downsample_kernel.repeat(
                            x.size(1),
                            1,
                            1,
                            1).to(x_r_s.device),
                        padding=2,
                        groups=x.size(1))
                    x_org = F.interpolate(x_org, scale_factor=0.5,
                                          mode='bilinear',
                                          align_corners=False)

        # Rate of compression:
        rate = (torch.sum(-torch.log2(p_y))
                / (x.size(0) * x.size(2) * x.size(3)))

        return ms_ssim_pyr, rate


class PenaltyA(nn.Module):
    def __init__(self, **kwargs):
        super(PenaltyA, self).__init__()

    def compute_penalty(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Compute A, the approximation to the variance introduced during the
        # analysis track
        with torch.no_grad():
            x_mean = torch.mean(x, dim=1)
            x_var = torch.var(x_mean, dim=(1, 2)).unsqueeze(dim=1) + 1e-10

        A = torch.var(y, dim=(2, 3)) / x_var.to(y.device)
        A = A / torch.sum(A, dim=1).unsqueeze(dim=1)

        # Compute the maximum energy consentrated among the layers
        with torch.no_grad():
            max_energy = A.max(dim=1)[0]

        P_A = torch.sum(-A * torch.log2(A + 1e-10), dim=1)

        return torch.mean(P_A), torch.mean(max_energy)


class PenaltyB(nn.Module):
    def __init__(self, **kwargs):
        super(PenaltyB, self).__init__()

    def compute_penalty(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Compute B, the approximation to the variance introduced during the
        # quantization and synthesis track
        _, K, H, W = y.size()

        with torch.no_grad():
            x_mean = torch.mean(x, dim=1)
            x_var = torch.var(x_mean, dim=(1, 2)).unsqueeze(dim=1) + 1e-10

            A = torch.var(y, dim=(2, 3)) / x_var.to(y.device)
            A = A / torch.sum(A, dim=1).unsqueeze(dim=1)

            # Select the maximum energy channel
            max_energy_channel = A.argmax(dim=1)

        fake_codes = torch.cat(
            [torch.zeros(1, K, H, W).index_fill_(1, torch.tensor([k]), 1)
             for k in range(K)], dim=0)

        fake_rec = net(fake_codes, synthesize_only=True)
        B = torch.var(fake_rec, dim=(1, 2, 3))
        B = B / torch.sum(B)

        P_B = F.max_pool1d(B.unsqueeze(dim=0).unsqueeze(dim=1), kernel_size=K, stride=K).squeeze()
        P_B = B[max_energy_channel]
        return P_B, P_B.detach()


class RateDistortionPenaltyA(RateDistortion, PenaltyA):
    def __init__(self, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateDistortionPenaltyA, self).__init__(distorsion_lambda, **kwargs)
        self._penalty_beta = penalty_beta

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Distortion and rate of compression loss:
        dist_rate, _ = super(RateDistortionPenaltyA, self).forward(x, y, x_r, p_y, net)
        P_A, max_energy = self.compute_penalty(x, y, x_r, p_y, net)

        return dist_rate + self._penalty_beta * P_A, max_energy


class RateDistortionPyramidPenaltyA(RateDistortionPyramid, PenaltyA):
    def __init__(self, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateDistortionPyramidPenaltyA, self).__init__(distorsion_lambda, **kwargs)
        self._penalty_beta = penalty_beta

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Distortion and rate of compression loss:
        dist_rate, _ = super(RateDistortionPyramidPenaltyA, self).forward(x, y, x_r, p_y, net)
        P_A, max_energy = self.compute_penalty(x, y, x_r, p_y, net)

        return dist_rate + self._penalty_beta * P_A, max_energy


class RateMSSSIMPenaltyA(MultiScaleSSIM, PenaltyA):
    def __init__(self, patch_size, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateMSSSIMPenaltyA, self).__init__(patch_size, distorsion_lambda, **kwargs)
        self._penalty_beta = penalty_beta

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Distortion and rate of compression loss:
        ms_ssim_rate, _ = super(RateMSSSIMPenaltyA, self).forward(x, y, x_r, p_y, net)
        P_A, max_energy = self.compute_penalty(x, y, x_r, p_y, net)

        return ms_ssim_rate + self._penalty_beta * P_A, max_energy


class RateMSSSIMPyramidPenaltyA(MultiScaleSSIMPyramid, PenaltyA):
    def __init__(self, patch_size, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateMSSSIMPyramidPenaltyA, self).__init__(patch_size, distorsion_lambda, **kwargs)
        self._penalty_beta = penalty_beta

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Distortion and rate of compression loss:
        ms_ssim_rate, _ = super(RateMSSSIMPyramidPenaltyA, self).forward(x, y, x_r, p_y, net)
        P_A, max_energy = self.compute_penalty(x, y, x_r, p_y, net)

        return ms_ssim_rate + self._penalty_beta * P_A, max_energy


class RateDistortionPenaltyB(RateDistortion, PenaltyB):
    def __init__(self, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateDistortionPenaltyB, self).__init__(distorsion_lambda, **kwargs)
        self._penalty_beta = penalty_beta

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Distortion and rate of compression loss:
        dist_rate, _ = super(RateDistortionPenaltyB, self).forward(x, y, x_r, p_y, net)
        P_B, P_B_mean = self.compute_penalty(x, y, x_r, p_y, net)
        return dist_rate + self._penalty_beta * P_B, P_B_mean


class RateDistortionPyramidPenaltyB(RateDistortionPyramid, PenaltyB):
    def __init__(self, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateDistortionPyramidPenaltyB, self).__init__(distorsion_lambda, **kwargs)
        self._penalty_beta = penalty_beta

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Distortion and rate of compression loss:
        dist_rate, _ = super(RateDistortionPyramidPenaltyB, self).forward(x, y, x_r, p_y, net)
        P_B, P_B_mean = self.compute_penalty(x, y, x_r, p_y, net)
        return dist_rate + self._penalty_beta * P_B, P_B_mean


class RateMSSSIMPenaltyB(MultiScaleSSIM, PenaltyB):
    def __init__(self, patch_size, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateMSSSIMPenaltyB, self).__init__(patch_size, distorsion_lambda, **kwargs)
        self._penalty_beta = penalty_beta

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Distortion and rate of compression loss:
        ms_ssim_rate, _ = super(RateMSSSIMPenaltyB, self).forward(x, y, x_r, p_y, net)
        P_B, P_B_mean = self.compute_penalty(x, y, x_r, p_y, net)
        return ms_ssim_rate + self._penalty_beta * P_B, P_B_mean


class RateMSSSIMPyramidPenaltyB(MultiScaleSSIMPyramid, PenaltyB):
    def __init__(self, patch_size, distorsion_lambda=0.01, penalty_beta=0.001, **kwargs):
        super(RateMSSSIMPyramidPenaltyB, self).__init__(patch_size, distorsion_lambda, **kwargs)
        self._penalty_beta = penalty_beta

    def forward(self, x=None, y=None, x_r=None, p_y=None, net=None):
        # Distortion and rate of compression loss:
        ms_ssim_rate, _ = super(RateMSSSIMPyramidPenaltyB, self).forward(x, y, x_r, p_y, net)
        P_B, P_B_mean = self.compute_penalty(x, y, x_r, p_y, net)
        return ms_ssim_rate + self._penalty_beta * P_B, P_B_mean


class CrossEnropy2D(nn.Module):
    def __init__(self):
        super(CrossEnropy2D, self).__init__()

        self._my_ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, y, t):
        return self._my_ce(y, t.squeeze().long())


class CrossEnropyDistance(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEnropyDistance, self).__init__()

    def forward(self, y, t):
        # Weight the cross-entropy according to the weight map.
        cew = t[:, :1] * F.binary_cross_entropy_with_logits(y, t[:, 1:], reduction='none')

        return cew


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
