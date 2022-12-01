from functools import reduce

import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistortionLossBase(object):
    def __init__(self, distortion_lambda=0.1, normalize=False, **kwargs):
        if not isinstance(distortion_lambda, list):
            distortion_lambda = [distortion_lambda]

        self._distortion_lambda = distortion_lambda
    
        if normalize:
            self._range_scale = 0.5
            self._range_offset = 0.5
        else:
            self._range_scale = 1.0
            self._range_offset = 0.0

    def compute_weighted_dist(self, x, x_r, **kwargs):
        dist = self.compute_dist(x, x_r)
        if not isinstance(dist, list):
            dist = [dist]

        dist = [d_l * d for d_l, d in zip(self._distortion_lambda, dist)]
        return dist


class PyramidLossMixin:
    def __init__(self, channels_org, **kwargs):
        super().__init__(**kwargs)

        pyr_kernel = torch.tensor(
            [[[[1, 4, 6, 4, 1],
               [4, 16, 24, 16, 4],
               [6, 24, 36, 24, 6],
               [4, 16, 24, 16, 4],
               [1, 4, 6, 4, 1]
               ]]], requires_grad=False) / 256.0

        self._pyr_kernel = pyr_kernel.repeat(channels_org, 1, 1, 1)

    def downsample_pyramid(self, x, **kwargs):
        with torch.no_grad():
            x_dwn = F.conv2d(x, self._pyr_kernel.to(x.device), padding=2,
                             groups=x.size(1))
            x_dwn = F.interpolate(x_dwn, scale_factor=0.5, mode='bilinear',
                                  align_corners=False)
        return x_dwn

    def compute_dist(self, x, x_r, **kwargs):
        dist = []
        x_org = x.clone()

        for s, (x_r_s, d_crt) in enumerate(zip(x_r, self._dist_criteria)):
            dist_s = d_crt.compute_dist(x_org, x_r_s)
            dist.append(dist_s)

            # Downsample the original input
            if s < (len(self._dist_criteria) - 1):
                x_org = self.downsample_pyramid(x_org)

        return dist


class RateDistortionMixin:
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def __call__(self, x, x_r, p_y, **kwargs):
        return self.loss_distortion_rate(x, x_r, p_y), None

    def loss_distortion_rate(self, x, x_r, p_y, **kwargs):
        dist = self.compute_weighted_dist(x, x_r)
        dist = reduce(lambda d1, d2: d1 + d2, dist)
        rate = self.compute_rate(x, p_y)
        return dist + rate


class RateDistortionPenaltyMixin:
    def __init__(self, penalty_beta=0.001, **kargs):
        super().__init__(**kargs)
        self._penalty_beta = penalty_beta

    def __call__(self, **kwargs):
        dist = self.compute_weighted_dist(x, x_r)
        dist = reduce(lambda d1, d2: d1 + d2, dist)
        rate = self.compute_rate(x, p_y)
        penalty, energy = self.compute_penalty(**kwargs)
        dist_rate_penalty = dist + rate + self._penalty_beta * penalty
        return dist_rate_penalty, energy


class RateLoss(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_rate(self, x, p_y, **kwargs):
        # Rate of compression:
        rate = (torch.sum(-torch.log2(p_y))
                / (x.size(0) * x.size(2) * x.size(3)))
        return rate


class DistortionLoss(DistortionLossBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_dist(self, x, x_r, **kwargs):
        dist = 255.0 ** 2 * F.mse_loss(self._range_scale * x_r
                                       + self._range_offset,
                                       self._range_scale * x
                                       + self._range_offset)
        return dist


class MSSSIMLoss(DistortionLossBase):
    def __init__(self, patch_size, scale=1, **kwargs):
        super().__init__(**kwargs)

        if ((11 - 2 * scale) - patch_size // 2 ** (scale + 4)) > 0:
            self.padding = nn.ZeroPad2d(
                ((11 - 2 * scale) - patch_size // 2 ** (scale + 4)) * 2 ** 3)
        else:
            self.padding = nn.Identity()

        self.msssim = torchmetrics.MultiScaleStructuralSimilarityIndexMeasure(
            kernel_size=(11 - 2 * scale, 11 - 2 * scale),
            sigma=(1.5 / 2 ** scale, 1.5 / 2 ** scale),
            reduction='elementwise_mean',
            k1=0.01,
            k2=0.03,
            data_range=None,
            betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
            normalize='relu')

    def compute_dist(self, x, x_r, **kwargs):
        ms_ssim = self.msssim.to(x_r.device)(
            self._range_scale * self.padding(x_r)
            + self._range_offset,
            self._range_scale * self.padding(x.to(x_r.device))
            + self._range_offset)

        ms_ssim = 1.0 - ms_ssim
        return ms_ssim


class DistortionPyramidLoss(PyramidLossMixin, DistortionLossBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._dist_criteria = [DistortionLoss(**kwargs)
                               for _ in range(len(self._distortion_lambda))]


class MSSSIMPyramidLoss(PyramidLossMixin, DistortionLossBase):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)

        self._dist_criteria = [MSSSIMLoss(patch_size=patch_size, scale=s,
                                          **kwargs)
                               for s in range(len(self._distortion_lambda))]


class PenaltyA(object):
    def __init__(self, **kwargs):
        pass

    def compute_penalty(self, x, y, **kwargs):
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


class PenaltyB(object):
    def __init__(self, **kwargs):
        pass

    def compute_penalty(self, x, y, net, **kwargs):
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

        P_B = F.max_pool1d(B.unsqueeze(dim=0).unsqueeze(dim=1),
                           kernel_size=K,
                           stride=K).squeeze()
        P_B = B[max_energy_channel]

        return P_B.mean(), P_B.detach()


class RateMSELoss(RateDistortionMixin,
                  RateLoss,
                  DistortionLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MultiscaleRateMSELoss(RateDistortionMixin,
                            RateLoss,
                            DistortionPyramidLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RateMSSSIMLoss(RateDistortionMixin,
                     RateLoss,
                     MSSSIMLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MultiscaleRateMSSSIM(RateDistortionMixin,
                           RateLoss,
                           MSSSIMPyramidLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RateMSEPenaltyA(RateDistortionPenaltyMixin,
                      RateLoss,
                      DistortionLoss,
                      PenaltyA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MultiscaleRateMSEPenaltyA(RateDistortionPenaltyMixin,
                                RateLoss,
                                DistortionPyramidLoss,
                                PenaltyA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RateMSSSIMPenaltyA(RateDistortionPenaltyMixin,                         
                         RateLoss,
                         MSSSIMLoss,
                         PenaltyA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MultiscaleRateMSSSIMPenaltyA(RateDistortionPenaltyMixin,
                                   RateLoss,
                                   MSSSIMPyramidLoss,
                                   PenaltyA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RateMSEPenaltyB(RateDistortionPenaltyMixin,
                      RateLoss,
                      DistortionLoss,
                      PenaltyB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MultiscaleRateMSEPenaltyB(RateDistortionPenaltyMixin,
                                RateLoss,
                                DistortionPyramidLoss,
                                PenaltyB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RateMSSSIMPenaltyB(RateDistortionPenaltyMixin,                         
                         RateLoss,
                         MSSSIMLoss,
                         PenaltyB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MultiscaleRateMSSSIMPenaltyB(RateDistortionPenaltyMixin,
                                   RateLoss,
                                   MSSSIMPyramidLoss,
                                   PenaltyB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

LOSS_LIST = {
    "RateMSELoss": RateMSELoss,
    "MultiscaleRateMSELoss": MultiscaleRateMSELoss,
    "RateMSSSIMLoss": RateMSSSIMLoss,
    "MultiscaleRateMSSSIM": MultiscaleRateMSSSIM,
    "RateMSEPenaltyA": RateMSEPenaltyA,
    "MultiscaleRateMSEPenaltyA": MultiscaleRateMSEPenaltyA,
    "RateMSSSIMPenaltyA": RateMSSSIMPenaltyA,
    "MultiscaleRateMSSSIMPenaltyA": MultiscaleRateMSSSIMPenaltyA,
    "RateMSEPenaltyB": RateMSEPenaltyB,
    "MultiscaleRateMSEPenaltyB": MultiscaleRateMSEPenaltyB,
    "RateMSSSIMPenaltyB": RateMSSSIMPenaltyB,
    "MultiscaleRateMSSSIMPenaltyB": MultiscaleRateMSSSIMPenaltyB,
    }

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


if __name__ == "__main__":
    class SynthsisTest(nn.Module):
        def __init__(self):
            super().__init__()
        
            self.synthesis = nn.Sequential(
                nn.ConvTranspose2d(48, 48, kernel_size=2, padding=0, stride=2),
                nn.ConvTranspose2d(48, 48, kernel_size=2, padding=0, stride=2),
                nn.ConvTranspose2d(48, 3, kernel_size=2, padding=0, stride=2),
            )

        def forward(self, x, **kwargs):
            return self.synthesis(x)

    x = torch.ones(5, 3, 128, 128)

    analysis = nn.Sequential(
        nn.Conv2d(3, 48, kernel_size=3, padding=1, stride=2),
        nn.Conv2d(48, 48, kernel_size=3, padding=1, stride=2),
        nn.Conv2d(48, 48, kernel_size=3, padding=1, stride=2),
    )

    synthesis = SynthsisTest()

    for l_k in LOSS_LIST.keys():
        loss_fun = LOSS_LIST[l_k](patch_size=3,
                                  channels_org=3,
                                  distortion_lambda=[1.0, 0.1, 0.01])
        y = analysis(x)
        p_y = torch.softmax(y, dim=1)
        x_r = synthesis(y)

        if 'Multiscale' in l_k:
            x_r = [x_r, x_r[:, :, ::2, ::2], x_r[:, :, ::4, ::4]]
        loss, _ = loss_fun(x=x, y=y, x_r=x_r, p_y=p_y, net=synthesis)
        print("Loss function %s value: %f" % (l_k, loss))
