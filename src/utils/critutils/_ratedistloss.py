from functools import reduce
import math

from pytorch_msssim import ms_ssim
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidLossMixin:
    def __init__(self, channels_org, **kwargs):
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

    def __call__(self, x, x_r, **kwargs):
        dist = []
        x_org = x.clone()

        for s, (x_r, d_crt) in enumerate(zip(x_r, self._dist_criteria)):
            dist_s = d_crt(x_org, [x_r])
            dist += dist_s['dist']

            # Downsample the original input
            if s < (len(self._dist_criteria) - 1):
                x_org = self.downsample_pyramid(x_org)

        return dict(dist=dist)


class RateLoss(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, x, p_y, **kwargs):
        # Rate of compression:
        rate_loss = (-torch.sum(torch.log2(p_y))
                     / (x.size(0) * x.size(2) * x.size(3)))

        return dict(rate_loss=rate_loss)


class DistMSELoss(object):
    def __init__(self, **kwargs):
        self._dist_loss = nn.MSELoss()

    def __call__(self, x, x_r, **kwargs):
        dist = self._dist_loss(x_r[0], x.to(x_r[0].device))
        return dict(dist=[dist])


class DistMSSSIMLoss(object):
    def __init__(self, patch_size, scale=0, normalize=False, **kwargs):
        if normalize:
            self._range = 2

        else:
            self._range = 1

        self.win_size = 11 - 2 * scale
        self.win_sigma = 1.5 / 2 ** scale 
        if (self.win_size - patch_size // 2 ** (scale + 4)) > 0:
            self.padding = nn.ZeroPad2d(
                (self.win_size - patch_size // 2 ** (scale + 4)) * 2 ** 3)
        else:
            self.padding = nn.Identity()

    def __call__(self, x, x_r, **kwargs):
        padded_x_r = self.padding(x_r[0])
        padded_x = self.padding(x.to(x_r[0].device))
        ms_ssim_res = ms_ssim(padded_x_r, padded_x, data_range=self._range, 
                              win_size=self.win_size,
                              win_sigma=self.win_sigma)

        ms_ssim_res = 1.0 - ms_ssim_res
        return dict(dist=[ms_ssim_res])


class DistMSEPyramidLoss(PyramidLossMixin):
    def __init__(self, compression_level=4, **kwargs):
        super().__init__(**kwargs)

        self._dist_criteria = [DistMSELoss(**kwargs)
                               for _ in range(compression_level)]


class DistMSSSIMPyramidLoss(PyramidLossMixin):
    def __init__(self, patch_size, compression_level=4, **kwargs):
        super().__init__(**kwargs)

        self._dist_criteria = [DistMSSSIMLoss(patch_size=patch_size, scale=s,
                                              **kwargs)
                               for s in range(compression_level)]


class PenaltyA:
    def __init__(self, **kwargs):
        pass

    def __call__(self, x, y, **kwargs):
        # Compute A, the approximation to the variance introduced during the
        # analysis track
        with torch.no_grad():
            x_mean = torch.mean(x, dim=1)
            x_var = torch.var(x_mean, dim=(1, 2)).unsqueeze(dim=1) + 1e-10

        A = torch.var(y, dim=(2, 3)) / x_var.to(y.device)
        A = A / torch.sum(A, dim=1).unsqueeze(dim=1)
        A = F.hardtanh(A, min_val=1e-10)

        P_A = torch.mean(torch.sum(-A * torch.log2(A), dim=1))

        # Compute the maximum energy consentrated among the layers
        max_energy, channel_e = A.detach().max(dim=1)
        max_energy = max_energy.median().cpu()
        channel_e = channel_e.median().cpu()

        return dict(weighted_penalty=P_A,
                    penalty=P_A,
                    energy=max_energy,
                    channel_e=channel_e)


class PenaltyB:
    def __init__(self, channel_e=0, **kwargs):
        self._channel_e = channel_e

    def __call__(self, y, net, **kwargs):
        # Compute B, the approximation to the variance introduced during the
        # quantization and synthesis track
        _, K, H, W = y.size()

        fake_codes = torch.zeros(1, K, H, W)
        fake_codes.index_fill_(1, torch.tensor([self._channel_e]), 1)

        fake_rec = net(fake_codes, synthesize_only=True)

        if isinstance(fake_rec, list):
            fake_rec = fake_rec[0]

        B = torch.var(fake_rec, dim=(1, 2, 3))

        P_B = B[0]
 
        return dict(weighted_penalty=P_B,
                    penalty=P_B.detach(),
                    energy=P_B.detach(),
                    channel_e=self._channel_e)


DIST_LOSS_LIST = {
    "MSE": DistMSELoss,
    "MultiscaleMSE": DistMSEPyramidLoss,
    "MSSSIM": DistMSSSIMLoss,
    "MultiscaleMSSSIM": DistMSSSIMPyramidLoss,
    }

PENALTY_LOSS_LIST = {
    "PenaltyA": PenaltyA,
    "PenaltyB": PenaltyB,
    }

RATE_LOSS_LIST = {
    "Rate": RateLoss,
    }