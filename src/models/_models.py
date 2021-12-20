import logging 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class Quantizer(nn.Module):
    """ Quantizer implements the additive uniform noise quantization method 
        from Balle et al. END-TO-END OPTIMIZED IMAGE COMPRESSION. ICLR 2017
    """
    def __init__(self, lower_bound=-0.5, upper_bound=0.5):
        super(Quantizer, self).__init__()
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def forward(self, x):
        if self.training:
            u = torch.rand_like(x) * (self._upper_bound - self._lower_bound) + self._lower_bound
            q = x + u
        else:
            q = torch.round(x)

        return q


class FactorizedEntropyLayer(nn.Module):
    def __init__(self, channels_bn, d=3, r=3):
        super(FactorizedEntropyLayer, self).__init__()
        self._channels = channels_bn

        # The non-parametric density model is initialized with random normal distributed weights
        self._H = nn.Parameter(nn.init.normal_(torch.empty(channels_bn * r, d, 1, 1), 0.0, 0.01))
        self._b = nn.Parameter(torch.zeros(channels_bn * r))
        self._a = nn.Parameter(nn.init.normal_(torch.empty(1, channels_bn * r, 1, 1), 0.0, 0.01))

    def forward(self, x):
        # Reparametrerize the matrix H, and vector a to generate nonegative Jacobian matrices
        H_k = F.softplus(self._H)
        a_k = torch.tanh(self._a)
            
        # Using the 2d convolution instead of simple element-wise product allows to operate over all channels at the same time
        fx = F.conv2d(x, weight=H_k, bias=self._b, groups=self._channels)
        fx = fx + a_k * torch.tanh(fx)

        return fx


class FactorizedEntropy(nn.Module):
    """ Univariate non-parametric density model to approximate the factorized entropy prior

        This function computes the function c(x) from Balle et al. VARIATIONAL IMAGE COMPRESSION WITH A SCALE HYPERPRIOR. ICLR 2018
        Function c(x) can be used to model the probability of a random variable that has been comvolved with a uniform distribution.
    """
    def __init__(self, channels_bn, K=4, r=3, **kwargs):
        super(FactorizedEntropy, self).__init__()
        self._channels = channels_bn
        self._K = K
        if isinstance(r, int):
            r = [r] * (K - 1) + [1]
        
        d = [1] + r[:-1]

        # The non-parametric density model is initialized with random normal distributed weights
        self._layers = nn.Sequential(*[FactorizedEntropyLayer(channels_bn=channels_bn, d=d_k, r=r_k) for d_k, r_k in zip(d[:-1], r[:-1])])
        self._H = nn.Parameter(nn.init.normal_(torch.empty(channels_bn * r[-1], d[-1], 1, 1), 0.0, 0.01))
        self._b = nn.Parameter(torch.zeros(channels_bn * r[-1]))

    def forward(self, x):
        fx = self._layers(x)

        H_K = F.softplus(self._H)
        fx = torch.sigmoid(F.conv2d(fx, weight=H_K, bias=self._b, groups=self._channels))

        return fx


class DownsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, groups=False, normalize=False, dropout=0.0, bias=False):
        super(DownsamplingUnit, self).__init__()

        model = [nn.Conv2d(channels_in, channels_in, 3, 1, 1, 1, channels_in if groups else 1, bias=bias)]

        if normalize:
            model.append(nn.BatchNorm2d(channels_in, affine=True))

        model.append(nn.LeakyReLU(inplace=False))
        model.append(nn.Conv2d(channels_in, channels_out, 3, 2, 1, 1, channels_in if groups else 1, bias=bias))

        if normalize:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.model(x)
        return fx
    

class UpsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, groups=False, normalize=False, dropout=0.0, bias=True):
        super(UpsamplingUnit, self).__init__()

        model = [nn.Conv2d(channels_in, channels_in, 3, 1, 1, 1, channels_in if groups else 1, bias=bias)]

        if normalize:
            model.append(nn.BatchNorm2d(channels_in, affine=True))

        model.append(nn.LeakyReLU(inplace=False))
        model.append(nn.ConvTranspose2d(channels_in, channels_out, 3, 2, 1, 1, channels_in if groups else 1, bias=bias))

        if normalize:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.model(x)
        return fx


class Analyzer(nn.Module):
    def __init__(self, channels_org=3, channels_net=8, channels_bn=16, compression_level=3, channels_expansion=1, groups=False, normalize=False, dropout=0.0, bias=False, **kwargs):
        super(Analyzer, self).__init__()

        # Initial color convertion
        down_track = [nn.Conv2d(channels_org, channels_net, 3, 1, 1, 1, channels_org if groups else 1, bias=bias)]

        down_track += [DownsamplingUnit(channels_in=channels_net * channels_expansion ** i, channels_out=channels_net * channels_expansion ** (i+1), 
                                     groups=groups, normalize=normalize, dropout=dropout, bias=bias)
                    for i in range(compression_level)]

        # Final convolution in the analysis track
        down_track.append(nn.Conv2d(channels_net * channels_expansion**compression_level, channels_bn, 3, 1, 1, 1, channels_bn if groups else 1, bias=bias))
        down_track.append(nn.Hardtanh(min_val=-127.5, max_val=127.5, inplace=False))

        self.analysis_track = nn.Sequential(*down_track)
        
        self.quantizer = Quantizer()

        self.apply(initialize_weights)

    def forward(self, x):
        y = self.analysis_track(x)

        y_q = self.quantizer(y)
        return y_q, y


class Synthesizer(nn.Module):
    def __init__(self, channels_org=3, channels_net=8, channels_bn=16, compression_level=3, channels_expansion=1, groups=False, normalize=False, dropout=0.0, bias=False, **kwargs):
        super(Synthesizer, self).__init__()

        # Initial deconvolution in the synthesis track
        up_track = [nn.Conv2d(channels_bn, channels_net * channels_expansion**compression_level, 3, 1, 1, 1, channels_bn if groups else 1, bias=bias)]
        up_track += [UpsamplingUnit(channels_in=channels_net * channels_expansion**(i+1), channels_out=channels_net * channels_expansion**i, 
                                     groups=groups, normalize=normalize, dropout=dropout, bias=bias)
                    for i in reversed(range(compression_level))]
        
        # Final color reconvertion
        up_track.append(nn.Conv2d(channels_net, channels_org, 3, 1, 1, 1, channels_org if groups else 1, bias=bias))

        self.synthesis_track = nn.Sequential(*up_track)

        self.apply(initialize_weights)

    def forward(self, x):
        x = self.synthesis_track(x)
        return x


class AutoEncoder(nn.Module):
    """ AutoEncoder encapsulates the full compression-decompression process. In this manner, the network can be trained end-to-end.
    """
    def __init__(self, channels_org=3, channels_net=8, channels_bn=16, compression_level=3, channels_expansion=1, groups=False, normalize=False, dropout=0.0, bias=False, K=4, r=3, **kwargs):
        super(AutoEncoder, self).__init__()

        self.analysis = Analyzer(channels_org, channels_net, channels_bn, compression_level, channels_expansion, groups, normalize, dropout, bias)
        self.synthesis = Synthesizer(channels_org, channels_net, channels_bn, compression_level, channels_expansion, groups, normalize, dropout, bias)
        self.fact_entropy = FactorizedEntropy(channels_bn, K, r)

    def forward(self, x):
        y_q, y = self.analysis(x)
        p_y = self.fact_entropy(y_q.detach() + 0.5) - self.fact_entropy(y_q.detach() - 0.5) + 1e-10
        p_y = torch.prod(p_y, dim=1) + 1e-10
        x_r = self.synthesis(y_q)

        return x_r, y, p_y
