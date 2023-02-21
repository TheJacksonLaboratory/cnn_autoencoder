import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import GDN
from compressai.entropy_models import EntropyBottleneck


CAE_ACT_LAYERS = ['LeakyReLU',
                  'ReLU',
                  'GDN',
                  'Identiy']

def _define_act_layer(act_layer_type, channels_in=None, track='analysis'):
    if act_layer_type is None:
        act_layer_type = 'Identity'

    if act_layer_type == 'Identity':
        act_layer = nn.Identity()
    elif act_layer_type == 'LeakyReLU':
        act_layer = nn.LeakyReLU(inplace=False)
    elif act_layer_type == 'ReLU':
        act_layer = nn.ReLU(inplace=False)
    elif act_layer_type == 'GDN':
        act_layer = GDN(in_channels=channels_in, inverse=track=='synthesis')
    else:
        raise ValueError(f'Activation layer {act_layer_type} not supported')

    return act_layer


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data, gain=math.sqrt(2 / 1.01))

        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.01)


class DownsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, groups=False,
                 batch_norm=False,
                 dropout=0.0,
                 bias=False,
                 act_layer_type=None):
        super(DownsamplingUnit, self).__init__()

        model = []
        if act_layer_type is not None and act_layer_type not in ['GDN']:
            model.append(nn.Conv2d(channels_in, channels_in,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=kernel_size//2,
                                   dilation=1,
                                   groups=channels_in if groups else 1,
                                   bias=bias,
                                   padding_mode='reflect'))

            if batch_norm:
                model.append(nn.BatchNorm2d(channels_in, affine=True))

            model.append(_define_act_layer(act_layer_type, channels_in,
                                           track='analysis'))

        model.append(nn.Conv2d(channels_in, channels_out, 
                               kernel_size=kernel_size,
                               stride=2,
                               padding=kernel_size//2,
                               dilation=1,
                               groups=channels_in if groups else 1,
                               bias=bias,
                               padding_mode='reflect'))

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        if act_layer_type is not None:
            model.append(_define_act_layer(act_layer_type, channels_out,
                                        track='analysis'))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.model(x)
        return fx


class ResidualDownsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, groups=False,
                 batch_norm=False,
                 dropout=0.0,
                 bias=False,
                 act_layer_type=None):
        super(ResidualDownsamplingUnit, self).__init__()

        res_model = []

        res_model.append(nn.Conv2d(channels_in, channels_in, 
                                    kernel_size=kernel_size,
                                    stride=1,
                                    pading=kernel_size//2,
                                    dilation=1,
                                    groups=channels_in if groups else 1,
                                    bias=bias,
                                    padding_mode='reflect'))

        if batch_norm:
            res_model.append(nn.BatchNorm2d(channels_in, affine=True))

        res_model.append(_define_act_layer(act_layer_type, channels_in,
                                            track='analysis'))

        if act_layer_type is not None and act_layer_type not in ['GDN']:
            res_model.append(nn.Conv2d(channels_in, channels_in, 
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=kernel_size//2,
                                    dilation=1,
                                    groups=channels_in if groups else 1,
                                    bias=bias,
                                    padding_mode='reflect'))

            if batch_norm:
                res_model.append(nn.BatchNorm2d(channels_in, affine=True))

        model = [_define_act_layer(act_layer_type, channels_out, 
                                   track='analysis')]

        model.append(nn.Conv2d(channels_in, channels_out, 
                               kernel_size=kernel_size,
                               stride=2,
                               padding=kernel_size//2,
                               dilation=1,
                               groups=channels_in if groups else 1,
                               bias=bias,
                               padding_mode='reflect'))

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        if act_layer_type is not None:
            model.append(_define_act_layer(act_layer_type, channels_out,
                                           track='analysis'))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.res_model = nn.Sequential(*res_model)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.res_model(x)
        fx = fx + x
        fx = self.model(fx)
        return fx


class UpsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, groups=False,
                 batch_norm=False,
                 dropout=0.0,
                 bias=True,
                 act_layer_type=None):
        super(UpsamplingUnit, self).__init__()

        model = []

        if act_layer_type is not None and act_layer_type not in ['GDN']:
            model.append(nn.Conv2d(channels_in, channels_in,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=kernel_size//2,
                                   dilation=1,
                                   groups=channels_in if groups else 1,
                                   bias=bias,
                                   padding_mode='reflect'))

            if batch_norm:
                model.append(nn.BatchNorm2d(channels_in, affine=True))

            model.append(_define_act_layer(act_layer_type, channels_in,
                                        track='synthesis'))

        model.append(nn.ConvTranspose2d(channels_in, channels_out,
                                        kernel_size=kernel_size,
                                        stride=2,
                                        padding=kernel_size//2,
                                        output_padding=1,
                                        dilation=1,
                                        groups=channels_in if groups else 1,
                                        bias=bias))

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        if act_layer_type is not None:
            model.append(_define_act_layer(act_layer_type, channels_out,
                                           track='synthesis'))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.model(x)
        return fx


class ResidualUpsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, groups=False,
                 batch_norm=False,
                 dropout=0.0,
                 bias=True,
                 act_layer_type=None):
        super(ResidualUpsamplingUnit, self).__init__()

        res_model = []

        res_model.append(nn.Conv2d(channels_in, channels_in,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=kernel_size//2,
                                   dilation=1,
                                   groups=channels_in if groups else 1,
                                   bias=bias,
                                   padding_mode='reflect'))

        if batch_norm:
            res_model.append(nn.BatchNorm2d(channels_in, affine=True))

        res_model.append(_define_act_layer(act_layer_type, channels_in,
                                            track='synthesis'))

        if act_layer_type is not None and act_layer_type not in ['GDN']:
            res_model.append(nn.Conv2d(channels_in, channels_in,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       padding=kernel_size//2,
                                       dilation=1,
                                       groups=channels_in if groups else 1,
                                       bias=bias,
                                       padding_mode='reflect'))

            if batch_norm:
                res_model.append(nn.BatchNorm2d(channels_in, affine=True))

            res_model.append(_define_act_layer(act_layer_type, channels_in,
                                               track='synthesis'))

        model = [_define_act_layer(act_layer_type, channels_in,
                                   track='synthesis')]

        model.append(nn.ConvTranspose2d(channels_in, channels_out,
                                        kernel_size=kernel_size,
                                        stride=2,
                                        padding=kernel_size//2,
                                        output_padding=1,
                                        dilation=1,
                                        groups=channels_in if groups else 1,
                                        bias=bias))

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        if act_layer_type is not None:
            model.append(_define_act_layer(act_layer_type, channels_out,
                                           track='synthesis'))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.res_model = nn.Sequential(*res_model)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.res_model(x)
        fx = fx + x
        fx = self.model(x)
        return fx


class Analyzer(nn.Module):
    def __init__(self, channels_org=3, channels_net=8, channels_bn=16,
                 compression_level=3,
                 channels_expansion=1,
                 kernel_size=3,
                 groups=False,
                 batch_norm=False,
                 dropout=0.0,
                 bias=False,
                 use_residual=False,
                 act_layer_type=None,
                 **kwargs):
        super(Analyzer, self).__init__()

        if use_residual:
            downsampling_op = ResidualDownsamplingUnit
        else:
            downsampling_op = DownsamplingUnit

        down_track = [downsampling_op(channels_in=channels_org,
                                      channels_out=channels_net,
                                      kernel_size=kernel_size,
                                      groups=groups,
                                      batch_norm=batch_norm,
                                      dropout=dropout,
                                      bias=bias,
                                      act_layer_type=act_layer_type)]

        down_track += [downsampling_op(channels_in=channels_net
                                       * channels_expansion ** i,
                                       channels_out=channels_net
                                       * channels_expansion ** (i+1),
                                       kernel_size=kernel_size,
                                       groups=groups,
                                       batch_norm=batch_norm,
                                       dropout=dropout,
                                       bias=bias,
                                       act_layer_type=act_layer_type)
                      for i in range(1, compression_level - 1)]

        # Final convolution in the analysis track
        down_track.append(nn.Conv2d(channels_net
                                    * channels_expansion**compression_level,
                                    channels_bn,
                                    kernel_size=kernel_size,
                                    stride=2,
                                    padding=kernel_size//2,
                                    dilation=1,
                                    groups=channels_bn if groups else 1,
                                    bias=bias,
                                    padding_mode='reflect'))

        # down_track.append(nn.Hardtanh(min_val=-127, max_val=127,
        #                               inplace=False))

        self.analysis_track = nn.Sequential(*down_track)

        # self.quantizer = Quantizer()

        self.apply(initialize_weights)

    def forward(self, x):
        y = self.analysis_track(x)
        # y_q = self.quantizer(y)
        return y


class Synthesizer(nn.Module):
    def __init__(self, channels_org=3, channels_net=8, channels_bn=16,
                 compression_level=3,
                 channels_expansion=1,
                 kernel_size=3,
                 groups=False,
                 batch_norm=False,
                 dropout=0.0,
                 bias=False,
                 use_residual=False,
                 act_layer_type=None,
                 **kwargs):
        super(Synthesizer, self).__init__()

        if use_residual:
            upsampling_op = ResidualUpsamplingUnit
        else:
            upsampling_op = UpsamplingUnit

        # Initial convolution in the synthesis track
        up_track = [upsampling_op(channels_in=channels_bn,
                                  channels_out=channels_net
                                  * channels_expansion**compression_level,
                                  kernel_size=kernel_size,
                                  groups=groups,
                                  batch_norm=batch_norm,
                                  dropout=dropout,
                                  bias=bias,
                                  act_layer_type=act_layer_type)]

        up_track += [upsampling_op(channels_in=channels_net
                                   * channels_expansion**(i+1),
                                   channels_out=channels_net
                                   * channels_expansion**i,
                                   kernel_size=kernel_size,
                                   groups=groups,
                                   batch_norm=batch_norm,
                                   dropout=dropout,
                                   bias=bias,
                                   act_layer_type=act_layer_type)
                     for i in reversed(range(1, compression_level - 1))]

        up_track += [upsampling_op(channels_in=channels_net,
                                   channels_out=channels_org,
                                   kernel_size=kernel_size,
                                   groups=groups,
                                   batch_norm=batch_norm,
                                   dropout=dropout,
                                   bias=bias,
                                   act_layer_type=None)]

        # Final color reconvertion
        self.synthesis_track = nn.Sequential(*up_track)

        color_layers = [
            nn.Sequential(
                 nn.Conv2d(channels_net * channels_expansion**i, channels_org,
                           kernel_size=kernel_size,
                           stride=1,
                           padding=kernel_size//2,
                           dilation=1,
                           groups=channels_org if groups else 1,
                           bias=bias,
                           padding_mode='reflect')
                )
             for i in reversed(range(compression_level - 1))]

        color_layers += [nn.Identity()]

        self.color_layers = nn.ModuleList(color_layers)

        self.rec_level = compression_level

        self.apply(initialize_weights)

    def inflate(self, x, color=True):
        x_brg = []
        # DataParallel only sends 'x' to the GPU memory when the forward method
        # is used and not for other methods
        fx = x.clone().to(self.synthesis_track[0].weight.device)
        for layer in self.synthesis_track:
            fx = layer(fx)
            x_brg.append(fx)

        if not color:
            return x_brg

        fx = self.color_layers[-1](fx)
        return fx, x_brg

    def forward(self, x):
        x = self.synthesis_track(x)
        x = self.color_layers[-1](x)
        # x.clip_(self._min_range, self._max_range)
        return x


class SynthesizerInflate(Synthesizer):
    def __init__(self, rec_level=-1, color=True, **kwargs):
        super(SynthesizerInflate, self).__init__(**kwargs)
        if rec_level < 1:
            rec_level = len(self.synthesis_track)

        if not color:
            self.color_layers = nn.ModuleList([nn.Identity()] * rec_level)

        self.rec_level = rec_level

    def forward(self, x):
        fx = x
        x_r_ms = []

        for up_layer, color_layer in zip(self.synthesis_track,
                                         self.color_layers):
            fx = up_layer(fx)
            x_r = color_layer(fx)
            # x_r.clip_(self._min_range, self._max_range)
            x_r_ms.insert(0, x_r)

        return x_r_ms


class AutoEncoder(nn.Module):
    """ AutoEncoder encapsulates the full compression-decompression process.
    In this manner, the network can be trained end-to-end, but its modules can
    be saved and accessed separatelly
    """
    def __init__(self, channels_org=3, channels_net=8, channels_bn=16,
                 compression_level=3,
                 channels_expansion=1,
                 kernel_size=3,
                 groups=False,
                 batch_norm=False,
                 dropout=0.0,
                 bias=False,
                 use_residual=False,
                 act_layer_type=None,
                 multiscale_analysis=False,
                 K=4,
                 r=3,
                 **kwargs):
        super(AutoEncoder, self).__init__()

        # Initial color embedding
        self.embedding = nn.Identity()

        self.analysis = Analyzer(channels_net=channels_net,
                                 channels_bn=channels_bn,
                                 compression_level=compression_level,
                                 channels_expansion=channels_expansion,
                                 kernel_size=kernel_size,
                                 groups=groups,
                                 batch_norm=batch_norm,
                                 dropout=dropout,
                                 bias=bias,
                                 use_residual=use_residual,
                                 act_layer_type=act_layer_type)

        if multiscale_analysis:
            synthesizer_class = SynthesizerInflate
        else:
            synthesizer_class = Synthesizer

        self.synthesis = synthesizer_class(
            channels_org=channels_org,
            channels_net=channels_net,
            channels_bn=channels_bn,
            compression_level=compression_level,
            channels_expansion=channels_expansion,
            kernel_size=kernel_size,
            groups=groups,
            batch_norm=batch_norm,
            dropout=dropout,
            bias=bias,
            use_residual=use_residual,
            act_layer_type=act_layer_type)

        self.fact_ent = EntropyBottleneck(channels=channels_bn,
                                              filters=[r] * K)

    def forward(self, x, synthesize_only=False, factorized_entropy_only=False):
        if synthesize_only:
            # When running on synthesize only mode, use x as y_q
            x_r = self.synthesis(x)
            return x_r

        elif factorized_entropy_only:
            # When running on factorized entropy only mode, use x as y_q
            # log_p_y = self.fact_ent(x)
            log_p_y = self.fact_ent._logits_cumulative(x, stop_gradient=True)
            return log_p_y

        fx = self.embedding(x)

        y = self.analysis(fx)

        y_q, p_y = self.fact_ent(y)

        x_r = self.synthesis(y_q)

        return x_r, y, p_y


def setup_autoencoder_modules(channels_bn=16, K=4, r=3,
                              multiscale_analysis=False,
                              **kwargs):

    encoder = Analyzer(channels_bn=channels_bn, **kwargs)

    if multiscale_analysis:
        synthesizer_class = SynthesizerInflate
    else:
        synthesizer_class = Synthesizer

    decoder = synthesizer_class(channels_bn=channels_bn, 
                                **kwargs)

    fact_ent = EntropyBottleneck(channels=channels_bn, 
                                           filters=[r] * K)

    return dict(encoder=encoder, decoder=decoder,
                fact_ent=fact_ent)


CAE_MODELS = {
    "AutoEncoder": AutoEncoder
}