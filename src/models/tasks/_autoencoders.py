import struct
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import GDN
from compressai.entropy_models import EntropyBottleneck

import torch
from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy, ensure_contiguous_ndarray


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


class NoneColorLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoneColorLayer, self).__init__()

    def forward(self, *args, **kwargs):
        return None


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
                                    padding=kernel_size//2,
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

        model = []
        
        if act_layer_type is not None and act_layer_type not in ['GDN']:
            model.append(_define_act_layer(act_layer_type, channels_out, 
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
            model.append(
                nn.ConvTranspose2d(channels_in, channels_in,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=kernel_size//2,
                                   output_padding=0,
                                   dilation=1,
                                   groups=channels_in if groups else 1,
                                   bias=bias))

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

        res_model.append(
            nn.ConvTranspose2d(channels_in,channels_in,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=kernel_size//2,
                               output_padding=0,
                               dilation=1,
                               groups=channels_in if groups else 1,
                               bias=bias))

        if batch_norm:
            res_model.append(nn.BatchNorm2d(channels_in, affine=True))

        res_model.append(_define_act_layer(act_layer_type, channels_in,
                                            track='synthesis'))

        if act_layer_type is not None and act_layer_type not in ['GDN']:
            res_model.append(
                nn.ConvTranspose2d(channels_in, channels_in,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=kernel_size//2,
                                   output_padding=0,
                                   dilation=1,
                                   groups=channels_in if groups else 1,
                                   bias=bias))

            if batch_norm:
                res_model.append(nn.BatchNorm2d(channels_in, affine=True))

            res_model.append(_define_act_layer(act_layer_type, channels_in,
                                               track='synthesis'))

        model = []
        if act_layer_type is not None and act_layer_type not in ['GDN']:
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

        self.res_model = nn.Sequential(*res_model)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.res_model(x)
        fx = fx + x
        fx = self.model(fx)
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

        down_track = []
        prev_channels_out = channels_org
        curr_channels_out = channels_net

        for _ in range(compression_level - 1):
            down_track.append(downsampling_op(channels_in=prev_channels_out,
                                              channels_out=curr_channels_out,
                                              kernel_size=kernel_size,
                                              groups=groups,
                                              batch_norm=batch_norm,
                                              dropout=dropout,
                                              bias=bias,
                                              act_layer_type=act_layer_type))

            prev_channels_out = curr_channels_out
            curr_channels_out = prev_channels_out * channels_expansion

        if compression_level > 0:
            down_track.append(downsampling_op(channels_in=prev_channels_out,
                                              channels_out=channels_bn,
                                              kernel_size=kernel_size,
                                              groups=groups,
                                              batch_norm=batch_norm,
                                              dropout=dropout,
                                              bias=bias,
                                              act_layer_type=None))
        else:
            down_track.append(nn.Identity())

        self.analysis_track = nn.Sequential(*down_track)

        self.apply(initialize_weights)

    def forward(self, x):
        y = self.analysis_track(x)
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
                 multiscale_analysis=False,
                 **kwargs):
        super(Synthesizer, self).__init__()

        if use_residual:
            upsampling_op = ResidualUpsamplingUnit
        else:
            upsampling_op = UpsamplingUnit

        up_track = []
        prev_channels_out = channels_bn
        curr_channels_out = (channels_net
                             *channels_expansion ** compression_level)

        for _ in range(compression_level - 1):
            up_track.append(upsampling_op(channels_in=prev_channels_out,
                                          channels_out=curr_channels_out,
                                          kernel_size=kernel_size,
                                          groups=groups,
                                          batch_norm=batch_norm,
                                          dropout=dropout,
                                          bias=bias,
                                          act_layer_type=act_layer_type))

            prev_channels_out = curr_channels_out
            curr_channels_out = prev_channels_out // channels_expansion

        if compression_level > 0:
            up_track.append(upsampling_op(channels_in=prev_channels_out,
                                          channels_out=channels_org,
                                          kernel_size=kernel_size,
                                          groups=groups,
                                          batch_norm=batch_norm,
                                          dropout=dropout,
                                          bias=bias,
                                          act_layer_type=None))
        else:
            up_track.append(nn.Identity())

        # Final color reconvertion
        self.synthesis_track = nn.Sequential(*up_track)

        if multiscale_analysis:
            color_layers = [
                nn.Sequential(
                    nn.Conv2d(channels_net * channels_expansion**i,
                              channels_org,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=kernel_size//2,
                              dilation=1,
                              groups=channels_org if groups else 1,
                              bias=bias,
                              padding_mode='reflect'))
                            for i in reversed(range(compression_level - 1))]
        else:
            color_layers = [nn.Sequential(NoneColorLayer())
                            for _ in reversed(range(compression_level - 1))]

        color_layers += [nn.Identity()]

        self.color_layers = nn.ModuleList(color_layers)

        self.rec_level = compression_level

        self.apply(initialize_weights)

    def forward(self, x):
        fx = x
        fx_brg = []
        x_r = []

        for up_layer, color_layer in zip(self.synthesis_track,
                                         self.color_layers):
            fx = up_layer(fx)
            x_r_i = color_layer(fx)

            x_r.insert(0, x_r_i)
            fx_brg.append(fx)

        return x_r, fx_brg


def setup_modules(channels_bn=192, compression_level=4, K=4, r=3,
                  enabled_modules=None,
                  **kwargs):
    if enabled_modules is None:
        enabled_modules = ['encoder', 'decoder', 'fact_ent']

    model = {}
    if 'encoder' in enabled_modules:
        model['encoder'] = Analyzer(channels_bn=channels_bn,
                                    compression_level=compression_level,
                                    **kwargs)

    if 'decoder' in enabled_modules:
        model['decoder'] = Synthesizer(channels_bn=channels_bn,
                                       compression_level=compression_level,
                                       **kwargs)

    if 'fact_ent' in enabled_modules:
        model['fact_ent'] = EntropyBottleneck(channels=channels_bn,
                                              filters=[r] * K)

    return model


def load_state_dict(model, encoder=None, decoder=None, fact_ent=None,
                    **kwargs):
    if 'encoder' in model.keys() and encoder is not None:
        model['encoder'].load_state_dict(encoder, strict=False)

    if 'decoder' in model.keys() and decoder is not None:
        model['decoder'].load_state_dict(decoder, strict=False)

    if 'fact_ent' in model.keys() and fact_ent is not None:
        model['fact_ent'].update(force=True)

        if '_quantized_cdf' in fact_ent:
            model['fact_ent']._quantized_cdf = fact_ent['_quantized_cdf']

        if '_offset' in fact_ent:
            model['fact_ent']._offset = fact_ent['_offset']

        if '_cdf_length' in fact_ent:
            model['fact_ent']._cdf_length = fact_ent['_cdf_length']

        model['fact_ent'].load_state_dict(fact_ent)


def autoencoder_from_state_dict(checkpoint, gpu=False, train=False):
    if isinstance(checkpoint, str):
        checkpoint_state = torch.load(checkpoint, map_location='cpu')
    else:
        checkpoint_state = checkpoint

    model = setup_modules(**checkpoint_state)
    load_state_dict(model, **checkpoint_state)

    # If there are more than one GPU, DataParallel handles automatically the
    # distribution of the work.
    for k in model.keys():
        model[k] = nn.DataParallel(model[k])

        if gpu and torch.cuda.is_available():
            model[k].cuda()

        if train:
            model[k].train()
        else:
            model[k].eval()

    return model


class ConvolutionalAutoencoder(Codec):
    codec_id = 'cae'
    def __init__(self, checkpoint, gpu=False):
        self.checkpoint = checkpoint
        self.gpu = gpu

        self._model = autoencoder_from_state_dict(checkpoint, gpu=gpu,
                                                  train=False)

    def encode(self, buf):
        h, w, c = buf.shape

        buf_x = torch.from_numpy(buf)
        buf_x = buf_x.permute(2, 0, 1)
        buf_x = buf_x.view(1, c, h, w)
        buf_x = buf_x.float() / 255.0

        buf_y = self._model['encoder'](buf_x)
        if isinstance(self._model['fact_ent'], torch.nn.DataParallel):
            buf_ae = self._model['fact_ent'].module.compress(buf_y)
        else:
            buf_ae = self._model['fact_ent'].compress(buf_y)

        chunk_size_code = struct.pack('>QQ', h, w)

        return chunk_size_code + buf_ae[0]

    def decode(self, buf, out=None):
        if out is not None:
            out = ensure_contiguous_ndarray(out)

        compression_level = len(self._model['decoder'].module.synthesis_track)

        h, w = struct.unpack('>QQ', buf[:16])

        buf_shape = (h // 2 ** compression_level, w // 2 ** compression_level)

        if isinstance(self._model['fact_ent'], torch.nn.DataParallel):
            buf_y_q = self._model['fact_ent'].module.decompress([buf[16:]],
                                                                size=buf_shape)
        else:
            buf_y_q = self._model['fact_ent'].decompress([buf[16:]],
                                                         size=buf_shape)

        buf_x_r, _ = self._model['decoder'](buf_y_q)

        buf_x_r = buf_x_r[0].cpu().detach()[0]
        buf_x_r = buf_x_r * 255.0
        buf_x_r = buf_x_r.clip(0, 255).to(torch.uint8)
        buf_x_r = buf_x_r.permute(1, 2, 0)
        buf_x_r = buf_x_r.numpy()
        buf_x_r = np.ascontiguousarray(buf_x_r)
        buf_x_r = ensure_contiguous_ndarray(buf_x_r)

        return ndarray_copy(buf_x_r, out)
