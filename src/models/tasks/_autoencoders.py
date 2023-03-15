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



class ConvolutionalAutoencoder(Codec):
    codec_id = 'cae'
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        checkpoint_state = torch.load(checkpoint, map_location='cpu')
        self._model = setup_autoencoder_modules(**checkpoint_state['args'])

        # If there are more than one GPU, DataParallel handles automatically the
        # distribution of the work.
        for k in self._model.keys():
            self._model[k] = nn.DataParallel(self._model[k])

        load_state_dict(self._model, checkpoint_state=checkpoint_state)

    def encode(self, buf):
        h, w, c = buf.shape

        buf_x = torch.from_numpy(buf).permute(2, 0, 1).view(1, c, h, w).float() / 255.0

        buf_y = self._model['encoder'](buf_x)
        if isinstance(self._model['fact_ent'], torch.nn.DataParallel):
            buf = self._model['fact_ent'].module.compress(buf_y)
        else:
            buf = self._model['fact_ent'].compress(buf_y)

        return buf[0]

    def decode(self, buf, out=None):
        if out is not None:
            h, w, c = out.shape
            out = ensure_contiguous_ndarray(out)

        compression_level = len(self._model['decoder'].module.synthesis_track)
        buf_shape = (h // 2 ** compression_level,
                        w // 2 ** compression_level)

        if isinstance(self._model['fact_ent'], torch.nn.DataParallel):
            buf_y_q = self._model['fact_ent'].module.decompress([buf], size=buf_shape)
        else:
            buf_y_q = self._model['fact_ent'].decompress([buf], size=buf_shape)

        buf_x_r = self._model['decoder'](buf_y_q)

        buf_x_r = buf_x_r.cpu().detach()[0]
        buf_x_r = buf_x_r * 255.0
        buf_x_r = buf_x_r.clip(0, 255).to(torch.uint8)
        buf_x_r = buf_x_r.permute(1, 2, 0)
        buf_x_r = buf_x_r.numpy()
        buf_x_r = np.ascontiguousarray(buf_x_r)
        buf_x_r = ensure_contiguous_ndarray(buf_x_r)
        return ndarray_copy(buf_x_r, out)


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


class EmptyBottleneck(nn.Module):
    def __init__(self, **kwargs):
        super(EmptyBottleneck, self).__init__()

    def update(self):
        pass

    def forward(self, x):
        return x, x


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

    def forward(self, x):
        x = self.synthesis_track(x)
        x = self.color_layers[-1](x)
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


def setup_autoencoder_modules(channels_bn=192, compression_level=4, K=4, r=3,
                              multiscale_analysis=False,
                              **kwargs):

    encoder = Analyzer(channels_bn=channels_bn,
                       compression_level=compression_level,
                       **kwargs)

    if multiscale_analysis:
        synthesizer_class = SynthesizerInflate
    else:
        synthesizer_class = Synthesizer

    decoder = synthesizer_class(channels_bn=channels_bn,
                                compression_level=compression_level,
                                **kwargs)

    if compression_level > 0:
        fact_ent = EntropyBottleneck(channels=channels_bn,
                                     filters=[r] * K)
    else:
        fact_ent = EmptyBottleneck()

    return dict(encoder=encoder, decoder=decoder, fact_ent=fact_ent)


def load_state_dict(model, checkpoint_state):
    for k in model.keys():
        if k in checkpoint_state:
            if k == 'fact_ent':
                model['fact_ent'].module.update(force=True)
                if '_quantized_cdf' in checkpoint_state[k]:
                    model['fact_ent'].module._quantized_cdf = checkpoint_state[k]['_quantized_cdf']
                if '_offset' in checkpoint_state[k]:
                    model['fact_ent'].module._offset = checkpoint_state[k]['_offset']
                if '_cdf_length' in checkpoint_state[k]:
                    model['fact_ent'].module._cdf_length = checkpoint_state[k]['_cdf_length']

            model[k].module.load_state_dict(checkpoint_state[k], strict=False)

    if 'fact_ent' in model.keys():
        model['fact_ent'].module.update(force=True)


CAE_MODELS = {
    "AutoEncoder": AutoEncoder
}