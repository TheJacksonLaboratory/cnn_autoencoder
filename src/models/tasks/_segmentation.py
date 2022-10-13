import argparse
import math

import torch
import torch.nn as nn


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class DownsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, groups=False,
                 batch_norm=False,
                 dropout=0.5,
                 bias=False):
        super(DownsamplingUnit, self).__init__()

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        model = [nn.Conv2d(channels_in, channels_in, kernel_size=3, stride=1,
                           padding=1,
                           dilation=1,
                           groups=channels_in if groups else 1,
                           bias=bias,
                           padding_mode='reflect')]

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_in, affine=True))

        model.append(nn.LeakyReLU(inplace=False))
        model.append(nn.Conv2d(channels_in, channels_out, kernel_size=3,
                               stride=1,
                               padding=1,
                               dilation=1,
                               groups=channels_in if groups else 1,
                               bias=bias,
                               padding_mode='reflect'))

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.downsample(x)
        fx = self.model(fx)
        return fx


class UpsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, groups=False,
                 batch_norm=False,
                 dropout=0.5,
                 bias=True):
        super(UpsamplingUnit, self).__init__()

        model = [nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1,
                           padding=1,
                           dilation=1,
                           groups=channels_in if groups else 1,
                           bias=bias,
                           padding_mode='reflect')]

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        model.append(nn.Conv2d(channels_out, channels_out, kernel_size=3,
                               stride=1,
                               padding=1,
                               dilation=1,
                               groups=channels_in if groups else 1,
                               bias=bias,
                               padding_mode='reflect'))

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*model)
        self.upsample = nn.ConvTranspose2d(channels_out, channels_out,
                                           kernel_size=2,
                                           stride=2,
                                           padding=0,
                                           dilation=1,
                                           groups=channels_in if groups else 1,
                                           bias=bias)

    def forward(self, x):
        fx = self.model(x)
        fx = self.upsample(fx)
        return fx


class Analyzer(nn.Module):
    def __init__(self, channels_org=3, channels_net=8, compression_level=3,
                 channels_expansion=1,
                 groups=False,
                 batch_norm=False,
                 dropout=0.0,
                 bias=False,
                 **kwargs):
        super(Analyzer, self).__init__()

        # Initial color convertion
        self.embedding = nn.Sequential(
            nn.Conv2d(channels_org, channels_net, kernel_size=3, stride=1,
                      padding=1,
                      dilation=1,
                      groups=channels_org if groups else 1,
                      bias=bias,
                      padding_mode='reflect'),
            nn.Conv2d(channels_net, channels_net, kernel_size=3, stride=1,
                      padding=1,
                      dilation=1,
                      groups=channels_org if groups else 1,
                      bias=bias,
                      padding_mode='reflect'),
            nn.Conv2d(channels_net, channels_net, kernel_size=3, stride=1,
                      padding=1,
                      dilation=1,
                      groups=channels_org if groups else 1,
                      bias=bias,
                      padding_mode='reflect'))

        down_track = [DownsamplingUnit(
            channels_in=channels_net * channels_expansion ** i,
            channels_out=channels_net * channels_expansion ** (i+1), 
            groups=groups,
            batch_norm=batch_norm,
            dropout=dropout,
            bias=bias)
                      for i in range(compression_level)]

        # Final convolution in the analysis track
        self.analysis_track = nn.ModuleList(down_track)

        self.apply(initialize_weights)

    def forward(self, x):
        fx = self.embedding(x)

        # Store the output of each layer as bridge connection to the synthesis
        # track
        fx_brg_list = []
        for i, layer in enumerate(self.analysis_track):
            fx_brg_list.append(fx)
            fx = layer(fx)

        return fx, fx_brg_list


class Synthesizer(nn.Module):
    def __init__(self, classes=1, channels_net=8, channels_bn=48,
                 compression_level=3,
                 channels_expansion=1,
                 groups=False,
                 batch_norm=False,
                 dropout=0.0,
                 autoencoder_channels_net=None,
                 autoencoder_channels_expansion=None,
                 use_bridge=True,
                 trainable_bridge=False,
                 bias=False,
                 **kwargs):
        super(Synthesizer, self).__init__()

        bridge_kernel_size = 3

        if use_bridge:
            input_channels_mult = 2
        else:
            input_channels_mult = 1

        if use_bridge and trainable_bridge:
            bridge_ops = [
                BridgeBlock(
                    autoencoder_channels_net
                    * autoencoder_channels_expansion**i,
                    channels_net * channels_expansion**i,
                    groups,
                    batch_norm,
                    dropout,
                    bias,
                    bridge_kernel_size)
                for i in reversed(range(compression_level))]
        else:
            bridge_ops = [nn.Identity()
                          for i in reversed(range(compression_level))]

        self.bridge_ops = nn.ModuleList(bridge_ops)

        self.embedding = nn.ConvTranspose2d(
            channels_bn,
            channels_net
            * channels_expansion**(compression_level-1),
            kernel_size=2,
            stride=2,
            padding=0,
            groups=channels_bn if groups else 1,
            bias=bias)

        # Initial deconvolution in the synthesis track
        up_track = [UpsamplingUnit(
            channels_in=input_channels_mult
                        * channels_net
                        * channels_expansion**(i+1),
            channels_out=channels_net * channels_expansion**i,
            groups=groups,
            batch_norm=batch_norm,
            dropout=dropout,
            bias=bias)
                    for i in reversed(range(compression_level-1))]

        self.synthesis_track = nn.ModuleList(up_track)

        # Final class prediction
        self.predict = nn.Sequential(
            nn.Conv2d(input_channels_mult * channels_net, channels_net, 3, 1,
                      1,
                      1,
                      classes if groups else 1,
                      bias=bias,
                      padding_mode='reflect'),
            nn.Conv2d(channels_net, channels_net, 3, 1, 1, 1,
                      classes if groups else 1,
                      bias=bias,
                      padding_mode='reflect'),
            nn.Conv2d(channels_net, classes, 1, 1, 0, 1,
                      classes if groups else 1,
                      bias=bias))

        self.apply(initialize_weights)

    def forward(self, x, x_brg):
        fx = self.embedding(x)

        for layer, bridge_layer, x_k in zip(self.synthesis_track
                                            + [self.predict],
                                            self.bridge_ops,
                                            reversed(x_brg)):
            fx_k = bridge_layer(x_k)
            fx = torch.cat((fx, fx_k), dim=1)
            fx = layer(fx)

        return fx

    def extract_features(self, x, x_brg):
        fx = self.embedding(x)

        for i, (layer, x_k) in enumerate(zip(self.synthesis_track,
                                             reversed(x_brg))):
            fx = torch.cat((fx, x_k), dim=1)
            fx = layer(fx)

        fx = torch.cat((fx, x_brg[0]), dim=1)
        for layer in self.predict[:-1]:
            fx = layer(fx)

        y = self.predict[-1](fx)
        return y, fx


class BridgeBlock(nn.Module):
    def __init__(self, autoencoder_channels_net=8, channels_out=48,
                 groups=False,
                 batch_norm=False,
                 dropout=0.5,
                 bias=False,
                 bridge_kernel_size=1,
                 **kwargs):
        super(BridgeBlock, self).__init__()

        bridge = [nn.Conv2d(autoencoder_channels_net, channels_out,
                            kernel_size=bridge_kernel_size,
                            stride=1,
                            padding=bridge_kernel_size // 2,
                            dilation=1,
                            groups=autoencoder_channels_net if groups else 1,
                            bias=bias,
                            padding_mode='reflect')]

        if batch_norm:
            bridge.append(nn.BatchNorm2d(channels_out, affine=True))

        bridge.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            bridge.append(nn.Dropout2d(dropout))

        bridge.append(
            nn.Conv2d(channels_out, channels_out, 3, 1, 1, 1,
                      channels_out if groups else 1,
                      bias=bias,
                      padding_mode='reflect'))

        if batch_norm:
            bridge.append(nn.BatchNorm2d(channels_out, affine=True))

        bridge.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            bridge.append(nn.Dropout2d(dropout))

        self.bridge = nn.Sequential(*bridge)

        self.apply(initialize_weights)

    def forward(self, x):
        fx = self.bridge(x)
        return fx


class BottleNeck(nn.Module):
    def __init__(self, channels_net=8, channels_bn=48, compression_level=3,
                 channels_expansion=1,
                 groups=False,
                 batch_norm=False,
                 dropout=0.5,
                 bias=False,
                 **kwargs):
        super(BottleNeck, self).__init__()

        bottleneck = [nn.Conv2d(channels_net
                                * channels_expansion ** compression_level,
                                channels_bn,
                                3,
                                1,
                                1,
                                1,
                                (channels_net
                                 * channels_expansion ** compression_level)
                                if groups else 1,
                                bias=bias,
                                padding_mode='reflect')]

        if batch_norm:
            bottleneck.append(nn.BatchNorm2d(channels_bn, affine=True))

        bottleneck.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            bottleneck.append(nn.Dropout2d(dropout))

        bottleneck.append(nn.Conv2d(channels_bn, channels_bn, 3, 1, 1, 1,
                                    channels_bn if groups else 1,
                                    bias=bias,
                                    padding_mode='reflect'))

        if batch_norm:
            bottleneck.append(nn.BatchNorm2d(channels_bn, affine=True))

        bottleneck.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            bottleneck.append(nn.Dropout2d(dropout))

        self.bottleneck = nn.Sequential(*bottleneck)

        self.apply(initialize_weights)

    def forward(self, x):
        fx = self.bottleneck(x)
        return fx


class UNet(nn.Module):
    """U-Net model for end-to-end segmentation.
    """
    def __init__(self, channels_org=3, classes=1, channels_net=8,
                 channels_bn=48,
                 compression_level=3,
                 channels_expansion=1,
                 groups=False,
                 batch_norm=False,
                 dropout=0.5,
                 bias=True,
                 **kwargs):
        super(UNet, self).__init__()

        self.analysis = Analyzer(channels_org, channels_net, compression_level,
                                 channels_expansion,
                                 groups,
                                 batch_norm,
                                 dropout,
                                 bias)
        self.bottleneck = BottleNeck(channels_net, channels_bn,
                                     compression_level,
                                     channels_expansion,
                                     groups,
                                     batch_norm,
                                     dropout,
                                     bias)
        self.synthesis = Synthesizer(classes, channels_net, channels_bn,
                                     compression_level,
                                     channels_expansion,
                                     groups,
                                     batch_norm,
                                     dropout,
                                     autoencoder_channels_net=None,
                                     use_bridge=True,
                                     trainable_bridge=False,
                                     bias=bias)

    def forward(self, x, fx_brg=None):
        fx, fx_brg = self.analysis(x)
        fx = self.bottleneck(fx)
        y = self.synthesis(fx, fx_brg)
        return y

    def extract_features(self, x, fx_brg=None):
        fx, fx_brg = self.analysis(x)
        fx = self.bottleneck(fx)
        y, fx = self.synthesis.extract_features(fx, fx_brg)
        return y, fx


class UNetNoBridge(nn.Module):
    """U-Net model for end-to-end segmentation without bridge connections.
    This is a custom architecture wich only end is comparing against the
    DecoderUNet model.
    """
    def __init__(self, channels_org=3, classes=1, channels_net=8,
                 channels_bn=48,
                 compression_level=3,
                 channels_expansion=1,
                 groups=False,
                 batch_norm=False,
                 dropout=0.5,
                 bias=True,
                 **kwargs):
        super(UNetNoBridge, self).__init__()

        self.analysis = Analyzer(channels_org, channels_net, compression_level,
                                 channels_expansion,
                                 groups,
                                 batch_norm,
                                 dropout,
                                 bias)
        self.bottleneck = BottleNeck(channels_net, channels_bn,
                                     compression_level,
                                     channels_expansion,
                                     groups,
                                     batch_norm,
                                     dropout,
                                     bias)
        self.synthesis = Synthesizer(classes, channels_net, channels_bn,
                                     compression_level,
                                     channels_expansion,
                                     groups,
                                     batch_norm,
                                     dropout,
                                     autoencoder_channels_net=None,
                                     use_bridge=False,
                                     trainable_bridge=False,
                                     bias=bias)
        self._compression_level = compression_level

    def forward(self, x, fx_brg=None):
        fx, _ = self.analysis(x)
        fx = self.bottleneck(fx)
        y = self.synthesis(fx, fx_brg)
        return y

    def extract_features(self, x, fx_brg=None):
        fx, _ = self.analysis(x)
        fx = self.bottleneck(fx)
        y, fx = self.synthesis.extract_features(fx, fx_brg)
        return y, fx


class DecoderUNet(nn.Module):
    """Also referred as J-Net, operates on compressed representations of the
    input image.
    """
    def __init__(self, channels_org=3, classes=1, channels_net=8,
                 channels_bn=48,
                 compression_level=3,
                 channels_expansion=1,
                 groups=False,
                 batch_norm=False,
                 dropout=0.5,
                 bias=True,
                 autoencoder_channels_bn=None,
                 autoencoder_channels_net=None,
                 autoencoder_channels_expansion=1,
                 use_bridge=False,
                 trainable_bridge=False,
                 **kwargs):
        super(DecoderUNet, self).__init__()
        self.bottleneck_embedding = BridgeBlock(autoencoder_channels_bn,
                                                channels_bn,
                                                groups,
                                                batch_norm,
                                                dropout,
                                                bias,
                                                1)
        self.synthesis = Synthesizer(classes, channels_net, channels_bn,
                                     compression_level,
                                     channels_expansion,
                                     groups,
                                     batch_norm,
                                     dropout,
                                     autoencoder_channels_net,
                                     autoencoder_channels_expansion,
                                     use_bridge,
                                     trainable_bridge,
                                     bias=bias)
        self._compression_level = compression_level
        self._channels_org = channels_org

    def forward(self, x, fx_brg=None):
        x_bn = self.bottleneck_embedding(x)
        y = self.synthesis(x_bn, fx_brg)
        return y

    def extract_features(self, x, fx_brg):
        x_bn = self.bottleneck_embedding(x)
        y, fx = self.synthesis.extract_features(x_bn, fx_brg)
        return y, fx


class EmptyBridge(nn.Module):
    """Mimics the inflate function of a trained decoder/synthesizer model
    """
    def __init__(self, compression_level=3, compressed_input=False, **kwargs):
        super(EmptyBridge, self).__init__()
        self._compression_level = compression_level
        if compressed_input:
            self._inflate_base = 2
            self._powers = list(reversed(range(1, compression_level + 1)))
        else:
            self._inflate_base = 0.5
            self._powers = list(range(compression_level))

    def forward(self, x):
        b, _, h, w = x.size()
        fx_brg = [torch.empty((b, 0,
                               int(h * self._inflate_base ** s),
                               int(w * self._inflate_base ** s)),
                              device=x.device, requires_grad=False)
                  for s in self._powers]
        return fx_brg
