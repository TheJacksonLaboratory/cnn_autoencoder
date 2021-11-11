import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

        if m.bias is not None:
            nn.init.xavier_ormal_(m.bias.data)


class DownsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, groups=False, normalize=False, dropout=0.0, bias=False):
        super(DownsamplingUnit, self).__init__()

        model = [nn.Conv2d(channels_in, channels_in, 3, 1, 1, 1, channels_in if groups else 1, bias=bias)]

        if normalize:
            model.append(nn.BatchNorm2d(channels_in, affine=True))

        model.append(nn.ReLU(inplace=False))
        model.append(nn.Conv2d(channels_in, channels_out, 3, 2, 1, 1, channels_in if groups else 1, bias=bias))

        if normalize:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.ReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.model(x)
        return fx
    

class UpsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, groups=False, normalize=False, dropout=0.0, bias=False):
        super(UpsamplingUnit, self).__init__()

        model = [nn.Conv2d(channels_in, channels_in, 3, 1, 1, 1, channels_in if groups else 1, bias=bias)]

        if normalize:
            model.append(nn.BatchNorm2d(channels_in, affine=True))

        model.append(nn.ReLU(inplace=False))
        model.append(nn.ConvTranspose2d(channels_in, channels_out, 3, 2, 1, 1, channels_in if groups else 1, bias=bias))

        if normalize:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.ReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.model(x)
        return fx


class AutoEncoder(nn.Module):
    def __init__(self, channels_org, channels_net, channels_bn, compression_level=3, channels_expansion=1, groups=False, normalize=False, dropout=0.0, bias=False):
        super(AutoEncoder, self).__init__()

        # Initial color convertion
        down_track = [nn.Conv2d(channels_org, channels_net, 3, 1, 1, 1, channels_org if groups else 1, bias=bias)]

        down_track += [DownsamplingUnit(channels_in=channels_net * channels_expansion ** i, channels_out=channels_net * channels_expansion ** (i+1), 
                                     groups=groups, normalize=normalize, dropout=dropout, bias=bias)
                    for i in range(compression_level)]

        # Final convolution in the analysis track
        down_track.append(nn.Conv2d(channels_net * channels_expansion**compression_level, channels_bn, 3, 1, 1, 1, channels_bn if groups else 1, bias=bias))
        
        self.analysis_track = nn.Sequential(*down_track)
        
        # TODO: Implement quantization, and codification modules

        # Initial deconvolution in the synthesis track
        up_track = [nn.Conv2d(channels_bn, channels_net * channels_expansion**compression_level, 3, 1, 1, 1, channels_bn if groups else 1, bias=bias)]
        up_track += [UpsamplingUnit(channels_in=channels_net * channels_expansion**(i+1), channels_out=channels_net * channels_expansion**i, 
                                     groups=groups, normalize=normalize, dropout=dropout, bias=bias)
                    for i in reversed(range(compression_level))]
        
        # Final color reconvertion
        up_track.append(nn.Conv2d(channels_net, channels_org, 3, 1, 1, 1, channels_org if groups else 1, bias=bias))
        
        self.synthesis_track = nn.Sequential(*up_track)

        self.apply(initialize_weights)

    def compress(self, x):
        # Analysis track
        y = self.analysis_track(x)
        return y

    def decompress(self, y):
        # Analysis track
        x_hat = self.synthesis_track(y)
        return x_hat

    def forward(self, x):
        # Analysis track
        y = self.compress(x)

        # TODO: y_hat must be the decoded version of the quantized and coded y
        y_hat = y

        # Synthesis track
        x_hat = self.decompress(y_hat)

        return x_hat


if __name__ == '__main__':
    print('Test the autoencoder')

    net = AutoEncoder(channels_org=3, channels_net=8, channels_bn=16, compression_level=3, channels_expansion=1, groups=False, normalize=False, dropout=0.0, bias=False)

    x = torch.randn([10, 3, 480, 320])
    x_hat = net(x)

    print('Reconstruction shape:', x_hat.size())
    res = torch.sum((x - x_hat)**2)

    print('Reconstruction error (MSE):', res)
