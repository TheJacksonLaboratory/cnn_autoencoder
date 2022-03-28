import argparse
import logging 

#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class DownsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, groups=False, batch_norm=False, dropout=0.5, bias=False):
        super(DownsamplingUnit, self).__init__()

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        model = [nn.Conv2d(channels_in, channels_in, kernel_size=3, stride=1, padding=1, dilation=1, groups=channels_in if groups else 1, bias=bias, padding_mode='reflect')]

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_in, affine=True))

        model.append(nn.LeakyReLU(inplace=False))
        model.append(nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=channels_in if groups else 1, bias=bias, padding_mode='reflect'))

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
    def __init__(self, channels_in, channels_out, groups=False, batch_norm=False, dropout=0.5, bias=True):
        super(UpsamplingUnit, self).__init__()

        model = [nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=channels_in if groups else 1, bias=bias, padding_mode='reflect')]

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))
        
        model.append(nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=channels_in if groups else 1, bias=bias, padding_mode='reflect'))

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))
        
        self.model = nn.Sequential(*model)
        self.upsample = nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2, padding=0, dilation=1, groups=channels_in if groups else 1, bias=bias)

    def forward(self, x):
        fx = self.model(x)
        fx = self.upsample(fx)
        return fx


class Analyzer(nn.Module):
    def __init__(self, channels_org=3, channels_net=8, compression_level=3, channels_expansion=1, groups=False, batch_norm=False, dropout=0.0, bias=False, **kwargs):
        super(Analyzer, self).__init__()        

        # Initial color convertion
        self.embedding = nn.Sequential(nn.Conv2d(channels_org, channels_net, kernel_size=3, stride=1, padding=1, dilation=1, groups=channels_org if groups else 1, bias=bias, padding_mode='reflect'),
                                       nn.Conv2d(channels_net, channels_net, kernel_size=3, stride=1, padding=1, dilation=1, groups=channels_org if groups else 1, bias=bias, padding_mode='reflect'),
                                       nn.Conv2d(channels_net, channels_net, kernel_size=3, stride=1, padding=1, dilation=1, groups=channels_org if groups else 1, bias=bias, padding_mode='reflect'))

        down_track = [DownsamplingUnit(channels_in=channels_net * channels_expansion ** i, channels_out=channels_net * channels_expansion ** (i+1), 
                                     groups=groups, batch_norm=batch_norm, dropout=dropout, bias=bias)
                        for i in range(compression_level)
                      ]

        # Final convolution in the analysis track
        self.analysis_track = nn.ModuleList(down_track)
        
        self.apply(initialize_weights)

    def forward(self, x):
        fx = self.embedding(x)

        # Store the output of each layer as bridge connection to the synthesis track
        fx_brg_list = []
        for i, layer in enumerate(self.analysis_track):
            fx_brg_list.append(fx)
            fx = layer(fx)
            
        return fx, fx_brg_list


class Synthesizer(nn.Module):
    def __init__(self, classes=1, channels_net=8, channels_bn=48, compression_level=3, channels_expansion=1, bridge=True, groups=False, batch_norm=False, dropout=0.0, bias=False, **kwargs):
        super(Synthesizer, self).__init__()
        
        input_channels_mult = 2 if bridge else 1
        
        self.embedding = nn.ConvTranspose2d(channels_bn, channels_net * channels_expansion**(compression_level-1), kernel_size=2, stride=2, padding=0, groups=channels_bn if groups else 1, bias=bias)

        # Initial deconvolution in the synthesis track
        up_track = [UpsamplingUnit(channels_in=input_channels_mult * channels_net * channels_expansion**(i+1), channels_out=channels_net * channels_expansion**i, 
                                     groups=groups, batch_norm=batch_norm, dropout=dropout, bias=bias)
                    for i in reversed(range(compression_level-1))]
        
        self.synthesis_track = nn.ModuleList(up_track)

        # Final class prediction
        self.predict = nn.Sequential(nn.Conv2d(input_channels_mult * channels_net, channels_net, 3, 1, 1, 1, classes if groups else 1, bias=bias, padding_mode='reflect'),
                                     nn.Conv2d(channels_net, channels_net, 3, 1, 1, 1, classes if groups else 1, bias=bias, padding_mode='reflect'),
                                     nn.Conv2d(channels_net, classes, 1, 1, 0, 1, classes if groups else 1, bias=bias))

        self.apply(initialize_weights)

    def forward(self, x, x_brg):
        fx = self.embedding(x)

        for i, (layer, x_k) in enumerate(zip(self.synthesis_track, reversed(x_brg))):
            fx = torch.cat((fx, x_k), dim=1)
            fx = layer(fx)
        
        fx = torch.cat((fx, x_brg[0]), dim=1)
        y = self.predict(fx)
        return y


    def extract_features(self, x, x_brg):
        fx = self.embedding(x)

        for i, (layer, x_k) in enumerate(zip(self.synthesis_track, reversed(x_brg))):
            fx = torch.cat((fx, x_k), dim=1)
            fx = layer(fx)
        
        fx = torch.cat((fx, x_brg[0]), dim=1)
        for layer in self.predict[:-1]:
            fx = layer(fx)
        
        y = self.predict[-1](fx)
        return y, fx


class BottleNeck(nn.Module):
    def __init__(self, channels_net=8, channels_bn=48, compression_level=3, channels_expansion=1, groups=False, batch_norm=False, dropout=0.5, bias=False, **kwargs):
        super(BottleNeck, self).__init__()
        
        bottleneck = [nn.Conv2d(channels_net * channels_expansion ** compression_level, channels_bn, 3, 1, 1, 1, (channels_net * channels_expansion ** compression_level) if groups else 1, bias=bias, padding_mode='reflect')]

        if batch_norm:
            bottleneck.append(nn.BatchNorm2d(channels_bn, affine=True))

        bottleneck.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            bottleneck.append(nn.Dropout2d(dropout))
        
        bottleneck.append(nn.Conv2d(channels_bn, channels_bn, 3, 1, 1, 1, channels_bn if groups else 1, bias=bias, padding_mode='reflect'))

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
    """ U-Net model for end-to-end segmentation.
    """
    def __init__(self, channels_org=3, classes=1, channels_net=8, channels_bn=48, compression_level=3, channels_expansion=1, groups=False, batch_norm=False, dropout=0.5, bias=True, **kwargs):
        super(UNet, self).__init__()

        self.analysis = Analyzer(channels_org, channels_net, compression_level, channels_expansion, groups, batch_norm, dropout, bias)
        self.bottleneck = BottleNeck(channels_net, channels_bn, compression_level, channels_expansion, groups, batch_norm, dropout, bias)
        self.synthesis = Synthesizer(classes, channels_net, channels_bn, compression_level, channels_expansion, True, groups, batch_norm, dropout, bias)

    def forward(self, x):
        fx, fx_brg = self.analysis(x)
        fx = self.bottleneck(fx)
        y = self.synthesis(fx, fx_brg)
        return y

    def extract_features(self, x):
        fx, fx_brg = self.analysis(x)
        fx = self.bottleneck(fx)
        y, fx = self.synthesis.extract_features(fx, fx_brg)
        return y, fx

class UNetNoBridge(nn.Module):
    """ U-Net model for end-to-end segmentation without bridge connections.
    This is a custom architecture wich only end is comparing against the DecoderUNet model.
    """
    def __init__(self, channels_org=3, classes=1, channels_net=8, channels_bn=48, compression_level=3, channels_expansion=1, groups=False, batch_norm=False, dropout=0.5, bias=True, **kwargs):
        super(UNetNoBridge, self).__init__()

        self.analysis = Analyzer(channels_org, channels_net, compression_level, channels_expansion, groups, batch_norm, dropout, bias)
        self.bottleneck = BottleNeck(channels_net, channels_bn, compression_level, channels_expansion, groups, batch_norm, dropout, bias)
        self.synthesis = Synthesizer(classes, channels_net, channels_bn, compression_level, channels_expansion, False, groups, batch_norm, dropout, bias)
        self._compression_level = compression_level

    def forward(self, x):
        b, _, h, w = x.size()
        fx, _ = self.analysis(x)
        fx = self.bottleneck(fx)
        fx_brg = [torch.empty((b, 0, h // 2**s, w // 2**s), device=x.device) for s in range(self._compression_level)]
        y = self.synthesis(fx, fx_brg)
        return y

    def extract_features(self, x):
        fx, fx_brg = self.analysis(x)
        fx = self.bottleneck(fx)
        y, fx = self.synthesis.extract_features(fx, fx_brg)
        return y, fx


class DecoderUNet(nn.Module):
    """ Synthesis track from the U-Net model.
    This architecture does not implement the bridge connections.
    It is intended to work on compressed representations obtained from a compressor encoder.
    """
    def __init__(self, channels_org=3, classes=1, channels_net=8, channels_bn=48, compression_level=3, channels_expansion=1, groups=False, batch_norm=False, dropout=0.5, bias=True, use_bridge=False, **kwargs):
        super(DecoderUNet, self).__init__()        
        self.synthesis = Synthesizer(classes, channels_net, channels_bn, compression_level, channels_expansion, use_bridge, groups, batch_norm, dropout, bias)
        self._compression_level = compression_level
        self._channels_org = channels_org

    def inflate(self, x, color=False):
        """ Mimics the inflate function of a trained decoder/synthesizer model
        """
        b, _, h, w = x.size()
        fx_brg = [torch.empty((b, 0, h * 2**s, w * 2**s), device=x.device) for s in range(self._compression_level+1)]
        if color:
            return torch.zeros([b, self._channels_org, h * 2**self._compression_level, w * 2**self._compression_level]).to(x.device), fx_brg
        
        return fx_brg

    def forward(self, x, fx_brg):        
        y = self.synthesis(x, fx_brg)
        return y

    def extract_features(self, x, fx_brg):
        y, fx = self.synthesis.extract_features(x, fx_brg)
        return y, fx
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test implementation of segmentation models')
    parser.add_argument('-m', '--model', type=str, dest='model_type', help='Type of model to test', choices=['UNet', 'UNetNoBridge', 'DecoderUNet'], default='UNetNoBridge')
    parser.add_argument('-ce', '--channels-expansion', type=int, dest='channels_expansion', help='Multiplier of channels expansion in the analysis track', default=1)
    parser.add_argument('-cbn', '--channels-bottleneck', type=int, dest='channels_bn', help='Channels in the bottleneck', default=48)
    parser.add_argument('-cn', '--channels-net', type=int, dest='channels_net', help='Channels in the first layer of the network', default=8)
    parser.add_argument('-cl', '--compression', type=int, dest='compression_level', help='Compression level at the deepest layer', default=3)
    
    args = parser.parse_args()
    
    models = {'UNet': UNet, 'UNetNoBridge': UNetNoBridge, 'DecoderUNet':DecoderUNet}
    
    net = models[args.model_type](compression_level=args.compression_level, channels_net=args.channels_net, channels_bn=args.channels_bn, channels_expansion=args.channels_expansion)
    
    if args.model_type in ['UNet', 'UNetNoBridge']:
        x = torch.rand([10, 3, 64, 64])
    elif args.model_type == 'DecoderUNet':
        x = torch.rand([10, args.channels_bn, 64//2**args.compression_level, 64//2**args.compression_level])
    
    y = net(x)

    t = torch.randint_like(y, high=2)
    
    print('Network output size: {}'.format(y.size()))
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    loss = criterion(y, t)
    
    print('Loss: shape {}, value {}'.format(loss.size(), torch.mean(loss)))
    