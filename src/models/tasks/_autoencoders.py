import logging 

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_dim=1024, dropout=0.0):
        super(PositionalEncoding, self).__init__()

        self._dropout = nn.Dropout2d(p=dropout, inplace=False)

        pos_y, pos_x = torch.meshgrid([torch.arange(max_dim)]*2)
        div_term = torch.exp(torch.arange(0, d_model//2, 2)*(-math.log(max_dim*2)/(d_model//2)))
        
        pe = torch.zeros(max_dim, max_dim, d_model)
        pe[..., 0:d_model//2:2] = torch.sin(pos_x.unsqueeze(dim=2) * div_term)
        pe[..., 1:d_model//2:2] = torch.cos(pos_x.unsqueeze(dim=2) * div_term)
        
        pe[..., d_model//2::2] = torch.sin(pos_y.unsqueeze(dim=2) * div_term)
        pe[..., (d_model//2+1)::2] = torch.cos(pos_y.unsqueeze(dim=2) * div_term)

        pe = pe.permute(2, 0, 1).unsqueeze(dim=0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        _, _, h, w = x.size()
        x = x + self.pe[..., :h, :w]
        return self._dropout(x)


class RandMasking(nn.Module):
    def __init__(self, n_masks=1, masks_size=64, disable_masking=False, **kwargs):
        super(RandMasking, self).__init__()

        self._n_masks = n_masks
        self._masks_size = masks_size
        self.disable_masking = disable_masking

    def forward(self, x):
        if self.disable_masking:
            return x
        
        b, _, h, w = x.size()

        mask_w = w // self._masks_size
        mask_h = h // self._masks_size
        
        with torch.no_grad():
            m = torch.ones((b, mask_h*mask_w), requires_grad=False)
            m_indices = torch.randint(0, mask_w*mask_h, (b, self._n_masks))
            m[(torch.arange(b).view(-1, 1).repeat(1, self._n_masks), m_indices)] = 0
            m = F.interpolate(m.view(b, 1, mask_h, mask_w), size=(h, w), mode='nearest')

        return x * m.to(x.device)


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
    def __init__(self, channels_in, channels_out, groups=False, batch_norm=False, dropout=0.0, bias=False):
        super(DownsamplingUnit, self).__init__()

        model = [nn.Conv2d(channels_in, channels_in, 3, 1, 1, 1, channels_in if groups else 1, bias=bias, padding_mode='reflect')]

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_in, affine=True))

        model.append(nn.LeakyReLU(inplace=False))
        model.append(nn.Conv2d(channels_in, channels_out, 3, 2, 1, 1, channels_in if groups else 1, bias=bias, padding_mode='reflect'))

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.model(x)
        return fx
    

class UpsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, groups=False, batch_norm=False, dropout=0.0, bias=True):
        super(UpsamplingUnit, self).__init__()

        model = [nn.Conv2d(channels_in, channels_in, 3, 1, 1, 1, channels_in if groups else 1, bias=bias, padding_mode='reflect')]

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_in, affine=True))

        model.append(nn.LeakyReLU(inplace=False))
        model.append(nn.ConvTranspose2d(channels_in, channels_out, 3, 2, 1, 1, channels_in if groups else 1, bias=bias))

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.model(x)
        return fx


class ColorEmbedding(nn.Module):
    def __init__(self, channels_org=3, channels_net=8, groups=False, bias=False, **kwargs):
        super(ColorEmbedding, self).__init__()
        self.embedding = nn.Conv2d(channels_org, channels_net, 3, 1, 1, 1, channels_org if groups else 1, bias=bias, padding_mode='reflect')

        self.apply(initialize_weights)

    def forward(self, x):
        fx = self.embedding(x)
        return fx


class Analyzer(nn.Module):
    def __init__(self, channels_net=8, channels_bn=16, compression_level=3, channels_expansion=1, groups=False, batch_norm=False, dropout=0.0, bias=False, **kwargs):
        super(Analyzer, self).__init__()

        down_track = [DownsamplingUnit(channels_in=channels_net * channels_expansion ** i, channels_out=channels_net * channels_expansion ** (i+1), 
                                     groups=groups, batch_norm=batch_norm, dropout=dropout, bias=bias)
                    for i in range(compression_level)]

        # Final convolution in the analysis track
        down_track.append(nn.Conv2d(channels_net * channels_expansion**compression_level, channels_bn, 3, 1, 1, 1, channels_bn if groups else 1, bias=bias, padding_mode='reflect'))
        down_track.append(nn.Hardtanh(min_val=-127.5, max_val=127.5, inplace=False))

        self.analysis_track = nn.Sequential(*down_track)
        
        self.quantizer = Quantizer()

        self.apply(initialize_weights)

    def forward(self, x):
        y = self.analysis_track(x)

        y_q = self.quantizer(y)
        return y_q, y


class Synthesizer(nn.Module):
    def __init__(self, channels_org=3, channels_net=8, channels_bn=16, compression_level=3, channels_expansion=1, groups=False, batch_norm=False, dropout=0.0, bias=False, **kwargs):
        super(Synthesizer, self).__init__()

        # Initial deconvolution in the synthesis track
        up_track = [nn.Conv2d(channels_bn, channels_net * channels_expansion**compression_level, 3, 1, 1, 1, channels_bn if groups else 1, bias=bias, padding_mode='reflect')]
        up_track += [UpsamplingUnit(channels_in=channels_net * channels_expansion**(i+1), channels_out=channels_net * channels_expansion**i, 
                                     groups=groups, batch_norm=batch_norm, dropout=dropout, bias=bias)
                    for i in reversed(range(compression_level))]
        
        # Final color reconvertion
        up_track.append(nn.Conv2d(channels_net, channels_org, 3, 1, 1, 1, channels_org if groups else 1, bias=bias, padding_mode='reflect'))

        self.synthesis_track = nn.Sequential(*up_track)

        self.apply(initialize_weights)
        
    def inflate(self, x, color=True):
        x_brg = []
        # DataParallel only sends 'x' to the GPU memory when the forward method is used and not for other methods
        fx = x.clone().to(self.synthesis_track[0].weight.device)
        for layer in self.synthesis_track[:-1]:
            fx = layer(fx)
            x_brg.append(fx / 127.5)
        
        if not color:
            return x_brg
        
        fx = self.synthesis_track[-1](fx)
        
        return fx, x_brg

    def forward(self, x):
        x = self.synthesis_track(x)
        return x


class AutoEncoder(nn.Module):
    """ AutoEncoder encapsulates the full compression-decompression process. In this manner, the network can be trained end-to-end.
    """
    def __init__(self, channels_org=3, channels_net=8, channels_bn=16, compression_level=3, channels_expansion=1, groups=False, batch_norm=False, dropout=0.0, bias=False, K=4, r=3, **kwargs):
        super(AutoEncoder, self).__init__()
        
        # Initial color embedding
        self.embedding = ColorEmbedding(channels_org=channels_org, channels_net=channels_net, groups=groups, bias=bias)

        self.analysis = Analyzer(channels_net=channels_net, channels_bn=channels_bn, compression_level=compression_level, channels_expansion=channels_expansion, groups=groups, batch_norm=batch_norm, dropout=dropout, bias=bias)
        
        self.synthesis = Synthesizer(channels_org=channels_org, channels_net=channels_net, channels_bn=channels_bn, compression_level=compression_level, channels_expansion=channels_expansion, groups=groups, batch_norm=batch_norm, dropout=dropout, bias=bias)
        
        self.fact_entropy = FactorizedEntropy(channels_bn, K=K, r=r)

    def forward(self, x, synthesize_only=False):
        if synthesize_only:
            return self.synthesis(x)
        
        fx = self.embedding(x)
        
        y_q, y = self.analysis(fx)
        p_y = self.fact_entropy(y_q.detach() + 0.5) - self.fact_entropy(y_q.detach() - 0.5) + 1e-10
        p_y = torch.prod(p_y, dim=1) + 1e-10
        
        x_r = self.synthesis(y_q)

        return x_r, y, p_y


class MaskedAutoEncoder(nn.Module):
    """ AutoEncoder encapsulates the full compression-decompression process. In this manner, the network can be trained end-to-end.
    """
    def __init__(self, channels_org=3, channels_net=8, channels_bn=16, compression_level=3, channels_expansion=1, groups=False, batch_norm=False, dropout=0.0, bias=False, K=4, r=3, n_masks=1, masks_size=64, **kwargs):
        super(MaskedAutoEncoder, self).__init__()

        # Initial color embedding
        self.embedding = ColorEmbedding(channels_org=channels_org, channels_net=channels_net, groups=groups, bias=bias)

        self.masking = RandMasking(n_masks=n_masks, masks_size=masks_size)
        self.pos_enc = PositionalEncoding(channels_net, max_dim=1024, dropout=dropout)

        self.analysis = Analyzer(channels_net=channels_net, channels_bn=channels_bn, compression_level=compression_level, channels_expansion=channels_expansion, groups=groups, batch_norm=batch_norm, dropout=dropout, bias=bias)
        self.synthesis = Synthesizer(channels_org=channels_org, channels_net=channels_net, channels_bn=channels_bn, compression_level=compression_level, channels_expansion=channels_expansion, groups=groups, batch_norm=batch_norm, dropout=dropout, bias=bias)
        self.fact_entropy = FactorizedEntropy(channels_bn, K, r)

    def forward(self, x, synthesize_only=False):
        if synthesize_only:
            return self.synthesis(x)

        fx = self.embedding(x)
        fx = self.masking(fx)
        fx = self.pos_enc(fx)
        
        y_q, y = self.analysis(fx)
        p_y = self.fact_entropy(y_q.detach() + 0.5) - self.fact_entropy(y_q.detach() - 0.5) + 1e-10
        p_y = torch.prod(p_y, dim=1) + 1e-10

        x_r = self.synthesis(y_q)

        return x_r, y, p_y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision.transforms import ToTensor, Normalize, Compose
    
    print('Testing random crop masking')

    im = Image.open(r'C:\Users\cervaf\Documents\Datasets\Kodak\kodim21.png')
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    x = transform(im).unsqueeze(0)

    checkpoint = torch.load(r'C:\Users\cervaf\Documents\Logging\tested\autoencoder\best_ver0.5.4_74691.pth', map_location='cpu')

    masker = MaskedAutoEncoder(n_masks=20, masks_size=64, **checkpoint['args'])
    
    x_m, _, _ = masker(x)

    print('Masked tensor', x_m.size())

    plt.subplot(2, 2, 1)
    plt.imshow(x[0].permute(1, 2, 0)*0.5 + 0.5)
    plt.subplot(2, 2, 2)
    plt.imshow(x_m[0].detach().permute(1, 2, 0)*0.5 + 0.5)

    masker.eval()
    x_m, _, _ = masker(x)

    print('Masked tensor', x_m.size())

    plt.subplot(2, 2, 3)
    plt.imshow(x[0].permute(1, 2, 0)*0.5 + 0.5)
    plt.subplot(2, 2, 4)
    plt.imshow(x_m[0].detach().permute(1, 2, 0)*0.5 + 0.5)
    plt.show()
