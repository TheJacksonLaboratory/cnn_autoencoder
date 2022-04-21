import logging 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)

        if m.bias is not None:
            nn.init.uniform_(m.bias.data, 0, 2*math.pi)

class KernelLayer(nn.Module):
    def __init__(self, in_channels, out_channels, gammas=1.0):
        super(KernelLayer, self).__init__()
        
        if isinstance(gammas, float):
            gammas = [gammas]
        
        self._sqrt_gammas = [g**0.5 for g in gammas]
        self._dim_scale = (2 / out_channels)**0.5
                
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # Initialize the kernel matrix
        self.apply(initialize_weights)
    
    def forward(self, x):
        wx = []
        for sg in self._sqrt_gammas:
            z = self._dim_scale * torch.cos(self.projection(sg * x))
            z = self.avg_pooling(z).unsqueeze(dim=1)
            wx.append(z)
        return torch.cat(wx, dim=1)


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser('Test of the Kernel Canonical Correlation Analysis (KCCA)')
    parser.add_argument('-n', '--sample-size', dest='sample_size', help='Size of the sample', default=200)
    parser.add_argument('-ps', '--patch-size', dest='patch_size', help='Size of the patch to be analyzed', default=1)
    parser.add_argument('-bch', '--bnch', type=int, dest='channels_bn', help='Number of channels of the compressed representation', default=48)
    parser.add_argument('-pch', '--prjch', type=int, dest='channels_projection', help='Dimension of the projection space', default=10000)
    parser.add_argument('-g', '--gammas', type=float, nargs='+', dest='gammas', help='List of scales for multiscale analysis', default=[1e-2])

    args = parser.parse_args()

    X = torch.randn(size=(args.sample_size, args.channels_bn, args.patch_size, args.patch_size))
    X_norm_2 = torch.norm(X.permute(2, 3, 0, 1), dim=3).unsqueeze(dim=3) ** 2
    d = X_norm_2 + X_norm_2.transpose(dim0=2, dim1=3) - 2*torch.matmul(X.permute(2, 3, 0, 1), X.permute(2, 3, 0, 1).transpose(dim0=2, dim1=3))
    d = d.clip(1e-12, None)
    K_true = torch.exp(-0.5 * args.gammas[0] * torch.mean(d, dim=(0, 1)))

    print('Distance matrix', d.size())
    print('Kernel matrix', K_true.size())
    kcca = KernelLayer(in_channels=args.channels_bn, out_channels=args.channels_projection, gammas=args.gammas)

    Z = kcca(X)
    Z = Z.detach().permute(1, 3, 4, 0, 2)
    K_approx = torch.matmul(Z, Z.transpose(dim0=3, dim1=4))
    print('Kernel approximation matrix', K_approx.size())
    
    plt.scatter(K_true.flatten(), K_approx[-1, ...].flatten())
    min_x = torch.min(K_true)
    min_y = torch.min(K_approx)

    plt.plot((min_x, 1), (min_y, 1))
    plt.show()