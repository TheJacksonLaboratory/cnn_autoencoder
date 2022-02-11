import argparse
import logging
from unittest.mock import patch

#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)

        if m.bias is not None:
            nn.init.uniform_(m.bias.data).multiply_(2.0*torch.pi)


class RandGaussianProjector(nn.Module):
    """ Projection of a tensor using random gaussian vector.
    """
    def __init__(self, channels_in=48, channels_out=1000, patch_size=64, gamma=1e-2, **kwargs):
        super(RandGaussianProjector, self).__init__()

        self._proj = nn.Conv2d(channels_in, channels_out, kernel_size=(1, 1), bias=True)
        self._dim = channels_out

        if not isinstance(gamma, list):
            gamma = [gamma]
        
        self._gamma = gamma

        self.apply(initialize_weights)

    def forward(self, x):
        y_list = []
        for gamma in self._gamma:
            w = self._proj(x * gamma ** 0.5)
            y_list.append(torch.cos(w) * (2 / self._dim)**0.5)
        y = torch.stack(y_list, dim=0)
        z = torch.mean(y, (3, 4), keepdim=False)
        return z


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser('Test implementation of segmentation models')
    
    parser.add_argument('-ci', '--channels-in', type=int, dest='channels_in', help='Channels from the input', default=10)
    parser.add_argument('-co', '--channels-out', type=int, dest='channels_out', help='Channels to output (projection dimension)', default=1000)
    parser.add_argument('-ps', '--patch-size', type=int, dest='patch_size', help='Size of the compressed representation patch', default=2)
    parser.add_argument('-g', '--gamma', type=float, nargs='+', dest='gamma', help='Parameter gamma, or a list of gamma parameters for the multiscale case', default=[1e-1, 2e-1, 3e-1, 1e0])
    parser.add_argument('-ss', '--sample-size', type=int, dest='sample_size', help='Size of the sample to test', default=200)
    
    args = parser.parse_args()
    
    kernel = RandGaussianProjector(channels_in=args.channels_in, channels_out=args.channels_out, patch_size=args.patch_size, gamma=args.gamma)
    kernel.eval()

    with torch.no_grad():    
        x = torch.randn((args.sample_size, args.channels_in, args.patch_size, args.patch_size))
        z = kernel(x)
        
        print('Network output size: {}'.format(z.size()))

        x = x.permute(2, 3, 0, 1)
        x_norm = x.norm(p=2, dim=-1)**2
        
        x_norm = x_norm.unsqueeze(dim=2).repeat(1, 1, args.sample_size, 1)
        dist_2 = x_norm + x_norm.transpose(2, 3) - 2*torch.matmul(x, x.transpose(2, 3))
        print('Square distances:', dist_2.size())

        K_approx = torch.matmul(z, z.transpose(1, 2))
        for i, g in enumerate(args.gamma):
            K_true = torch.exp(-g/2 * dist_2).mean(dim=(0,1))
            plt.subplot(len(args.gamma), 1, i+1)
            plt.scatter(K_true.flatten(), K_approx[i].flatten())
            plt.plot([K_true.min(), K_true.max()], [K_true.min(), K_true.max()], '-r')
        
        plt.show()