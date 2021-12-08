import argparse
import struct 

import math
import torch

import models
import matplotlib.pyplot as plt


def open_compressed(filename):
    if '.pth' in filename:
        y_q = torch.load(filename)
    elif '.comp' in filename:
        decoder = models.Decoder(512)
        
        with open(filename, mode='rb') as f:
            # Write the size of the image:
            size_b = f.read(16)
            size = struct.unpack('IIII', size_b)

            # Write the compressed bitstream
            y_b = f.read()

        y_q = decoder(y_b, size)
    else:
        raise ValueError('File \'%s\' has an unsoported format' % filename)

    return y_q


def visualize(y_q):
    """ Visualize the compressed representation of an input image.
    """
    b, channels, h, w = y_q.size()

    k_w = int(math.ceil(math.sqrt(channels)))
    k_h = int(math.ceil(channels / k_w))

    for i in range(channels):        
        plt.subplot(k_h, k_w, i + 1)
        plt.imshow(y_q[0, i], cmap='gray')

    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize the compressed representation of an image')
    
    parser.add_argument('-i', '--input', type=str, dest='input', help='Input compressed representation in binary file .comp, or .pth')

    args = parser.parse_args()

    y_q = open_compressed(args.input)
    
    visualize(y_q)
