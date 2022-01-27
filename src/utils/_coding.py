import torch
import torch.nn as nn
import zarr
from numcodecs import Blosc


class Encoder(nn.Module):
    """ The encoder implements a uniform prior for the CDF of the quantized output of a cnn.
    """
    def __init__(self, ):
        super(Encoder, self).__init__()
        self._compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)


    def forward(self, x):
        """ Encode x using an uniform cdf
        """
        x = x.to(dtype=torch.uint8).numpy()
        x_e = zarr.array(x, chunks=(1, 1, 64, 64), compressor=self._compressor)

        return x_e


class Decoder(nn.Module):
    """ The encoder implements a uniform prior for the CDF of the quantized output of a cnn.
    """
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        """ Decode x using an uniform cdf
        """
        x_d = torch.from_numpy(x[:]).to(dtype=torch.float32)
        return x_d


if __name__ == '__main__':
    import argparse
    from PIL import Image
    from torchvision import transforms
    import os

    parser = argparse.ArgumentParser('Test encoder and decoder modules')

    parser.add_argument('-i', '--input', type=str, dest='input', help='Input image to be compressed and decompressed using the this module')
    parser.add_argument('-o', '--output-dir', type=str, dest='out_dir', help='Directory where to save the output reconstructed image')
    
    args = parser.parse_args()

    encoder = Encoder()
    decoder = Decoder()

    im = Image.open(args.input)

    pil2ten = transforms.PILToTensor()
    ten2pil = transforms.ToPILImage()

    x = pil2ten(im).unsqueeze(dim=0)
    print('Original image:', x[0, 0, -10:, -1])

    x_e = encoder(x)
    print('Endoded tensor size:', len(x_e))
    x_d = decoder(x_e)
    print('Reconstructed image:', x_d[0, 0, -10:, -1])
    im_rec = ten2pil(x_d.squeeze(dim=0).to(torch.uint8))

    zarr.save(os.path.join(args.out_dir, 'comp.zarr'), x_e)
    im_rec.save(os.path.join(args.out_dir, 'recons.png'))
