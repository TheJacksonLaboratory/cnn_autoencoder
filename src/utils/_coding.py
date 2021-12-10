import torch
import torch.nn as nn
import torchac


def _get_uniform_cdf(n_symbols, precision=16):
    # Generate a histogram from an uniform distribution
    hist = torch.ones(size=(1, 1, 1, 1, n_symbols), dtype=torch.float32) / n_symbols
    
    # Compute the cummulative distribution function
    cdf = torch.cumsum(hist, dim=-1)
    cdf = torch.cat((torch.zeros(1, 1, 1, 1, 1), cdf), dim=-1)
    
    # Convert from float32 to int16
    cdf.mul_(2**precision)
    cdf = cdf.round().to(torch.int16)
    return cdf


class Encoder(nn.Module):
    """ The encoder implements a uniform prior for the CDF of the quantized output of a cnn.
    """
    def __init__(self, n_symbols, cdf=None):
        super(Encoder, self).__init__()

        self._n_symbols = n_symbols

        if cdf is not None:
            self._cdf = torch.load(cdf)
        else:
            self._cdf = _get_uniform_cdf(n_symbols)

    def forward(self, x):
        """ Encode x using an uniform cdf
        """
        x = x.to(dtype=torch.int16)

        cdf = torch.ones(size=(*x.size(), self._n_symbols + 1), dtype=torch.int16) * self._cdf

        x_e = torchac.encode_int16_normalized_cdf(cdf, x)

        return x_e


class Decoder(nn.Module):
    """ The encoder implements a uniform prior for the CDF of the quantized output of a cnn.
    """
    def __init__(self, n_symbols, cdf=None):
        super(Decoder, self).__init__()

        self._n_symbols = n_symbols

        if cdf is not None:
            self._cdf = torch.load(cdf)
        else:
            self._cdf = _get_uniform_cdf(n_symbols)

    def forward(self, x, size):
        """ Decode x using an uniform cdf
        """
        cdf = torch.ones(size=(*size, self._n_symbols + 1), dtype=torch.int16) * self._cdf

        x_d = torchac.decode_int16_normalized_cdf(cdf, x)

        return x_d.to(dtype=torch.float32)


if __name__ == '__main__':
    import argparse
    from PIL import Image
    from torchvision import transforms

    parser = argparse.ArgumentParser('Test encoder and decoder modules')

    parser.add_argument('-i', '--input', type=str, dest='input', help='Input image to be compressed and decompressed using the this module')
    parser.add_argument('-o', '--output', type=str, dest='output', help='Filename to save the output reconstructed image')
    
    args = parser.parse_args()

    encoder = Encoder(256)
    decoder = Decoder(256)

    im = Image.open(args.input)

    pil2ten = transforms.PILToTensor()
    ten2pil = transforms.ToPILImage()

    x = pil2ten(im).unsqueeze(dim=0)
    print('Original image:', x[0, 0, -10:, -1])

    x_e = encoder(x)
    print('Endoded tensor size:', len(x_e))
    x_d = decoder(x_e, size=x.size())
    print('Reconstructed image:', x_d[0, 0, -10:, -1])
    im_rec = ten2pil(x_d.squeeze(dim=0).to(torch.uint8))

    im_rec.save(args.output)
