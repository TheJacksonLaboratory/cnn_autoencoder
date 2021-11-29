import torch
import torch.nn as nn
import torchac


def _get_uniform_cdf(n_symbols):
    hist = torch.ones(size=(1, 1, 1, 1, n_symbols), dtype=torch.float32) / n_symbols
    cdf = torch.cumsum(hist, dim=-1)
    cdf = torch.cat((torch.zeros(1, 1, 1, 1), cdf), dim=-1)
    return cdf


class Encoder(nn.Module):
    """ The encoder implements a uniform prior for the CDF of the quantized output of a cnn.
    """
    def __init__(self, n_symbols):
        super(Encoder, self).__init__()

        self._n_symbols = n_symbols

        self._cdf = self.get_uniform_cdf(n_symbols)

    def forward(self, x):
        """ Encode x using an uniform cdf
        """
        b, c, h, w = x.size()
        device = x.device()

        x = x.to(torch.device('cpu'))

        cdf = torch.ones(size=(b, c, h, w, self._n_symbols + 1), dtype=torch.float32) * self._cdf

        x_e = torchac.encode_float_cdf(cdf, x)

        return x_e.to(device)


class Decoder(nn.Module):
    """ The encoder implements a uniform prior for the CDF of the quantized output of a cnn.
    """
    def __init__(self, n_symbols):
        super(Decoder, self).__init__()

        self._n_symbols = n_symbols

        self._cdf = self.get_uniform_cdf(n_symbols)

    def forward(self, x):
        """ Decode x using an uniform cdf
        """
        b, c, h, w = x.size()
        device = x.device()

        x = x.to(torch.device('cpu'))

        cdf = torch.ones(size=(b, c, h, w, self._n_symbols + 1), dtype=torch.float32) * self._cdf

        x_d = torchac.decode_float_cdf(cdf, x)

        return x_d.to(device)


if __name__ == '__main__':
    import argparse
    from PIL import Image
    from torchvision import transforms

    parser = argparse.ArgumentParser('Test encoder and decoder modules')

    parser.add_argument('-i', '--input', type=str, dest='input', help='Input image to be compressed and decompressed using the this module')
    parser.add_argument('-o', '--output', type=str, dest='output', help='Filename to save the output reconstructed image')
    
    args = parser.parse_args()

    encoder = Encoder(255)
    decoder = Decoder(255)

    im = Image.open(args.input)
    pil2ten = transforms.PILToTensor()
    ten2pil = transforms.ToPILImage()
    x = pil2ten(im)

    x_e = encoder(x)
    print('Endoded tensor sie:', len(x_e))
    x_d = encoder(x_e)

    im_rec = ten2pil(x_d)

    im_rec.save(args.output)
