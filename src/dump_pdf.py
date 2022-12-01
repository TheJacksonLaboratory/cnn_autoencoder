import logging
import argparse
import os

import torch
import torch.nn as nn

import models


def save_cdf(args):
    """Extract and save the cummulative distribution function learned during
    the cnn autoencoder training.
    """
    save_fn = os.path.join(args.trained_model)

    if not torch.cuda.is_available():
        state = torch.load(save_fn, map_location=torch.device('cpu'))
    else:
        state = torch.load(save_fn)

    channels_bn = state['args']['channels_bn']

    fact_ent = models.FactorizedEntropy(**state['args'])
    # Load the state into the factorized entropy model
    fact_ent.load_state_dict(state['fact_ent'])

    fact_ent = nn.DataParallel(fact_ent)
    x = torch.linspace(-127.5, 127.5, 257).reshape(1, 1, 1, -1).tile([1, channels_bn, 1, 1]).float()

    if torch.cuda.is_available():
        fact_ent.cuda()
        x = x.cuda()

    fact_ent.eval()

    with torch.no_grad():
        pdf = fact_ent(x + 0.5).cpu() - fact_ent(x - 0.5).cpu()

        # Correct the squeence according to the ranges of each channel
        min_sym = []
        max_sym = []
        for ch in range(channels_bn):
            min_sym_ch = torch.min(torch.where(pdf[0, ch, 0, :] > 1e-10)[0])
            max_sym_ch = torch.max(torch.where(pdf[0, ch, 0, :] > 1e-10)[0])
            min_sym.append(min_sym_ch)
            max_sym.append(max_sym_ch)

        min_sym = torch.IntTensor(min_sym)
        max_sym = torch.IntTensor(max_sym)
        max_len = max_sym.max()

        x = torch.arange(max_len).reshape(1, 1, 1, -1).tile([1, channels_bn, 1, 1]).float()
        x = x + min_sym[None, :, None, None] - 127.5

        if torch.cuda.is_available():
            x = x.cuda()

        pdf = fact_ent(x + 0.5) - fact_ent(x - 0.5)

    pdf_info = dict(pdf=pdf.cpu(), min_sym=min_sym - 128, len_sym=max_sym - min_sym + 1)
    torch.save(pdf_info, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save the factorized entropy model learned during a cnn autoencoder training')

    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be tested')
    parser.add_argument('-o', '--output', type=str, dest='output', help='The output filename to store the cdf', default='cdf.pth')

    args = parser.parse_args()
    args.mode = 'save_cdf'

    save_cdf(args)

    logging.shutdown()
