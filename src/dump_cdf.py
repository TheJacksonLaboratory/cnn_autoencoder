import logging
import argparse
import os

import torch
import torch.nn as nn

import models


def save_cdf(args):
    """ Extract and save the cummulative distribution function learned during the cnn autoencoder training.
    """    
    save_fn = os.path.join(args.trained_model)

    if not torch.cuda.is_available():
        state = torch.load(save_fn, map_location=torch.device('cpu'))
    else:
        state = torch.load(save_fn)

    fact_ent = models.FactorizedEntropy(**state['args'])

    fact_ent = nn.DataParallel(fact_ent)

    x = torch.arange(-1, 1, 2/512).reshape(1, 1, 512, 1).tile([1, 48, 1, 1]).float()

    if torch.cuda.is_available():
        fact_ent.cuda()
        x = x.cuda()
    
    fact_ent.eval()

    with torch.no_grad():
        p_y = fact_ent(x+0.5) - fact_ent(x-0.5)
        torch.save(p_y, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save the factorized entropy model learned during a cnn autoencoder training')

    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be tested')
    parser.add_argument('-o', '--output', type=str, dest='output', help='The output filename to store the cdf', default='cdf.pth')
    
    args = parser.parse_args()
    args.mode = 'save_cdf'

    save_cdf(args)

    logging.shutdown()