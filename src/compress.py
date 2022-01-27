import logging
import os

import numpy as np
import torch
import torch.nn as nn

import zarr
from numcodecs import Blosc

import models

import utils


def compress(args):
    """ Compress a list of images into binary files.
    The images can be provided as a lis of tensors, or a tensor stacked in the first dimension.
    """
    logger = logging.getLogger(args.mode + '_log')

    state = utils.load_state(args)

    comp_model = models.Analyzer(**state['args'])

    comp_model.load_state_dict(state['encoder'])

    comp_model = nn.DataParallel(comp_model)
    if torch.cuda.is_available():
        comp_model.cuda()
    
    comp_model.eval()

    encoder = utils.Encoder()

    for i, fn in enumerate(args.input):
        x = utils.open_image(fn, state['args']['compression_level'])

        with torch.no_grad():
            y_q, _ = comp_model(x)
            y_q = y_q + 127.5
            y_q = y_q.round().to(torch.int16)

        logger.info('Compressed representation: {} in [{}, {}], from [{}, {}]'.format(y_q.size(), y_q.min(), y_q.max(), x.min(), x.max()))

        # Save the compressed representation as the output of the cnn autoencoder
        if args.store_pth:
            torch.save(y_q, os.path.join(args.output_dir, '{:03d}.pth'.format(i)))

        y_b = encoder(y_q.cpu())
                
        zarr.save(os.path.join(args.output_dir, '{:03d}.zarr'.format(i)), y_b)


if __name__ == '__main__':
    args = utils.get_compress_args()
    
    utils.setup_logger(args)
    
    compress(args)
    
    logging.shutdown()