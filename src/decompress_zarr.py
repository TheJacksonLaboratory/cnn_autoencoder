import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import zarr
from numcodecs import Blosc

import models

import utils as utils


def decompress(args):
    """ Decmpress a compressed representation stored in zarr format.
    """
    logger = logging.getLogger(args.mode + '_log')

    state = utils.load_state(args)

    decomp_model = models.Synthesizer(**state['args'])
    decomp_model.load_state_dict(state['decoder'])

    decomp_model = nn.DataParallel(decomp_model)
    if torch.cuda.is_available():
        decomp_model.cuda()
    
    decomp_model.eval()

    # Conver the single zarr file into a dataset to be iterated
    comp_level = state['args']['compression_level']    
    offset = 2**comp_level

    comp_patch_size = args.patch_size//2**comp_level
    histo_ds = utils.Histology_zarr(root=args.input[0], patch_size=comp_patch_size, offset=1)
    data_queue = DataLoader(histo_ds, batch_size=1, num_workers=args.workers, shuffle=False, pin_memory=True)
    
    _, _, H, W = histo_ds._z_list[0].shape
    H *= 2**comp_level
    W *= 2**comp_level
    
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)
    z_decomp = zarr.zeros((1, state['args']['channels_org'], H, W), chunks=(1, state['args']['channels_org'], args.patch_size, args.patch_size), compressor=compressor)
    
    with torch.no_grad():
        for i, (y_b, _) in enumerate(data_queue):
            y_b = y_b.to(torch.float32)
            x = decomp_model(y_b - 127.5)

            x = x.detach().cpu().numpy()
            x = x[..., offset:-offset, offset:-offset]
            
            _, tl_y, tl_x = histo_ds._compute_grid(i)
            tl_y *= 2**comp_level
            tl_x *= 2**comp_level
            z_decomp[..., tl_y:(tl_y+args.patch_size), tl_x:(tl_x+args.patch_size)] = x

    # Output dir is actually the absolute path to the file where to store the decompressed image
    zarr.save(args.output_dir, z_decomp)

    logger.info('Decompressed file from size {} into {}'.format(histo_ds._z_list[0].shape, z_decomp.shape))


if __name__ == '__main__':
    args = utils.get_decompress_args()
    
    utils.setup_logger(args)
    
    decompress(args)
    
    logging.shutdown()
