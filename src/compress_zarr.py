import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import zarr
from numcodecs import Blosc

import models

import utils as utils


def compress(args):
    """ Compress a whole image into a compressed zarr format.
    The image must be provided as a zarr file
    """
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)
    
    logger = logging.getLogger(args.mode + '_log')

    state = utils.load_state(args)

    comp_model = models.Analyzer(**state['args'])

    comp_model.load_state_dict(state['encoder'])

    comp_model = nn.DataParallel(comp_model)
    if torch.cuda.is_available():
        comp_model.cuda()
    
    comp_model.eval()

    # Conver the single zarr file into a dataset to be iterated
    transform = utils.get_histo_transform()

    comp_level = state['args']['compression_level']
    offset = 2**comp_level

    logger.info('Openning zarr file from {}'.format(args.input))
    
    if not args.input[0].endswith('.zarr'):
        # If a directory has been passed, get all zarr files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: fn.endswith('.zarr'), os.listdir(args.input[0]))))
        output_fn_list = list(map(lambda fn: os.path.join(args.output_dir, fn + '_comp.zarr'), map(lambda fn: os.path.splitext(os.path.basename(fn))[0], input_fn_list)))
    else:
        input_fn_list = args.input
        output_fn_list = args.output_dir
    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        histo_ds = utils.Histology_seg_zarr(root=in_fn, patch_size=args.patch_size, offset=offset, transform=transform)
        data_queue = DataLoader(histo_ds, batch_size=1, num_workers=args.workers, shuffle=False, pin_memory=True)
        
        H, W = histo_ds._z_list[0].shape[-2:]
        comp_patch_size = args.patch_size//2**comp_level        

        # Output dir is actually the absolute path to the file where to store the compressed representation
        group = zarr.group(out_fn, overwrite=True)
        comp_group = group.create_group('0', overwrite=True)

        z_comp = zarr.zeros((1, state['args']['channels_bn'], int(np.ceil(H/2**comp_level)), int(np.ceil(W/2**comp_level))), chunks=(1, state['args']['channels_bn'], comp_patch_size, comp_patch_size), compressor=compressor)
        comp_group.create_dataset('0', data=z_comp, compression=compressor)

        with torch.no_grad():
            for i, (x, _) in enumerate(data_queue):
                y_q, _ = comp_model(x)
                y_q = y_q + 127.5
                y_q = y_q.round().to(torch.uint8)

                y_q = y_q.detach().cpu().numpy()
                y_q = y_q[..., 1:-1, 1:-1]

                _, tl_y, tl_x = histo_ds._compute_grid(i)
                tl_y //= 2**comp_level
                tl_x //= 2**comp_level
                z_comp[..., tl_y:(tl_y+comp_patch_size), tl_x:(tl_x+comp_patch_size)] = y_q

        logger.info(' Compressed file of size{} into {}'.format(histo_ds._z_list[0].shape, z_comp.shape))
        

if __name__ == '__main__':
    args = utils.get_compress_args()
    
    utils.setup_logger(args)
    
    compress(args)
    
    logging.shutdown()
