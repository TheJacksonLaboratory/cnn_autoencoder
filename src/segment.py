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

import utils


def segment(args):
    """ Segment the objects in the images into a set of learned classes.    
    """
    logger = logging.getLogger(args.mode + '_log')

    state = utils.load_state(args)
    
    if state['args']['model_type'] == 'DecoderUNet':
        seg_model = models.DecoderUNet(**state['args'])
        transform = utils.get_histo_transform(normalize=False)
        in_patch_size = state['args']['patch_size']
        compression_level = state['args']['compression_level']
    
    elif state['args']['model_type'] == 'UNet':
        seg_model = models.UNet(**state['args'])
        transform = utils.get_histo_transform(normalize=True)
        in_patch_size = args.patch_size
        compression_level = 0
    
    out_patch_size = args.patch_size
    
    seg_model.load_state_dict(state['model'])

    seg_model = nn.DataParallel(seg_model)
    if torch.cuda.is_available():
        seg_model.cuda()
    
    seg_model.eval()
    
    # Conver the single zarr file into a dataset to be iterated    
    logger.info('Openning zarr file from {}'.format(args.input))

    offset = 0 # 2**state['args']['compression_level']

    if not args.input[0].endswith('.zarr'):
        # If a directory has been passed, get all zarr files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: fn.endswith('.zarr'), os.listdir(args.input[0]))))
        output_fn_list = list(map(lambda fn: os.path.join(args.output_dir, fn + '_seg.zarr'), map(lambda fn: os.path.splitext(os.path.basename(fn))[0], input_fn_list)))
    else:
        input_fn_list = args.input
        output_fn_list = [args.output_dir]
        
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        histo_ds = utils.Histology_zarr(root=in_fn, patch_size=in_patch_size, offset=offset, transform=transform)
        data_queue = DataLoader(histo_ds, batch_size=1, num_workers=args.workers, shuffle=False, pin_memory=True)
        
        H, W = histo_ds._z_list[0].shape[-2:]
        H *= 2**compression_level
        W *= 2**compression_level
        
        compressor = Blosc(cname='zlib', clevel=1, shuffle=Blosc.BITSHUFFLE)
        
        # Output dir is actually the absolute path to the file where to store the compressed representation
        group = zarr.group(out_fn, overwrite=True)
        seg_group = group.create_group('0', overwrite=True)

        z_seg = zarr.zeros((1, state['args']['classes'], H, W), chunks=(1, state['args']['classes'], out_patch_size, out_patch_size), compressor=compressor, dtype=np.float32)
        
        with torch.no_grad():
            for i, (x, _) in enumerate(data_queue):
                y = seg_model(x)

                y = y.detach().cpu().numpy()
                if offset > 0:
                    y = y[..., offset:-offset, offset:-offset]
                
                _, tl_y, tl_x = histo_ds._compute_grid(i)
                tl_y *= out_patch_size
                tl_x *= out_patch_size

                z_seg[..., tl_y:(tl_y+out_patch_size), tl_x:(tl_x+out_patch_size)] = y

        seg_group.create_dataset('0', data=z_seg, compression=compressor)
        logger.info('Segmentation of file of size {} into {}, saved in {}'.format(histo_ds._z_list[0].shape, z_seg.shape, out_fn))


if __name__ == '__main__':
    args = utils.get_compress_args()
    
    utils.setup_logger(args)
    
    segment(args)
    
    logging.shutdown()