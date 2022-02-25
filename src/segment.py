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

seg_model_types = {"UNetNoBridge": models.UNetNoBridge, "UNet": models.UNet, "DecoderUNet": models.DecoderUNet}


def segment(args):
    """ Segment the objects in the images into a set of learned classes.    
    """
    logger = logging.getLogger(args.mode + '_log')

    state = utils.load_state(args)
    
    # The argument passed defines the size of the patches that will form the final segmentation
    out_patch_size = args.patch_size
    input_offset = 0

    if state['args']['model_type'] == 'DecoderUNet':
        # The compressed representation does not require normalization into the [0, 1] range
        transform = utils.get_histology_transform(normalize=False)
        
        # Find the size of the compressed patches in the checkpoint file
        in_patch_size = state['args']['patch_size']
        decompression_level = state['args']['compression_level']
        
        if args.add_offset:
            input_offset = 1
                    
    elif state['args']['model_type'] in ['UNet', 'UNetNoBridge']:
        transform = utils.get_histology_transform(normalize=True)
        in_patch_size = args.patch_size
        
        # The segmentation output is the same size of the input
        decompression_level = 0
        
        if args.add_offset:
            input_offset = 2 ** state['args']['compression_level']
        
    if args.add_offset:
        # Find the compression level in the checkpoint file
        output_offset = 2 ** state['args']['compression_level']
    
    seg_model = seg_model_types[state['args']['model_type']](**state['args'])
    seg_model.load_state_dict(state['model'])
    
    logger.debug(seg_model)
    
    seg_model = nn.DataParallel(seg_model)
    if torch.cuda.is_available():
        seg_model.cuda()
    
    seg_model.eval()
    
    # Conver the single zarr file into a dataset to be iterated    
    logger.info('Openning zarr file from {}'.format(args.input))

    if not args.input[0].endswith('.zarr'):
        # If a directory has been passed, get all zarr files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: fn.endswith('.zarr'), os.listdir(args.input[0]))))
        output_fn_list = list(map(lambda fn: os.path.join(args.output_dir, fn + '_seg.zarr'), map(lambda fn: os.path.splitext(os.path.basename(fn))[0], input_fn_list)))
    else:
        input_fn_list = args.input
        output_fn_list = [args.output_dir]
        
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        histo_ds = utils.Histology_zarr(root=in_fn, patch_size=in_patch_size, offset=input_offset, transform=transform)
        data_queue = DataLoader(histo_ds, batch_size=1, num_workers=args.workers, shuffle=False, pin_memory=True)
        
        H, W = histo_ds._z_list[0].shape[-2:]
        H_seg = H * 2**decompression_level
        W_seg = W * 2**decompression_level
        
        compressor = Blosc(cname='zlib', clevel=1, shuffle=Blosc.BITSHUFFLE)
        
        # Output dir is actually the absolute path to the file where to store the compressed representation
        group = zarr.group(out_fn, overwrite=True)
        seg_group = group.create_group('0', overwrite=True)

        z_seg = zarr.zeros((1, state['args']['classes'], H_seg, W_seg), chunks=(1, state['args']['classes'], out_patch_size, out_patch_size), compressor=compressor, dtype=np.float32)
        
        with torch.no_grad():
            for i, (x, _) in enumerate(data_queue):
                y = seg_model(x)
                
                logger.info('Network prediction shape: {}'.format(y.size()))
                y = y.detach().cpu().numpy()
                if args.add_offset > 0:
                    logger.info('Offsetted prediction shape: {}'.format(y[..., output_offset:-output_offset, output_offset:-output_offset].shape))
                    y = y[..., output_offset:-output_offset, output_offset:-output_offset]
                
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