import logging
import os

import numpy as np
from setuptools import setup
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import zarr
import dask
from PIL import Image
from numcodecs import Blosc

import models

import utils


def setup_network(state, use_gpu=False):
    """ Setup a neural network-based image decompression model.

    Parameters
    ----------
    state : Dictionary
        A checkpoint state saved during the network training
    
    Returns
    -------
    decomp_model : torch.nn.Module
        The decompressor model
    
    channels_bn : int
        The number of channels in the compressed representation
    """
    decomp_model = models.Synthesizer(**state['args'])

    decomp_model.load_state_dict(state['decoder'])

    if torch.cuda.is_available() and use_gpu:
        decomp_model = nn.DataParallel(decomp_model)
        decomp_model.cuda()
    
    decomp_model.eval()

    return decomp_model    


def decompress_image(decomp_model, filename, output_dir, channels_org, comp_level, patch_size, offset, source_format, destination_format, workers, batch_size=1):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    comp_patch_size = patch_size//2**comp_level

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    zarr_ds = utils.ZarrDataset(root=filename, patch_size=comp_patch_size, offset=1 if offset > 0 else 0, transform=None, source_format='zarr')
    data_queue = DataLoader(zarr_ds, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)
    
    H_comp, W_comp = zarr_ds.get_shape()
    H = H_comp * 2**comp_level
    W = W_comp * 2**comp_level

    if 'zarr' in destination_format.lower():
        # Output dir is actually the absolute path to the file where to store the compressed representation
        if 'memory' in destination_format.lower():
            group = zarr.group()
        else:
            group = zarr.group(output_dir, overwrite=True)
        
        comp_group = group.create_group('0', overwrite=True)
    
        z_decomp = comp_group.create_dataset('0', shape=(1, channels_org, H, W), chunks=(1, channels_org, patch_size, patch_size), dtype='u1', compressor=compressor)
    else:
        z_decomp = zarr.zeros(shape=(1, channels_org, H, W), chunks=(1, channels_org, patch_size, patch_size), dtype='u1', compressor=compressor)

    with torch.no_grad():
        for i, (y_q, _) in enumerate(data_queue):
            y_q = y_q.float() - 127.5
            x = decomp_model(y_q)
            x = 0.5 * x + 0.5

            x = (x * 255).round().clip(0, 255).to(torch.uint8)
            x = x.detach().cpu().numpy()

            if offset > 0:
                x = x[..., offset:-offset, offset:-offset]
                                    
            for k, x_k in enumerate(x):
                _, tl_y, tl_x = utils.compute_grid(i*batch_size + k, imgs_shapes=[(H, W)], imgs_sizes=[0, len(zarr_ds)], patch_size=patch_size)
                tl_y *= patch_size
                tl_x *= patch_size
                z_decomp[0, ..., tl_y:tl_y + patch_size, tl_x:tl_x + patch_size] = x_k

    # If the output format is not zarr, and it is supported by PIL, an image is generated from the segmented image.
    # It should be used with care since this can generate a large image file.
    if 'zarr' not in destination_format.lower():
        im = Image.fromarray(z_decomp[0].transpose(1, 2, 0))
        im.save(output_dir, destination_format)
    
    if 'memory' in destination_format.lower():
        return group
    
    return True


def decompress(args):
    """ Decompress a compressed representation stored in zarr format with the same model used for decompression.
    """    
    logger = logging.getLogger(args.mode + '_log')

    # Open checkpoint from trained model state
    state = utils.load_state(args)

    decomp_model = setup_network(state, args)

    # Get the compression level from the model checkpoint
    comp_level = state['args']['compression_level']
    offset = (2 ** comp_level) if args.add_offset else 0

    if isinstance(args.input, (zarr.Group, zarr.Array, np.ndarray, dask.array.core.Array)):
        input_fn_list = [args.input]
    elif not args.input[0].lower().endswith('zarr'):
        # If a directory has been passed, get all image files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: fn.lower().endswith('zarr'), os.listdir(args.input[0]))))
    else:
        input_fn_list = args.input
                
    if 'memory' in args.destination_format.lower():
        output_fn_list = [None for _ in range(len(input_fn_list))]
    else:
        output_fn_list = list(map(lambda fn: os.path.join(args.output_dir, fn + '_rec.%s' % args.destination_format), map(lambda fn: os.path.splitext(os.path.basename(fn))[0], input_fn_list)))

    # Compress each file by separate. This allows to process large images    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        decomp_group = decompress_image(
            decomp_model,
            filename=in_fn, 
            output_dir=out_fn, 
            channels_org=state['args']['channels_org'], 
            comp_level=comp_level, 
            patch_size=args.patch_size, 
            offset=offset, 
            source_format=args.source_format,
            destination_format=args.destination_format,
            workers=args.workers,
            batch_size=args.batch_size)

        yield decomp_group
    

if __name__ == '__main__':
    args = utils.get_decompress_args()
    
    utils.setup_logger(args)
    
    logger = logging.getLogger(args.mode + '_log')

    for _ in decompress(args):
        logger.info('Image decompressed successfully')
    

    logging.shutdown()