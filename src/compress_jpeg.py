from atexit import register
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import zarr
from numcodecs import Blosc, register_codec
from imagecodecs.numcodecs import Jpeg2k, JpegXl
import imageio

import utils

register_codec(Jpeg2k)
register_codec(JpegXl)


def compress_image(filename, output_dir, comp_level, patch_size, source_format, destination_format, workers, is_labeled=False, batch_size=1, stitch_batches=False):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    zarr_ds = utils.ZarrDataset(root=filename, patch_size=patch_size, source_format=source_format)
    data_queue = DataLoader(zarr_ds, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True, worker_init_fn=utils.zarrdataset_worker_init)
    
    org_H, org_W = zarr_ds.get_img_shape(0)
    org_channels = zarr_ds.get_channels()

    # Output dir is actually the absolute path to the file where to store the compressed representation
    if 'memory' in destination_format.lower():
        group = zarr.group()
    else:
        group = zarr.group(output_dir, overwrite=True)
    
    group.attrs['height'] = org_H
    group.attrs['width'] = org_W
    group.attrs['org_channels'] = org_channels
    group.attrs['compression_level'] = comp_level
    group.attrs['model'] = 'jpeg'

    comp_group = group.create_group('0', overwrite=True)
    
    if not stitch_batches:
        raise ValueError('Storing multiple patches in the same jpeg image is not supported, stitch_bathces must be set to true')

    z_comp = comp_group.create_dataset('0',
            shape=(org_H, org_W, org_channels), 
            chunks=(patch_size, patch_size, org_channels),
            dtype='u1',
            compressor=Jpeg2k(level=comp_level),
            overwrite=True)
    
    for i, (x, _) in enumerate(data_queue):            
        x = x.numpy()
        for k, x_k in enumerate(x):
            _, tl_y, tl_x = utils.compute_grid(i*batch_size + k, imgs_shapes=[(org_H, org_W)], imgs_sizes=[0, len(zarr_ds)], patch_size=patch_size)
            tl_y *= patch_size
            tl_x *= patch_size

            br_y = min(tl_y + patch_size, org_H)
            br_x = min(tl_x + patch_size, org_W)
            valid_patch_size_y = br_y - tl_y
            valid_patch_size_x = br_x - tl_x
            
            z_comp[tl_y:tl_y + patch_size, tl_x:tl_x + patch_size, ...] = x_k.transpose(1, 2, 0)[:valid_patch_size_y, :valid_patch_size_x, ...]
    
    # If the output format is not zarr, and it is supported by PIL, an image is generated from the segmented image.
    # It should be used with care since this can generate a large image file.
    if 'memory' in destination_format.lower():
        return group
    
    return True


def compress(args):
    """ Compress any supported file format (zarr, or any supported by PIL) into a compressed representation in zarr format.
    """    
    logger = logging.getLogger(args.mode + '_log')
    
    if not args.source_format.startswith('.'):
        args.source_format = '.' + args.source_format
        
    if isinstance(args.input, (zarr.Group, zarr.Array, np.ndarray)):
        input_fn_list = [args.input]
    elif args.source_format.lower() not in args.input[0].lower():
        # If a directory has been passed, get all image files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: args.source_format.lower() in fn.lower(), os.listdir(args.input[0]))))
    else:
        input_fn_list = args.input
    
    if 'memory' in args.destination_format.lower():
        output_fn_list = [None for _ in range(len(input_fn_list))]
    else:
        output_fn_list = [os.path.join(args.output_dir, '%04d_comp.zarr' % i) for i in range(len(input_fn_list))]

    # Compress each file by separate. This allows to process large images    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        comp_group = compress_image( 
            filename=in_fn,
            output_dir=out_fn,  
            comp_level=args.compression_level, 
            patch_size=args.patch_size, 
            source_format=args.source_format, 
            destination_format=args.destination_format, 
            workers=args.workers, 
            is_labeled=args.is_labeled,
            batch_size=args.batch_size,
            stitch_batches=args.stitch_batches)

        yield comp_group


if __name__ == '__main__':
    args = utils.get_compress_args()
    
    utils.setup_logger(args)
    
    logger = logging.getLogger(args.mode + '_log')

    for _ in compress(args):
        logger.info('Image compressed successfully')
        
    logging.shutdown()