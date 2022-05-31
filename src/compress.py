import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import zarr
from numcodecs import Blosc

import models

import utils


def setup_network(state, use_gpu=False):
    """ Setup a neural network-based image compression model.

    Parameters
    ----------
    state : Dictionary
        A checkpoint state saved during the network training
    
    Returns
    -------
    comp_model : torch.nn.Module
        The compressor model
    """
    embedding = models.ColorEmbedding(**state['args'])
    comp_model = models.Analyzer(**state['args'])

    embedding.load_state_dict(state['embedding'])
    comp_model.load_state_dict(state['encoder'])

    comp_model = nn.Sequential(embedding, comp_model)

    if use_gpu:
        comp_model = nn.DataParallel(comp_model)
        comp_model.cuda()
    
    comp_model.eval()

    return comp_model


def compress_image(comp_model, filename, output_dir, channels_bn, comp_level, patch_size, offset, transform, source_format, destination_format, workers, is_labeled=False, batch_size=1, stitch_batches=False):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    zarr_ds = utils.ZarrDataset(root=filename, patch_size=patch_size, offset=offset, transform=transform, source_format=source_format)
    data_queue = DataLoader(zarr_ds, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True, worker_init_fn=utils.zarrdataset_worker_init)
    
    org_H, org_W = zarr_ds.get_img_shape(0)
    H, W = zarr_ds.get_shape()
    comp_patch_size = patch_size//2**comp_level

    # Output dir is actually the absolute path to the file where to store the compressed representation
    if 'memory' in destination_format.lower():
        group = zarr.group()
    else:
        group = zarr.group(output_dir, overwrite=True)
    
    group.attrs['height'] = org_H
    group.attrs['width'] = org_W
    group.attrs['compression_level'] = comp_level
    group.attrs['model'] = str(comp_model)

    comp_group = group.create_group('0', overwrite=True)
    
    if stitch_batches:
        z_comp = comp_group.create_dataset('0', 
                shape=(1, channels_bn, int(np.ceil(H/2**comp_level)), int(np.ceil(W/2**comp_level))), 
                chunks=(1, channels_bn, comp_patch_size, comp_patch_size), 
                dtype='u1', compressor=compressor)
    else:
        z_comp = comp_group.create_dataset('0', 
                shape=(len(zarr_ds), channels_bn, comp_patch_size, comp_patch_size), 
                chunks=(1, channels_bn, comp_patch_size, comp_patch_size),
                dtype='u1', compressor=compressor)
    
    with torch.no_grad():
        for i, (x, _) in enumerate(data_queue):
            y_q, _ = comp_model(x)
            y_q = y_q + 127.5
            
            y_q = y_q.round().to(torch.uint8)
            y_q = y_q.detach().cpu().numpy()

            if offset > 0:
                y_q = y_q[..., 1:-1, 1:-1]
            
            if stitch_batches:
                for k, y_k in enumerate(y_q):
                    _, tl_y, tl_x = utils.compute_grid(i*batch_size + k, imgs_shapes=[(H, W)], imgs_sizes=[0, len(zarr_ds)], patch_size=patch_size)
                    tl_y *= comp_patch_size
                    tl_x *= comp_patch_size
                    z_comp[0, ..., tl_y:tl_y + comp_patch_size, tl_x:tl_x + comp_patch_size] = y_k
            else:
                z_comp[i*batch_size:i*batch_size+x.size(0), ...] = y_q

    
    if is_labeled:
        label_group = group.create_group('1', overwrite=True)
        z_org = zarr.open(filename, 'r')
        zarr.copy(z_org['1/0'], label_group)
    
    if 'memory' in destination_format.lower():
        return group
    
    return True


def compress(args):
    """ Compress any supported file format (zarr, or any supported by PIL) into a compressed representation in zarr format.
    """    
    logger = logging.getLogger(args.mode + '_log')

    # Open checkpoint from trained model state
    state = utils.load_state(args)

    comp_model = setup_network(state, args.gpu)
    
    # Conver the single zarr file into a dataset to be iterated
    transform, _ = utils.get_zarr_transform(normalize=True)

    # Get the compression level from the model checkpoint
    comp_level = state['args']['compression_level']
    offset = (2 ** comp_level) if args.add_offset else 0

    if isinstance(args.input, (zarr.Group, zarr.Array, np.ndarray)):
        input_fn_list = [args.input]
    elif not args.input[0].lower().endswith(args.source_format.lower()):
        # If a directory has been passed, get all image files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: fn.endswith(args.source_format.lower()), os.listdir(args.input[0]))))
    else:
        input_fn_list = args.input
    
    if 'memory' in args.destination_format.lower():
        output_fn_list = [None for _ in range(len(input_fn_list))]
    else:
        output_fn_list = [os.path.join(args.output_dir, '%04d_comp.zarr' % i) for i in range(len(input_fn_list))]

    # Compress each file by separate. This allows to process large images    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        comp_group = compress_image(
            comp_model=comp_model, 
            filename=in_fn,
            output_dir=out_fn, 
            channels_bn=state['args']['channels_bn'], 
            comp_level=comp_level, 
            patch_size=args.patch_size, 
            offset=offset, 
            transform=transform, 
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