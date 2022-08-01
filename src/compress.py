from genericpath import isdir
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


COMP_VERSION='0.1'


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

    comp_model = nn.DataParallel(comp_model)
    if use_gpu:
        comp_model.cuda()
    
    comp_model.eval()

    return comp_model


def compress_image(comp_model, input_filename, output_filename, channels_bn, comp_level, 
        patch_size=512,
        offset=0, 
        stitch_batches=False, 
        transform=None, 
        destination_format='zarr', 
        workers=0, 
        batch_size=1,
        data_mode='train',
        data_axes='TCZYX',
        data_group='0/0'):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    zarr_ds = utils.ZarrDataset(root=input_filename,
            patch_size=patch_size,
            dataset_size=-1,
            data_mode=data_mode,
            offset=offset,
            transform=transform,
            source_format='zarr',
            workers=workers,
            data_axes=data_axes,
            data_group=data_group)
    
    data_queue = DataLoader(zarr_ds, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True, worker_init_fn=utils.zarrdataset_worker_init)
    
    if workers > 0:
        data_queue_iter = iter(data_queue)
        x, _ = next(data_queue_iter)
        _, channels_org, patch_H, patch_W = x.size()
        patch_H = patch_H - offset*2
        patch_W = patch_W - offset*2
        H = patch_H
        W = patch_W

    else:
        H, W = zarr_ds.get_shape()
        channels_org = zarr_ds.get_channels()
    
    comp_patch_size = patch_size//2**comp_level

    if 'memory' in destination_format.lower():
        group = zarr.group()
    else:
        if os.path.isdir(output_filename):
            group = zarr.open_group(output_filename, mode='rw')
        else:
            group = zarr.group(output_filename)
    
    comp_group = group.create_group('compressed', overwrite=True)

    # Add metadata to the compressed zarr file
    comp_group.attrs['compression_metadata'] = dict(
        height=H,
        width=W,
        channels=channels_org,        
        compressed_channels=channels_bn,
        axes='TCZYX',
        compression_level=comp_level,
        patch_size=patch_size,
        offset=offset,
        stitch_batches=stitch_batches,
        model=str(comp_model),
        original=zarr_ds._data_group,
        version=COMP_VERSION
    )
    
    if stitch_batches:
        z_comp = comp_group.create_dataset(zarr_ds._data_group, 
                shape=(1, channels_bn, 1, int(np.ceil(H/2**comp_level)), int(np.ceil(W/2**comp_level))), 
                chunks=(1, channels_bn, 1, comp_patch_size, comp_patch_size), 
                dtype='u1', compressor=compressor)
    else:
        z_comp = comp_group.create_dataset(zarr_ds._data_group, 
                shape=(len(zarr_ds), channels_bn, 1, comp_patch_size, comp_patch_size), 
                chunks=(1, channels_bn, 1, comp_patch_size, comp_patch_size),
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
                    z_comp[0, :, 0, tl_y:tl_y + comp_patch_size, tl_x:tl_x + comp_patch_size] = y_k
            else:
                z_comp[i*batch_size:i*batch_size+x.size(0), :, 0, :, :] = y_q

    z_org = zarr.open_group(input_filename.split(';')[0], 'r')
    if 'labels' in z_org.keys() and z_org.store.path != group.store.path:
        zarr.copy(z_org, group, 'labels')
    
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
    transform, _, _ = utils.get_zarr_transform(normalize=True)

    # Get the compression level from the model checkpoint
    comp_level = state['args']['compression_level']
    offset = (2 ** comp_level) if args.add_offset else 0
    
    if not args.source_format.startswith('.'):
        args.source_format = '.' + args.source_format
    
    if isinstance(args.data_dir, (zarr.Group, zarr.Array, np.ndarray)):
        input_fn_list = [args.data_dir]
    elif args.source_format.lower() not in args.data_dir[0].lower():
        # If a directory has been passed, get all image files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.data_dir[0], fn), filter(lambda fn: args.source_format.lower() in fn.lower(), os.listdir(args.data_dir[0]))))
    else:
        input_fn_list = args.data_dir
    
    if 'memory' in args.destination_format.lower() or isinstance(args.data_dir, (zarr.Group, zarr.Array, np.ndarray)):
        output_fn_list = [None for _ in range(len(input_fn_list))]
    elif args.destination_format.lower() not in args.output_dir[0].lower():
        # If the output path is a directory, append the input filenames to it, so the compressed files have the same name as the original file
        fn_list = map(lambda fn: fn.split(args.source_format)[0].replace('\\', '/').split('/')[-1], input_fn_list)
        output_fn_list = [os.path.join(args.output_dir[0], '%s%s.zarr' % (fn, args.comp_identifier)) for fn in fn_list]
    else:
        output_fn_list = args.output_dir

    # Compress each file by separate. This allows to process large images    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        comp_group = compress_image(
            comp_model=comp_model,
            input_filename=in_fn,
            output_filename=out_fn,
            channels_bn=state['args']['channels_bn'],
            comp_level=comp_level, 
            patch_size=args.patch_size,
            offset=offset, 
            stitch_batches=args.stitch_batches, 
            transform=transform, 
            destination_format=args.destination_format, 
            workers=args.workers, 
            batch_size=args.batch_size,
            data_mode=args.data_mode,
            data_axes=args.data_axes,
            data_group=args.data_group)

        logger.info('Compressed image %s into %s' % (in_fn, out_fn))

        yield comp_group


if __name__ == '__main__':
    args = utils.get_args(task='encoder', mode='inference')
    
    utils.setup_logger(args)
    
    logger = logging.getLogger(args.mode + '_log')

    for _ in compress(args):
        logger.info('Image compressed successfully')
        
    logging.shutdown()