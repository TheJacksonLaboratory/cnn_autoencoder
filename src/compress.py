import logging
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import zarr
from numcodecs import Blosc

import models

import utils


def compress_image(comp_model, filename, output_dir, channels_bn, comp_level, patch_size, offset, transform, source_format, workers, is_labeled=False, batch_size=1):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    histo_ds = utils.ZarrDataset(root=filename, patch_size=patch_size, offset=offset, transform=transform, source_format=source_format)
    data_queue = DataLoader(histo_ds, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)
    
    H, W = histo_ds.get_shape()
    comp_patch_size = patch_size//2**comp_level

    # Output dir is actually the absolute path to the file where to store the compressed representation
    group = zarr.group(output_dir, overwrite=True)
    comp_group = group.create_group('0', overwrite=True)
    
    z_comp = comp_group.create_dataset('0', shape=(1, channels_bn, int(np.ceil(H/2**comp_level)), int(np.ceil(W/2**comp_level))), chunks=(1, channels_bn, comp_patch_size, comp_patch_size), dtype='u1', compressor=compressor)

    with torch.no_grad():
        for i, (x, _) in enumerate(data_queue):
            y_q, _ = comp_model(x)
            y_q = y_q + 127.5
            
            y_q = y_q.round().to(torch.uint8)
            y_q = y_q.detach().cpu().numpy()

            if offset > 0:
                y_q = y_q[..., 1:-1, 1:-1]
            
            for k, y_k in enumerate(y_q):
                _, tl_y, tl_x = utils.compute_grid(i*batch_size + k, imgs_shapes=[(H, W)], imgs_sizes=[0, len(histo_ds)], patch_size=patch_size)
                tl_y *= comp_patch_size
                tl_x *= comp_patch_size
                z_comp[0, ..., tl_y:tl_y + comp_patch_size, tl_x:tl_x + comp_patch_size] = y_k
    
    if is_labeled:
        label_group = group.create_group('1', overwrite=True)
        z_org = zarr.open(filename, 'r')
        zarr.copy(z_org['1/0'], label_group)
    

def compress(args):
    """ Compress any supported file format (zarr, or any supported by PIL) into a compressed representation in zarr format.
    """    
    logger = logging.getLogger(args.mode + '_log')

    # Open checkpoint from trained model state
    state = utils.load_state(args)

    embedding = models.ColorEmbedding(**state['args'])
    comp_model = models.Analyzer(**state['args'])

    embedding.load_state_dict(state['embedding'])
    comp_model.load_state_dict(state['encoder'])

    comp_model = nn.Sequential(embedding, comp_model)

    if torch.cuda.is_available() and args.gpu:
        comp_model = nn.DataParallel(comp_model)
        comp_model.cuda()
    
    comp_model.eval()

    logger.debug('Model')
    logger.debug(comp_model)
    
    # Conver the single zarr file into a dataset to be iterated
    transform, _ = utils.get_zarr_transform(normalize=True)

    # Get the compression level from the model checkpoint
    comp_level = state['args']['compression_level']
    offset = (2 ** comp_level) if args.add_offset else 0

    if not args.input[0].lower().endswith(args.source_format.lower()):
        # If a directory has been passed, get all image files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: fn.endswith(args.source_format.lower()), os.listdir(args.input[0]))))
    else:
        input_fn_list = args.input
        
    output_fn_list = [os.path.join(args.output_dir, '%04d_comp.zarr' % i) for i in range(len(input_fn_list))]

    # Compress each file by separate. This allows to process large images    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        compress_image(
            comp_model=comp_model, 
            filename=in_fn,
            output_dir=out_fn, 
            channels_bn=state['args']['channels_bn'], 
            comp_level=comp_level, 
            patch_size=args.patch_size, 
            offset=offset, 
            transform=transform, 
            source_format=args.source_format, 
            workers=args.workers, 
            is_labeled=args.is_labeled,
            batch_size=args.batch_size)


if __name__ == '__main__':
    args = utils.get_compress_args()
    
    utils.setup_logger(args)
    
    compress(args)
    
    logging.shutdown()