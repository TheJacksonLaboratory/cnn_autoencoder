import logging
import os
from attr import has

import numpy as np
from pyparsing import WordEnd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import zarr
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

    decomp_model = nn.DataParallel(decomp_model)

    if torch.cuda.is_available() and use_gpu:
        decomp_model.cuda()
    
    decomp_model.eval()

    return decomp_model    


def decompress_image(decomp_model, filename, output_dir, channels_org, comp_level, patch_size, offset, source_format, destination_format, workers, batch_size=1, stitch_batches=False, reconstruction_level=-1, compute_pyramids=True):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    comp_patch_size = patch_size//2**comp_level

    if reconstruction_level < 0:
        reconstruction_level = comp_level
    
    # Compute the scales of the pyramids and take only thse taht will be stored on disk
    scales = [r for r in range(comp_level-reconstruction_level, comp_level)]

    if not compute_pyramids:
        scales = [scales[0]]

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    zarr_ds = utils.ZarrDataset(root=filename, mode='all', patch_size=comp_patch_size, offset=1 if offset > 0 else 0, transform=None, source_format='.zarr', workers=workers)
    data_queue = DataLoader(zarr_ds, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True, worker_init_fn=utils.zarrdataset_worker_init)
    
    if workers > 0:
        data_queue_iter = iter(data_queue)
        x, _ = next(data_queue_iter)
        _, _, H_comp, W_comp = x.size()
        H_comp = H_comp - offset*2
        W_comp = W_comp - offset*2
        org_H = H_comp * 2**comp_level
        org_W = W_comp * 2**comp_level

    else:
        H_comp, W_comp = zarr_ds.get_shape()
        org_H, org_W = zarr_ds.get_img_original_shape(0)

    H = H_comp * 2**comp_level
    W = W_comp * 2**comp_level
    
    if '.zarr' in destination_format.lower() and 'memory' not in destination_format.lower():
        # If the output is a zarr file, but will not be kept in memory, create a group (folder) to store the output into a sub-group
        group = zarr.group(output_dir + '.zarr', overwrite=True)
    
    else:
        group = zarr.group()
        
    comp_group = group.create_group('0', overwrite=True)
    
    if stitch_batches:
        z_decomp = dict([(r, 
            comp_group.create_dataset(str(r), 
                shape=(1, channels_org, org_H//2**r, org_W//2**r),
                chunks=(1, channels_org, patch_size//2**r, patch_size//2**r), 
                dtype='u1', compressor=compressor))
            for r in scales
            ])

    else:
        z_decomp = dict([(r,             
            comp_group.create_dataset(str(r), 
                shape=(len(zarr_ds), channels_org, patch_size//2**r, patch_size//2**r),
                chunks=(1, channels_org, patch_size//2**r, patch_size//2**r), 
                dtype='u1', compressor=compressor))
            for r in scales
            ])

    with torch.no_grad():
        for i, (y_q, _) in enumerate(data_queue):
            y_q = y_q.float() - 127.5
            if reconstruction_level != comp_level or compute_pyramids:
                x = decomp_model.module.forward_steps(y_q, reconstruction_level=reconstruction_level)
                if not compute_pyramids:
                    x = [x[0]]
            else:
                x = [decomp_model(y_q)]
            
            x = [(127.5 * x_r + 127.5).round().clip(0, 255).to(torch.uint8).detach().cpu().numpy() for x_r in x]

            if offset > 0:
                x = [x_r[..., offset//2**r:-offset//2**r, offset//2**r:-offset//2**r] for r, x_r in zip(scales, x)]

            for r, x_r in zip(scales, x):
                if stitch_batches:
                    for k, x_k in enumerate(x_r):
                        _, tl_y, tl_x = utils.compute_grid(i*batch_size + k, imgs_shapes=[(H//2**r, W//2**r)], imgs_sizes=[0, len(zarr_ds)], patch_size=patch_size//2**r)
                        tl_y *= patch_size // 2**r
                        tl_x *= patch_size // 2**r
                        br_y = min(tl_y + patch_size//2**r, org_H//2**r)
                        br_x = min(tl_x + patch_size//2**r, org_W//2**r)
                        valid_patch_size_y = br_y - tl_y
                        valid_patch_size_x = br_x - tl_x

                        z_decomp[r][0, ..., tl_y:br_y, tl_x:br_x] = x_k[..., :valid_patch_size_y, :valid_patch_size_x]
                else:
                    z_decomp[r][i*batch_size:i*batch_size+y_q.size(0), ...] = x_r

    # If the output will be stored in memory instead of on disk, return the main group
    if 'memory' in destination_format.lower():
        return group
    
    # If the output format is not zarr, and it is supported by PIL, an image is generated from the segmented image.
    # It should be used with care since this can generate a large image file.
    if '.zarr' not in destination_format.lower():
        if compute_pyramids:
            output_filenames = [output_dir + str(r) for r in range(reconstruction_level)]
        else:
            output_filenames = [output_dir]
        
        for r, out_fn in zip(scales, output_filenames):
            im = Image.fromarray(z_decomp[r][0].transpose(1, 2, 0))
            im.save(out_fn + destination_format)

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

    if not args.destination_format.startswith('.'):
        args.destination_format = '.' + args.destination_format

    if isinstance(args.input, (zarr.Group, zarr.Array, np.ndarray)):
        input_fn_list = [args.input]
    elif '.zarr' not in args.input[0].lower():
        # If a directory has been passed, get all image files inside to decompress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: '.zarr' in fn.lower(), os.listdir(args.input[0]))))
    else:
        input_fn_list = args.input
                
    if 'memory' in args.destination_format.lower():
        output_fn_list = [None for _ in range(len(input_fn_list))]
    else:
        output_fn_list = list(map(lambda fn: os.path.join(args.output_dir, fn + '_rec'), map(lambda fn: os.path.splitext(os.path.basename(fn))[0], input_fn_list)))

    if not hasattr(args, 'reconstruction_level'):
        args.reconstruction_level = -1

    if not hasattr(args, 'compute_pyramids'):
        args.compute_pyramids = False

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
            batch_size=args.batch_size,
            stitch_batches=args.stitch_batches,
            reconstruction_level=args.reconstruction_level,
            compute_pyramids=args.compute_pyramids)

        yield decomp_group
    

if __name__ == '__main__':
    args = utils.get_decompress_args()
    
    utils.setup_logger(args)
    
    logger = logging.getLogger(args.mode + '_log')

    for _ in decompress(args):
        logger.info('Image decompressed successfully')
    

    logging.shutdown()