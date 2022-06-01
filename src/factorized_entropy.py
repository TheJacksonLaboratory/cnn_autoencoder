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

model_types = {
    'FactorizedEntropy': models.FactorizedEntropy,
    'FactorizedEntropyLaplace': models.FactorizedEntropyLaplace    
}


def setup_network(state, model_type, use_gpu=False):
    """ Setup a neural network-based factorized entropy model.

    Parameters
    ----------
    state : Dictionary
        A checkpoint state saved during the network training
    
    Returns
    -------
    fact_ent_model : torch.nn.Module
        The factorized entropy model
    
    channels_bn : int
        The number of channels in the compressed representation
    """
    fact_ent_model = model_types[model_type](**state['args'])
    if model_type != 'FactorizedEntropyLaplace':
        fact_ent_model.load_state_dict(state['fact_ent'])
    
    if use_gpu:
        fact_ent_model = nn.DataParallel(fact_ent_model)
        fact_ent_model.cuda()
    
    state['args']['compressed_input'] = False

    return fact_ent_model


def fact_ent_image(fact_ent_model, filename, output_dir, channels_bn, comp_level, patch_size, offset, transform, source_format, destination_format, workers, batch_size=1, stitch_batches=False):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    comp_patch_size = patch_size//2**comp_level

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    zarr_ds = utils.ZarrDataset(root=filename, patch_size=comp_patch_size, offset=1 if offset > 0 else 0, transform=transform, source_format=source_format)
    data_queue = DataLoader(zarr_ds, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True, worker_init_fn=utils.zarrdataset_worker_init)
    
    H_comp, W_comp = zarr_ds.get_shape()

    # Compute the size of the reconstructed image
    H = H_comp * 2**comp_level
    W = W_comp * 2**comp_level

    # Output dir is actually the absolute path to the file where to store the compressed representation
    if 'memory' in destination_format.lower():
        group = zarr.group()
    else:
        group = zarr.group(output_dir, overwrite=True)
    
    fact_ent_group = group.create_group('0', overwrite=True)
    
    z_fact_ent = fact_ent_group.create_dataset('0', 
                shape=(1 if stitch_batches else len(zarr_ds), channels_bn, H_comp, W_comp), 
                chunks=(1, channels_bn, comp_patch_size, comp_patch_size), 
                dtype=np.float32, compressor=compressor)

    with torch.no_grad():
        for i, (x, _) in enumerate(data_queue):
            fact_ent_model.reset(x)
            
            p_y_q = fact_ent_model(x + 0.5) - fact_ent_model(x - 0.5) + 1e-10
            p_y_q = p_y_q.detach().cpu().numpy()

            if offset > 0:
                p_y_q = p_y_q[..., 1:-1, 1:-1]
            
            if stitch_batches:
                for k, p_y_k in enumerate(p_y_q):
                    _, tl_y, tl_x = utils.compute_grid(i*batch_size + k, imgs_shapes=[(H_comp, W_comp)], imgs_sizes=[0, len(zarr_ds)], patch_size=comp_patch_size)
                    tl_y *= comp_patch_size
                    tl_x *= comp_patch_size
                    z_fact_ent[0, ..., tl_y:tl_y + comp_patch_size, tl_x:tl_x + comp_patch_size] = p_y_k
            else:
                z_fact_ent[i*batch_size:i*batch_size+x.size(0), ...] = p_y_q
    
    if 'memory' in destination_format.lower():
        return group
    
    return True


def fact_ent(args):
    """ Compute the factorized entropy using a learned model.
    """    
    logger = logging.getLogger(args.mode + '_log')

    # Open checkpoint from trained model state
    state = utils.load_state(args)

    if not hasattr(args, 'model_type'):
        args.model_type = 'FactorizedEntropy'
    
    
    fact_ent_model = setup_network(state, args.model_type, args.gpu)
    
    # Conver the single zarr file into a dataset to be iterated
    transform, _ = utils.get_zarr_transform(normalize=True, compressed_input=True)

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
        output_fn_list = [os.path.join(args.output_dir, '%04d_entropy.zarr' % i) for i in range(len(input_fn_list))]

    # Compress each file by separate. This allows to process large images    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        fact_ent_group = fact_ent_image(
            fact_ent_model=fact_ent_model, 
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
            batch_size=args.batch_size,
            stitch_batches=args.stitch_batches)

        yield fact_ent_group


if __name__ == '__main__':
    args = utils.get_fact_ent_args()
    
    utils.setup_logger(args)
    
    logger = logging.getLogger(args.mode + '_log')

    for _ in fact_ent(args):
        logger.info('Computed the factorized entropy from this image successfully')
    
    logging.shutdown()