from functools import partial
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


def forward_undecoded_step(x, seg_model, decoder_model=None):
    y = seg_model(x)
    return y


def forward_decoded_step(x, seg_model, decoder_model=None):
    with torch.no_grad():
        _, x_brg = decoder_model.inflate(x, color=False)
    y = seg_model(x, x_brg[:0:-1])

    return y


def setup_network(args):
    """ Setup a nerual network for object segmentation.

    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. All the parameters are passed directly to the model constructor.
        This way, the constructor can take the parameters needed that have been passed by the user.
    
    Returns
    -------
    seg_model : nn.Module
        The segmentation mode implemented by a convolutional neural network
    
    forward_function : function
        The function to used for the feed-forward step
    """
    # When the model works on compressed representation, tell the dataloader to obtain the compressed input and normal size target
    if 'Decoder' in args['model_type']:
        args['compressed_input'] = True

    # If a decoder model is passed as argument, use the decoded step version of the feed-forward step
    if args['autoencoder_model'] is not None:
        if not args['gpu']:
            checkpoint_state = torch.load(args['autoencoder_model'], map_location=torch.device('cpu'))
        
        else:
            checkpoint_state = torch.load(args['autoencoder_model'])
       
        decoder_model = models.Synthesizer(**checkpoint_state['args'])
        decoder_model.load_state_dict(checkpoint_state['decoder'])

        if args['gpu']:
            decoder_model = nn.DataParallel(decoder_model)        
            decoder_model.cuda()

        decoder_model.eval()
        args['use_bridge'] = True
    else:
        args['use_bridge'] = False
    
    seg_model_class = seg_model_types.get(args['model_type'], None)
    if seg_model_class is None:
        raise ValueError('Model type %s not supported' % args['model_type'])
    
    seg_model = seg_model_class(**args)

    # If there are more than one GPU, DataParallel handles automatically the distribution of the work
    seg_model = nn.DataParallel(seg_model)
    if args['gpu']:
        seg_model.cuda()

    # Define what funtion use in the feed-forward step
    if args['autoencoder_model'] is not None:
        forward_function = partial(forward_decoded_step, seg_model=seg_model, decoder_model=decoder_model)

    elif 'Decoder' in args['model_type']:
        # If no decoder is loaded, use the inflate function inside the segmentation model
        forward_function = partial(forward_decoded_step, seg_model=seg_model, decoder_model=seg_model)
    
    else:
        forward_function = partial(forward_undecoded_step, seg_model=seg_model, decoder_model=None)

    return seg_model, forward_function


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
        transform = utils.get_histo_transform(normalize=False)
        
        # Find the size of the compressed patches in the checkpoint file
        in_patch_size = state['args']['patch_size']
        decompression_level = state['args']['compression_level']
        
        if args.add_offset:
            input_offset = 1
                    
    elif state['args']['model_type'] in ['UNet', 'UNetNoBridge']:
        transform = utils.get_histo_transform(normalize=True)
        in_patch_size = args.patch_size
        
        # The segmentation output is the same size of the input
        decompression_level = 0
        
        if args.add_offset:
            input_offset = 2 ** state['args']['compression_level']
    
    if args.add_offset:
        # Find the compression level in the checkpoint file
        output_offset = 2 ** state['args']['compression_level']
    
    # Override the checkpoint arguments with the passed when running the segmentation module
    for k in args.__dict__.keys():
        state['args'][k] = args.__dict__[k]
    
    seg_model, forward_function = setup_network(state['args'])
    
    logger.debug(seg_model)
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
                y = forward_function(x)
                
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
    args = utils.get_segment_args()
    
    utils.setup_logger(args)
    
    segment(args)
    
    logging.shutdown()