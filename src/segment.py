import logging
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import zarr
from PIL import Image
from numcodecs import Blosc

import models

import utils

seg_model_types = {"UNetNoBridge": models.UNetNoBridge, "UNet": models.UNet, "DecoderUNet": models.DecoderUNet}



def forward_undecoded_step(x, seg_model, decoder_model=None):
    y = seg_model(x)
    return y


def forward_decoded_step(x, seg_model, decoder_model=None):
    with torch.no_grad():
        x_rec, x_brg = decoder_model.inflate(x, color=True)
    y = seg_model(x / 127.5, x_brg[:0:-1])
    
    return y


def forward_parallel_decoded_step(x, seg_model, decoder_model=None):
    with torch.no_grad():
        x_brg = decoder_model.module.inflate(x, color=False)
    y = seg_model(x / 127.5, x_brg[:0:-1])
    return y


def segment_image(forward_function, filename, output_dir, classes, input_comp_level, input_patch_size, output_patch_size, input_offset, output_offset, transform, source_format, destination_format, workers, is_labeled=False):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    histo_ds = utils.ZarrDataset(root=filename, patch_size=input_patch_size, offset=input_offset, transform=transform, source_format=source_format)
    data_queue = DataLoader(histo_ds, batch_size=1, num_workers=workers, shuffle=False, pin_memory=True)

    H_comp, W_comp = histo_ds.get_shape()
    H = H_comp * 2**input_comp_level
    W = W_comp * 2**input_comp_level

    if 'zarr' in destination_format:
        # Output dir is actually the absolute path to the file where to store the compressed representation
        if 'memory' in destination_format:
            group = zarr.group()
        else:
            group = zarr.group(output_dir, overwrite=True)

        comp_group = group.create_group('0', overwrite=True)
    
        z_seg = comp_group.create_dataset('0', shape=(1, classes, H, W), chunks=(1, classes, output_patch_size, output_patch_size), dtype=np.float32, compressor=compressor)

    else:
        z_seg = zarr.zeros(shape=(1, classes, H, W), chunks=(1, classes, output_patch_size, output_patch_size), dtype='u1', compressor=compressor)

    with torch.no_grad():
        for i, (x, _) in enumerate(data_queue):
            y = forward_function(x)

            y = torch.sigmoid(y.detach())
            y = y.cpu().numpy()

            if output_offset > 0:
                y = y[..., output_offset:-output_offset, output_offset:-output_offset]
                
            _, tl_y, tl_x = utils.compute_grid(i, 1, H, W, output_patch_size)
            tl_y *= output_patch_size
            tl_x *= output_patch_size
            
            z_seg[..., tl_y:(tl_y+output_patch_size), tl_x:(tl_x+output_patch_size)] = y if 'zarr' in destination_format else (y * 255).astype(np.uint8)

    # If the output format is not zarr, and it is supported by PIL, an image is generated from the segmented image.
    # It should be used with care since this can generate a large image file.
    if 'zarr' not in destination_format:
        im = Image.fromarray(z_seg[0, 0])
        im.save(output_dir, destination_format)
    elif is_labeled:
        label_group = group.create_group('1', overwrite=True)
        z_org = zarr.open(filename, 'r')
        zarr.copy(z_org['1/0'], label_group)

    if 'memory' in destination_format:
        return group
    
    return True


def setup_network(state):
    """ Setup a nerual network for object segmentation.

    Parameters
    ----------
    state : Dictionary
        A checkpoint state saved during the network training
    
    Returns
    -------
    seg_model : nn.Module
        The segmentation mode implemented by a convolutional neural network
    
    forward_function : function
        The function to be used as feed-forward step
    """
    # When the model works on compressed representation, tell the dataloader to obtain the compressed input and normal size target
    state['args']['compressed_input'] = 'Decoder' in state['args']['model_type']

    # If a decoder model is passed as argument, use the decoded step version of the feed-forward step    
    if state['args']['autoencoder_model'] is None:
        if 'Decoder' in state['args']['model_type'] or 'NoBridge' in state['args']['model_type']:
            state['args']['use_bridge'] = False
        else:
            state['args']['use_bridge'] = True
    else:
        if not state['args']['gpu']:
            checkpoint_state = torch.load(state['args']['autoencoder_model'], map_location=torch.device('cpu'))
        else:
            checkpoint_state = torch.load(state['args']['autoencoder_model'])
       
        decoder_model = models.Synthesizer(**checkpoint_state['args'])
        decoder_model.load_state_dict(checkpoint_state['decoder'])

        if state['args']['gpu']:
            decoder_model = nn.DataParallel(decoder_model)        
            decoder_model.cuda()

        decoder_model.eval()
        state['args']['use_bridge'] = True
        print(decoder_model)
    
    seg_model_class = seg_model_types.get(state['args']['model_type'], None)
    if seg_model_class is None:
        raise ValueError('Model type %s not supported' % state['args']['model_type'])
    
    seg_model = seg_model_class(**state['args'])       
    seg_model.load_state_dict(state['model'])

    # If there are more than one GPU, DataParallel handles automatically the distribution of the work
    if state['args']['gpu']:
        seg_model = nn.DataParallel(seg_model)
        seg_model.cuda()

    # Define what funtion use in the feed-forward step
    if state['args']['autoencoder_model'] is None:
        if 'Decoder' in state['args']['model_type']:
            # If no decoder is loaded, use the inflate function inside the segmentation model
            if state['args']['gpu']:
                forward_function = partial(forward_parallel_decoded_step, seg_model=seg_model, decoder_model=seg_model)
            else:
                forward_function = partial(forward_decoded_step, seg_model=seg_model, decoder_model=seg_model)   
        else:
            forward_function = partial(forward_undecoded_step, seg_model=seg_model, decoder_model=None)
    else:
        if state['args']['gpu']:
            forward_function = partial(forward_parallel_decoded_step, seg_model=seg_model, decoder_model=decoder_model)
        else:
            forward_function = partial(forward_decoded_step, seg_model=seg_model, decoder_model=decoder_model)

    return seg_model, forward_function


def segment(args):
    """ Compress any supported file format (zarr, or any supported by PIL) into a compressed representation in zarr format.
    """    
    logger = logging.getLogger(args.mode + '_log')

    # Open checkpoint from trained model state
    state = utils.load_state(args)

    # Find the size of the compressed patches in the checkpoint file
    compression_level = state['args']['compression_level']
                
    output_offset = (2 ** compression_level) if args.add_offset else 0
    input_offset = (2 ** compression_level) if args.add_offset else 0

    for k in args.__dict__.keys():
        state['args'][k] = args.__dict__[k]

    seg_model, forward_function = setup_network(state)

    if state['args']['compressed_input']:
        input_comp_level = compression_level
        input_offset = 1
    else:
        input_comp_level = 0
    
    input_patch_size = args.patch_size // 2 ** input_comp_level

    logger.debug('Model')
    logger.debug(seg_model)
    
    # Conver the single zarr file into a dataset to be iterated
    transform = utils.get_zarr_transform(normalize=True, compressed_input=state['args']['compressed_input'])

    if not args.input[0].lower().endswith(args.source_format.lower()):
        # If a directory has been passed, get all image files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: fn.lower().endswith(args.source_format.lower()), os.listdir(args.input[0]))))
    else:
        input_fn_list = args.input
        
    output_fn_list = list(map(lambda fn: os.path.join(args.output_dir, fn + '_seg.%s' % args.destination_format), map(lambda fn: os.path.splitext(os.path.basename(fn))[0], input_fn_list)))

    # Segment each file by separate    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        seg_group = segment_image(forward_function=forward_function, filename=in_fn, output_dir=out_fn, classes=state['args']['classes'], input_comp_level=input_comp_level, input_patch_size=input_patch_size, output_patch_size=args.patch_size, input_offset=input_offset, output_offset=output_offset, transform=transform, source_format=args.source_format, destination_format=args.destination_format, workers=args.workers, is_labeled=args.is_labeled)
        yield seg_group


if __name__ == '__main__':
    args = utils.get_segment_args()
    
    utils.setup_logger(args)
    
    for _ in segment(args):
        logging.info('Image segmented successfully')
    
    logging.shutdown()