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
cae_model_types = {"MaskedAutoEncoder": models.Synthesizer, "AutoEncoder": models.Synthesizer}


def forward_undecoded_step(x, seg_model=None, dec_model=None):
    y, fx = seg_model.extract_features(x)
    return y, fx


def forward_parallel_undecoded_step(x, seg_model=None, dec_model=None):
    y, fx = seg_model.module.extract_features(x)
    return y, fx


def forward_decoded_step(x, seg_model=None, dec_model=None):
    # The compressed representation is stored as an unsigned integer between [0, 255].
    # The transformation used in the dataloader transforms it into the range [-127.5, 127.5].
    # However, the synthesis track of the segmentation task works better if the compressed representation is in the range [-1, 1].
    # For this reason the tensor x is divided by 127.5.
    with torch.no_grad():
        x_brg = dec_model.inflate(x, color=False)
    y, fx = seg_model.extract_features(x / 127.5, x_brg[:0:-1])
    return y, fx


def forward_parallel_decoded_step(x, seg_model=None, dec_model=None):
    with torch.no_grad():
        x_brg = dec_model.module.inflate(x, color=False)
    y, fx = seg_model.extract_features(x / 127.5, x_brg[:0:-1])
    return y, fx


# These two functions are for the reconstruction step of the decompression/synthesis model for image reconstruction
def forward_reconstruct_step(x, seg_model=None, dec_model=None):
    with torch.no_grad():
        y, x_brg = dec_model.inflate(x, color=True)
    fx = x_brg[-1]
    return y, fx


def forward_parallel_reconstruct_step(x, seg_model=None, dec_model=None):
    with torch.no_grad():
        y, x_brg = dec_model.module.inflate(x, color=True)
    fx = x_brg[-1]
    return y, fx


def extract_image(forward_function, filename, output_dir, features, output_channels, input_comp_level, input_patch_size, output_patch_size, input_offset, output_offset, transform, source_format, destination_format, workers):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    histo_ds = utils.ZarrDataset(root=filename, patch_size=input_patch_size, offset=input_offset, transform=transform, source_format=source_format)
    data_queue = DataLoader(histo_ds, batch_size=1, num_workers=workers, shuffle=False, pin_memory=True)

    H_comp, W_comp = histo_ds.get_shape()
    H = H_comp * 2**input_comp_level
    W = W_comp * 2**input_comp_level

    if destination_format == 'zarr':
        # Output dir is actually the absolute path to the file where to store the compressed representation
        group = zarr.group(output_dir, overwrite=True)
        # The group 0 is to store the prediction, and group 1 to store the features map
        comp_group_pred = group.create_group('0', overwrite=True)
        comp_group_feat = group.create_group('1', overwrite=True)
    
        # The sub-group 0 is meant to store the image at maximum resolution, if different resolutions are generated in the future, use the next sub-groups under the corresponding main group
        z_pred = comp_group_pred.create_dataset('0', shape=(1, output_channels, H, W), chunks=(1, output_channels, output_patch_size, output_patch_size), dtype=np.float32, compressor=compressor)
        z_feat = comp_group_feat.create_dataset('0', shape=(1, features, H, W), chunks=(1, features, output_patch_size, output_patch_size), dtype=np.float32, compressor=compressor)

    with torch.no_grad():
        for i, (x, _) in enumerate(data_queue):
            y, fx = forward_function(x)

            y = y.cpu().numpy()
            fx = fx.cpu().numpy()

            if output_offset > 0:
                y = y[..., output_offset:-output_offset, output_offset:-output_offset]
                fx = fx[..., output_offset:-output_offset, output_offset:-output_offset]
                
            _, tl_y, tl_x = utils.compute_grid(i, 1, H, W, output_patch_size)
            tl_y *= output_patch_size
            tl_x *= output_patch_size
            
            z_pred[..., tl_y:(tl_y+output_patch_size), tl_x:(tl_x+output_patch_size)] = y
            z_feat[..., tl_y:(tl_y+output_patch_size), tl_x:(tl_x+output_patch_size)] = fx


def setup_network(state):
    """ Setup a nerual network for object segmentation/autoencoder.

    Parameters
    ----------
    state : Dictionary
        A checkpoint state saved during the network training
    
    Returns
    -------    
    forward_function : function
        The function to be used as feed-forward step
    
    output_channels : int
        The number of channels in the output image.
        For segmentation models, it is the number of classes, whereas for autoencoder models, it is the number of color channels.

    """
    # When the model works on compressed representation, tell the dataloader to obtain the compressed input and normal size target
    if state['args']['task'] == 'segmentation':
        if ('Decoder' in state['args']['model_type'] and state['args']['autoencoder_model'] is None) or 'NoBridge' in state['args']['model_type']:
            state['args']['use_bridge'] = False
        else:
            state['args']['use_bridge'] = True
         
        if state['args']['autoencoder_model'] is not None:
            # If a decoder model is passed as argument, use the decoded step version of the feed-forward step
            if not state['args']['gpu']:
                checkpoint_state = torch.load(state['args']['autoencoder_model'], map_location=torch.device('cpu'))
            else:
                checkpoint_state = torch.load(state['args']['autoencoder_model'])
        
            dec_model = models.Synthesizer(**checkpoint_state['args'])
            dec_model.load_state_dict(checkpoint_state['decoder'])

            if state['args']['gpu']:
                dec_model = nn.DataParallel(dec_model)        
                dec_model.cuda()

            dec_model.eval()
            state['args']['use_bridge'] = True
        else:
            dec_model = None
            
        seg_model_class = seg_model_types.get(state['args']['model_type'], None)
        if seg_model_class is None:
            raise ValueError('Model type %s not supported' % state['args']['model_type'])

        seg_model = seg_model_class(**state['args'])
        seg_model.load_state_dict(state['model'])
        
        if state['args']['gpu']:
            seg_model = nn.DataParallel(seg_model)
            seg_model.cuda()

        output_channels = state['args']['classes']

    elif state['args']['task'] == 'autoencoder':
        seg_model = None

        cae_model_class = cae_model_types.get(state['args']['model_type'], None)
        if cae_model_class is None:
            raise ValueError('Model type %s not supported' % state['args']['model_type'])

        dec_model = cae_model_class(**state['args'])
        dec_model.load_state_dict(state['decoder'])
        
        if state['args']['gpu']:
            dec_model = nn.DataParallel(dec_model)
            dec_model.cuda()

        output_channels = state['args']['channels_org']

    if 'Decoder' in state['args']['model_type']:
        state['args']['compressed_input'] = True

        if dec_model is None:
            dec_model = seg_model
        
    elif  'AutoEncoder' in state['args']['model_type']:
        state['args']['compressed_input'] = True

    else:
        state['args']['compressed_input'] = False

    # Define what funtion use in the feed-forward step
    if seg_model is not None and dec_model is None:
        # Segmentation w/o decoder
        if state['args']['gpu']:
            forward_function = partial(forward_parallel_undecoded_step, seg_model=seg_model, dec_model=dec_model)
        else:
            forward_function = partial(forward_undecoded_step, seg_model=seg_model, dec_model=dec_model)
    
    elif seg_model is not None and dec_model is not None:
        # Segmentation w/ decoder
        if state['args']['gpu']:
            forward_function = partial(forward_parallel_decoded_step, seg_model=seg_model, dec_model=dec_model)
        else:
            forward_function = partial(forward_decoded_step, seg_model=seg_model, dec_model=dec_model)

    elif seg_model is None and dec_model is not None:
        # Decoder
        if state['args']['gpu']:
            forward_function = partial(forward_parallel_reconstruct_step, seg_model=seg_model, dec_model=dec_model)
        else:
            forward_function = partial(forward_reconstruct_step, seg_model=seg_model, dec_model=dec_model)
    
    return forward_function, output_channels


def extract(args):
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

    forward_function, output_channels = setup_network(state)

    if state['args']['compressed_input']:
        input_comp_level = compression_level
        input_offset = 1
    else:
        input_comp_level = 0
    
    input_patch_size = args.patch_size // 2 ** input_comp_level
    
    # Convert the single zarr file into a dataset to be iterated
    transform = utils.get_zarr_transform(normalize=True, compressed_input=state['args']['compressed_input'])

    if not args.input[0].lower().endswith(args.source_format.lower()):
        # If a directory has been passed, get all image files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: fn.lower().endswith(args.source_format.lower()), os.listdir(args.input[0]))))
    else:
        input_fn_list = args.input
        
    output_fn_list = list(map(lambda fn: os.path.join(args.output_dir, fn + '_feats.%s' % args.destination_format), map(lambda fn: os.path.splitext(os.path.basename(fn))[0], input_fn_list)))

    # Segment each file by separate    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        extract_image(forward_function=forward_function, filename=in_fn, output_dir=out_fn, features=state['args']['channels_net'], output_channels=output_channels, input_comp_level=input_comp_level, input_patch_size=input_patch_size, output_patch_size=args.patch_size, input_offset=input_offset, output_offset=output_offset, transform=transform, source_format=args.source_format, destination_format=args.destination_format, workers=args.workers)


if __name__ == '__main__':
    args = utils.get_segment_args()
    
    utils.setup_logger(args)
    
    extract(args)
    
    logging.shutdown()