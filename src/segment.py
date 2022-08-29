import logging
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

import zarr
from PIL import Image
from numcodecs import Blosc

import models

import utils

SEG_VERSION = '0.1'


seg_model_types = {"UNetNoBridge": models.UNetNoBridge, "UNet": models.UNet, "DecoderUNet": models.DecoderUNet}


def forward_undecoded_step(x, seg_model=None, dec_model=None):
    y = seg_model(x)
    return y


def forward_decoded_step(x, seg_model=None, dec_model=None):
    # The compressed representation is stored as an unsigned integer between [0, 255].
    # The transformation used in the dataloader transforms it into the range [-127.5, 127.5].
    # However, the synthesis track of the segmentation task works better if the compressed representation is in the range [-1, 1].
    # For this reason the tensor x is divided by 127.5.
    with torch.no_grad():
        x_brg = dec_model.inflate(x, color=False)
    y = seg_model(x / 127.5, x_brg[:0:-1])
    return y


def forward_parallel_decoded_step(x, seg_model=None, dec_model=None):
    with torch.no_grad():
        x_brg = dec_model.module.inflate(x, color=False)
    y = seg_model(x / 127.5, x_brg[:0:-1])
    return y


def setup_network(state, use_gpu=False):
    """ Setup a nerual network for object segmentation.

    Parameters
    ----------
    state : Dictionary
        A checkpoint state saved during the network training
    
    Returns
    -------
    forward_function : function
        The function to be used as feed-forward step
    
    output_channels : int
        The number of classes predicted by this model
    """
    # When the model works on compressed representation, tell the dataloader to obtain the compressed input and normal size target
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
    
    if use_gpu:
        seg_model = nn.DataParallel(seg_model)
        seg_model.cuda()

    output_channels = state['args']['classes']
    
    if 'Decoder' in state['args']['model_type']:
        state['args']['compressed_input'] = True

        if dec_model is None:
            dec_model = seg_model
    else:
        state['args']['compressed_input'] = False

    # Define what funtion use in the feed-forward step
    if seg_model is not None and dec_model is None:
        # Segmentation w/o decoder
        forward_function = partial(forward_undecoded_step, seg_model=seg_model, dec_model=dec_model)
    
    elif seg_model is not None and dec_model is not None:
        # Segmentation w/ decoder
        if use_gpu:
            forward_function = partial(forward_parallel_decoded_step, seg_model=seg_model, dec_model=dec_model)
        else:
            forward_function = partial(forward_decoded_step, seg_model=seg_model, dec_model=dec_model)

    return forward_function, output_channels


def segment_image(forward_function, input_filename, output_filename, classes,
                  seg_threshold=0.5,
                  input_comp_level=3,
                  input_patch_size=64,
                  output_patch_size=512,
                  input_offset=0,
                  output_offset=0,
                  stitch_batches=False,
                  transform=None,
                  source_format='zarr',
                  destination_format='zarr',
                  workers=0,
                  batch_size=1,
                  data_mode='train',
                  data_axes='TCZYX',
                  data_group='0/0'):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    zarr_ds = utils.ZarrDataset(root=input_filename,
                                patch_size=input_patch_size,
                                dataset_size=-1,
                                data_mode=data_mode,
                                offset=input_offset,
                                transform=transform,
                                source_format=source_format,
                                workers=0,
                                data_axes=data_axes,
                                data_group=data_group)

    data_queue = DataLoader(zarr_ds,
                            batch_size=batch_size,
                            num_workers=workers,
                            shuffle=False,
                            pin_memory=True,
                            worker_init_fn=utils.zarrdataset_worker_init)

    H_comp, W_comp = zarr_ds.get_shape()

    if 'zarr' in source_format:
        z_org = zarr.open(input_filename.split(';')[0], mode='r')
    else:
        z_org = None

    label_key = -1
    H = H_comp * 2**input_comp_level
    W = W_comp * 2**input_comp_level
    if z_org is not None:
        if 'compressed' in z_org.keys():
            comp_metadata = z_org['compressed'].attrs['compression_metadata']
            H = comp_metadata['height']
            W = comp_metadata['width']
        if z_org.get('labels', None) is not None:
            label_key = len(z_org['labels']) - 1

    if 'zarr' in destination_format.lower():
        # Output dir is actually the absolute path to the file where to store the compressed representation
        if 'memory' in destination_format:
            group = zarr.group()
        else:
            if os.path.isdir(output_filename):
                group = zarr.group(output_filename, mode='rw')
            else:
                group = zarr.group(output_filename, mode='w')

        if 'labels' in group.keys():
            labs_group = group.open_group('labels')
        else:
            labs_group = group.create_group('labels')

    # Create a new label group avoiding overwriting existing labels
    seg_group = labs_group.create_group(str(label_key + 1))

    # Integrate metadata to the predicted labels generated from segmentation
    seg_group.attrs['image-label'] = {
        "version": "0.5-dev",
        "colors": [
            {"label-value": 0, "rgba": [0, 0, 0, 0]},
            {"label-value": 1, "rgba": [255, 255, 255, 127]}
            ],
        "properties": [
            {"label-value": 0, "class": "background"},
            {"label-value": 1, "class": "Glomerulus"}
        ],
        "source": "../../0"
    }

    seg_group.attrs['multiscales'] = {
        "version": "0.5-dev",
        "name": "glomeruli_segmentation",
        "datasets": [
                {"path": "0"}
        ]
    }

    seg_group.attrs['segmentation_metadata'] = dict(
        height=H,
        width=W,
        classes=classes,
        axes='TCZYX',
        offset=output_offset,
        stitch_batches=stitch_batches,
        model=str(forward_function),
        original=zarr_ds._data_group,
        segmentation_threshold=seg_threshold,
        version=SEG_VERSION
    )

    if stitch_batches:
        z_seg = seg_group.create_dataset('0',
                                         shape=(1, classes, 1, H, W),
                                         chunks=(1,
                                                 classes,
                                                 1,
                                                 output_patch_size,
                                                 output_patch_size),
                                         dtype='u1',
                                         compressor=compressor)
    else:
        z_seg = seg_group.create_dataset('0',
                                         shape=(len(zarr_ds),
                                                classes,
                                                1,
                                                output_patch_size,
                                                output_patch_size),
                                         chunks=(1,
                                                 classes,
                                                 1,
                                                 output_patch_size,
                                                 output_patch_size),
                                         dtype='u1',
                                         compressor=compressor)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof, torch.no_grad():
        with record_function("model_inference"):
            with torch.no_grad():
                for i, (x, _) in enumerate(data_queue):
                    y = forward_function(x)

                    y = torch.sigmoid(y.detach())

                    if output_offset > 0:
                        y = y[..., output_offset:-output_offset, output_offset:-output_offset]

                    y = y.cpu().numpy() > seg_threshold
                    y = y.astype(np.uint8)

                    if stitch_batches:
                        for k, y_k in enumerate(y):
                            _, tl_y, tl_x = utils.compute_grid(i*batch_size + k, imgs_shapes=[(H, W)], imgs_sizes=[0, len(zarr_ds)], patch_size=output_patch_size)
                            tl_y *= output_patch_size
                            tl_x *= output_patch_size
                            z_seg[0, ..., 0, tl_y:tl_y + output_patch_size, tl_x:tl_x + output_patch_size] = y_k
                    else:
                        z_seg[i*batch_size:i*batch_size+x.size(0), ..., 0, tl_y:tl_y + output_patch_size, tl_x:tl_x + output_patch_size] = y

    # If the output format is not zarr, and it is supported by PIL, an image is generated from the segmented image.
    # It should be used with care since this can generate a large image file.
    if 'zarr' not in destination_format:
        im = Image.fromarray(z_seg[0, 0])
        im.save(output_filename, destination_format)

    if z_org is not None \
       and 'labels' in z_org.keys() \
       and ('memory' in destination_format or z_org.store.path != group.store.path):
        zarr.copy(z_org['labels/%i' % label_key], group, 'labels/%i' % label_key)

    if 'memory' in destination_format.lower():
        return group

    return True


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

    forward_function, output_channels = setup_network(state, args.gpu)

    if state['args']['compressed_input']:
        input_comp_level = compression_level
        input_offset = 1
    else:
        input_comp_level = 0

    input_patch_size = args.patch_size // 2 ** input_comp_level

    # Conver the single zarr file into a dataset to be iterated
    transform, _, _ = utils.get_zarr_transform(normalize=True, compressed_input=state['args']['compressed_input'])

    if not args.source_format.startswith('.'):
        args.source_format = '.' + args.source_format

    if not args.destination_format.startswith('.'):
        args.destination_format = '.' + args.destination_format

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

    # Segment each file separately
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        seg_group = segment_image(forward_function=forward_function,
                                  input_filename=in_fn,
                                  output_filename=out_fn,
                                  classes=output_channels,
                                  seg_threshold=args.seg_threshold,
                                  input_comp_level=input_comp_level,
                                  input_patch_size=input_patch_size,
                                  output_patch_size=args.patch_size,
                                  input_offset=input_offset,
                                  output_offset=output_offset,
                                  stitch_batches=args.stitch_batches,
                                  transform=transform,
                                  source_format=args.source_format,
                                  destination_format=args.destination_format,
                                  workers=args.workers,
                                  batch_size=args.batch_size,
                                  data_mode=args.data_mode,
                                  data_axes=args.data_axes,
                                  data_group=args.data_group)

        yield seg_group


if __name__ == '__main__':
    args = utils.get_args(task='segmentation', mode='inference')

    utils.setup_logger(args)

    for _ in segment(args):
        logging.info('Image segmented successfully')

    logging.shutdown()
