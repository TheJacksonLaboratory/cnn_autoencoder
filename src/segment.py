import logging
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn

import zarr
import dask
import dask.array as da

from PIL import Image
from numcodecs import Blosc

import models

import utils

SEG_VERSION = '0.1'


seg_model_types = {"UNetNoBridge": models.UNetNoBridge,
                   "UNet": models.UNet,
                   "DecoderUNet": models.DecoderUNet}


@dask.delayed
def threshold_seg_binary(x, forward_fun, transform, seg_threshold=0.5,
                         offset=0):
    h, w = x.shape[-2:]
    x_t = transform(np.moveaxis(x, 0, -1)).view(1, -1, h, w)
    y = forward_fun(x_t)

    y = torch.sigmoid(y).cpu().numpy()
    y = y > seg_threshold

    y = y.astype(np.int32)

    H, W = y.shape[-2:]
    y = y[..., offset:H - offset, offset:W - offset]
    return y


@dask.delayed
def threshold_seg_multiclass(x, forward_fun, transform, seg_threshold=None, offset=0):
    h, w = x.shape[-2:]
    x = transform(x).view(1, -1, h, w)
    y = forward_fun(x)

    H, W = y.shape[-2:]
    y = y[..., offset:H - offset, offset:W - offset]

    y = torch.argmax(y, 1).cpu().numpy().astype(np.int32)
    return x


def forward_undecoded_step(x, seg_model=None, dec_model=None):
    with torch.no_grad():
        y = seg_model(x)
    return y


def forward_decoded_step(x, seg_model=None, dec_model=None):
    with torch.no_grad():
        x_brg = dec_model.module.inflate(x, color=False)
        y = seg_model(x / 127.5, x_brg[:0:-1])
    return y


def segment_image(forward_fun, input_filename, output_filename, classes,
                  seg_threshold=0.5,
                  compressed_input=False,
                  patch_size=512,
                  add_offset=False,
                  transform=None,
                  destination_format='zarr',
                  data_group='0/0',
                  data_axes='TCZYX',
                  seed=None,
                  seg_label='segmentation'):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    z_src = zarr.open(input_filename.split(';')[0], mode='r')

    # When the input is a compressed representation, load the information from
    # the contained metadata.
    if compressed_input and 'compressed' in z_src.keys():
        comp_metadata = z_src['compressed'].attrs['compression_metadata']
        H = comp_metadata['height']
        W = comp_metadata['width']
        in_H = comp_metadata['compressed_height']
        in_W = comp_metadata['compressed_width']
        compression_level = comp_metadata['compression_level']
        in_channels = comp_metadata['compressed_channels']
    else:
        shape = z_src[data_group].shape
        in_channels, H, W = [shape[data_axes.index(s)] for s in 'CYX']
        in_H = H
        in_W = W
        compression_level = 0

    # Open the zarr file lazily using Dask
    in_offset = 0
    out_offset = 0
    in_patch_size = patch_size // 2 ** compression_level
    if add_offset:
        in_offset = 1 if compressed_input else 8
        out_offset = 2 ** compression_level
        padding = [(in_offset, in_offset), (in_offset, in_offset),
                   (0, 0),
                   (0, 0),
                   (0, 0)]
        padding = tuple([padding['XYZCT'.index(a)] for a in data_axes])

    chunks = (in_patch_size,
              in_patch_size,
              1,
              classes,
              1)
    chunks = tuple([chunks['XYZCT'.index(a)] for a in data_axes])

    np_H = utils.compute_num_patches(in_H, in_patch_size + 2 * in_offset,
                                     2 * in_offset,
                                     in_patch_size)
    np_W = utils.compute_num_patches(in_W, in_patch_size + 2 * in_offset,
                                     2 * in_offset,
                                     in_patch_size)

    z = da.from_zarr(input_filename, component=data_group, chunks=chunks)
    z = dask.delayed(utils.unfold_input)(z, in_channels,
                                         in_patch_size + in_offset * 2,
                                         in_patch_size,
                                         padding)
    z = da.from_delayed(z, shape=(np_H*np_W, in_channels,
                                  in_patch_size + 2 * in_offset,
                                  in_patch_size + 2 * in_offset),
                        dtype=np.uint8)

    # Map the network inference with overlapping edges to prevent artifacts.
    if classes > 1:
        threshold_seg = threshold_seg_multiclass
    else:
        threshold_seg = threshold_seg_binary

    y = da.concatenate(
            [da.from_delayed(threshold_seg(z_patch, forward_fun, transform,
                                           seg_threshold,
                                           out_offset),
                             shape=(1, classes, patch_size, patch_size),
                             dtype=np.int32)
             for z_patch in z])

    y = da.from_delayed(dask.delayed(utils.fold_input)(y, 1, patch_size, np_H,
                                                       np_W,
                                                       H,
                                                       W),
                        shape=(1, classes, 1, H, W),
                        dtype=np.int32)

    # If the output format is not zarr, and it is supported by PIL, an image is
    # generated from the segmented image.
    # It should be used with care since this can generate a large image file.
    if 'zarr' in destination_format:
        y.to_zarr(output_filename, component='labels/%s/0/0' % seg_label,
                  overwrite=True,
                  compressor=compressor)
        seg_group = zarr.open(output_filename)['labels/%s' % seg_label]
        # Integrate metadata to the predicted labels generated from
        # the segmentation.
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
            "source": data_group
        }

        seg_group.attrs['multiscales'] = {
            "version": "0.5-dev",
            "name": "glomeruli_segmentation",
            "datasets": [
                {"path": "0/0"}
            ]
        }

        seg_group.attrs['segmentation_metadata'] = dict(
            height=H,
            width=W,
            classes=classes,
            axes='TCZYX',
            offset=add_offset,
            model=str(forward_fun),
            model_seed=seed,
            original=input_filename,
            segmentation_threshold=seg_threshold,
            version=SEG_VERSION
        )
    else:
        # Note that the image should have a number of classes that can be
        # interpreted as a GRAYSCALE image, RGB image or RBGA image.
        im = Image.fromarray(z.reshape(classes, H, W).compute())
        im.save(output_filename, destination_format)


def setup_network(state, autoencoder_model=None, use_gpu=False):
    """Setup a neural network for object segmentation.

    Parameters
    ----------
    state : Dictionary
        A checkpoint state saved during the network training
    autoencoder_model : path or None
        Path to the checkpoint of the autoencoder model used for training the
        segmentation model.
    use_gpu : bool
        Whether GPU is available and the user is willing to use it.

    Returns
    -------
    forward_fun : function
        The function to be used as feed-forward step

    compressed_input : bool
        Whether the input requires to be compressed or not according to the
        segmentation model.
    """
    # When the model works on compressed representation, tell the dataloader to
    # obtain the compressed input and normal size target.
    if ('Decoder' in state['args']['model_type']
       and autoencoder_model is None
       or 'NoBridge' in state['args']['model_type']):
        state['args']['use_bridge'] = False
    else:
        state['args']['use_bridge'] = True

    if autoencoder_model is not None:
        # If a decoder model is passed as argument, use the decoded step
        # version of the feed-forward step.
        checkpoint_state = torch.load(autoencoder_model,
                                      map_location=None if use_gpu else 'cpu')
        dec_model = models.Synthesizer(**checkpoint_state['args'])
        dec_model.load_state_dict(checkpoint_state['decoder'])

        dec_model = nn.DataParallel(dec_model)
        if state['args']['gpu']:
            dec_model.cuda()

        dec_model.eval()
        state['args']['use_bridge'] = True
    else:
        dec_model = None

    seg_model_class = seg_model_types.get(state['args']['model_type'], None)

    if seg_model_class is None:
        raise ValueError('Model type %s'
                         ' not supported' % state['args']['model_type'])

    seg_model = seg_model_class(**state['args'])
    seg_model.load_state_dict(state['model'])

    seg_model = nn.DataParallel(seg_model)
    if use_gpu:
        seg_model.cuda()

    if 'Decoder' in state['args']['model_type']:
        compressed_input = True

        if dec_model is None:
            dec_model = seg_model
    else:
        compressed_input = False

    # Define what funtion use in the feed-forward step
    if dec_model is None:
        # Segmentation w/o decoder
        forward_fun = partial(forward_undecoded_step,
                              seg_model=seg_model,
                              dec_model=dec_model)
    else:
        # Segmentation w/ decoder
        forward_fun = partial(forward_decoded_step,
                              seg_model=seg_model,
                              dec_model=dec_model)

    return forward_fun, compressed_input


def segment(args):
    """Compress any supported file format (zarr, or any supported by PIL)
    into a compressed representation in zarr format.
    """
    logger = logging.getLogger(args.mode + '_log')

    # Open checkpoint from trained model state
    state = utils.load_state(args)

    # Find the size of the compressed patches in the checkpoint file
    classes = state['args']['classes']

    forward_fun, compressed_input = setup_network(state,
                                                  args.autoencoder_model,
                                                  args.gpu)

    (transform,
     _,
     _) = utils.get_zarr_transform(normalize=True,
                                   compressed_input=compressed_input)

    if not args.destination_format.startswith('.'):
        args.destination_format = '.' + args.destination_format

    input_fn_list = utils.get_filenames(args.data_dir, args.source_format,
                                        args.data_mode)

    if args.destination_format.lower() not in args.output_dir[0].lower():
        # If the output path is a directory, append the input filenames to it,
        # so the compressed files have the same name as the original file.
        fn_list = map(lambda fn:
                      fn.split('.zarr')[0].replace('\\', '/').split('/')[-1],
                      input_fn_list)
        output_fn_list = [os.path.join(args.output_dir[0],
                                       '%s%s.%s' % (fn, args.comp_identifier,
                                                    args.destination_format))
                          for fn in fn_list]

    else:
        output_fn_list = args.output_dir

    # Segment each file separately
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        segment_image(forward_fun=forward_fun, input_filename=in_fn,
                      output_filename=out_fn,
                      classes=classes,
                      seg_threshold=args.seg_threshold,
                      patch_size=args.patch_size,
                      compressed_input=compressed_input,
                      add_offset=args.add_offset,
                      transform=transform,
                      destination_format=args.destination_format,
                      data_group=args.data_group,
                      data_axes=args.data_axes,
                      seed=state['args']['seed'],
                      seg_label=args.task_label_identifier)
        logger.info('Image `%s` segmented succesfully. '
                    'Output saved at `%s/%s/0`' % (in_fn, out_fn,
                                                   args.task_label_identifier))


if __name__ == '__main__':
    args = utils.get_args(task='segmentation', mode='inference')

    utils.setup_logger(args)

    segment(args)
    logging.info('Image segmented successfully')

    logging.shutdown()
