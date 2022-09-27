import logging
import os
from functools import partial
from itertools import product

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

SEG_VERSION = '0.1.1'


seg_model_types = {"UNetNoBridge": models.UNetNoBridge,
                   "UNet": models.UNet,
                   "DecoderUNet": models.DecoderUNet}


def segment_block(x, forward_fun, transform=None, seg_threshold=None,
                  offset=0):
    if transform is not None:
        x_t = transform(x.squeeze()).unsqueeze(0)
    else:
        x_t = x.clone()

    y = forward_fun(x_t)

    if y.shape[1] > 1:
        if seg_threshold is None:
            y = torch.softmax(y.cpu(), dim=1)
        else:
            y = torch.argmax(y.cpu(), dim=1)
            y = y.numpy().astype(np.int32)

    else:
        y = torch.sigmoid(y.cpu())
        if seg_threshold is not None:
            y = y > seg_threshold
            y = y.numpy().astype(np.int32)

    if offset > 0:
        H, W = y.shape[-2:]
        y = y[:, :, np.newaxis, offset:H - offset, offset:W - offset]

    return y


def forward_step_base(x, seg_model, dec_model=None, scale_input=1.0):
    if dec_model is not None:
        with torch.no_grad():
            x_brg = dec_model(x)
    else:
        x_brg = None

    # The compressed tensor is in the range of [-127.5, 127.5], so it has to be
    # rescaled to [-1, 1]. This is not necessary for uncompressed input.
    y = seg_model(x / scale_input, x_brg)
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

    src_group = zarr.open(input_filename.split(';')[0], mode='r')
    z_arr = src_group[data_group]

    in_channels, in_H, in_W = [z_arr.shape[data_axes.index(s)] for s in "CYX"]
    # Extract the compression level, original channels, and original shape from
    # the compression metadata
    if compressed_input:
        comp_metadata = src_group[data_group.split('/')[0]]
        comp_metadata = comp_metadata.attrs['compression_metadata']
        comp_level = comp_metadata['compression_level']
        H = comp_metadata['height']
        W = comp_metadata['width']
        comp_label = comp_metadata.get('group', 'compressed')
    else:
        comp_level = 0
        H = in_H
        W = in_W
        comp_label = None

    in_patch_size = patch_size // 2 ** comp_level
    if add_offset:
        if compressed_input:
            in_offset = 1
            out_offset = 2 ** comp_level
        else:
            out_offset = 8
            in_offset = 8
    else:
        in_offset = 0
        out_offset = 0

    np_H_prior = utils.compute_num_patches(in_H, in_patch_size, 0,
                                           in_patch_size)
    np_H_prior += (np_H_prior * in_patch_size - in_H) < 0
    pad_y = np_H_prior * in_patch_size - in_H

    np_W_prior = utils.compute_num_patches(in_W, in_patch_size, 0,
                                           in_patch_size)
    np_W_prior += (np_W_prior * in_patch_size - in_W) < 0
    pad_x = np_W_prior * in_patch_size - in_W

    np_H = utils.compute_num_patches(in_H, in_patch_size + 2 * in_offset,
                                     pad_y + 2 * in_offset,
                                     in_patch_size)
    np_W = utils.compute_num_patches(in_W, in_patch_size + 2 * in_offset,
                                     pad_x + 2 * in_offset,
                                     in_patch_size)

    z = da.from_zarr(z_arr)

    padding = [(in_offset, in_offset), (in_offset, in_offset), (0, 0), (0, 0),
               (0, 0)]
    padding = tuple([padding['XYZCT'.index(a)] for a in data_axes])

    padding_match = [(0, pad_x), (0, pad_y), (0, 0), (0, 0), (0, 0)]
    padding_match = tuple([padding_match['XYZCT'.index(a)] for a in data_axes])

    # The first padding adds enough information from the same image to prevent
    # edge artidacts.
    z = da.pad(z, pad_width=padding, mode='reflect')

    # The second padding is added to match the size of the patches that are
    # processed by the model. Commonly, padding larger than the orginal image
    # are not allowed by the `reflect` paddin mode.
    z = da.pad(z, pad_width=padding_match, mode='constant')

    slices = map(lambda ij:
                 [slice(ij[1]*in_patch_size,
                        ij[1]*in_patch_size + in_patch_size + 2 * in_offset, 1),
                  slice(ij[0]*in_patch_size,
                        ij[0]*in_patch_size + in_patch_size + 2 * in_offset, 1),
                  slice(0, 1, 1),
                  slice(0, in_channels, 1),
                  slice(0, 1, 1)],
                 product(range(np_H), range(np_W)))

    slices = [tuple([None]
              + [s['XYZCT'.index(a)] for a in data_axes])
              for s in slices]

    unused_axis = list(set(data_axes) - set('YXC'))
    transpose_order = [0]
    transpose_order += [data_axes.index(a) + 1 for a in unused_axis]
    transpose_order += [data_axes.index(a) + 1 for a in 'YXC']

    y = [da.from_delayed(
            dask.delayed(segment_block)(np.transpose(z[slices[ij]],
                                                     transpose_order),
                                        forward_fun,
                                        transform,
                                        seg_threshold=seg_threshold,
                                        offset=out_offset),
            shape=(1, classes, 1, patch_size, patch_size),
            meta=np.empty((), dtype=np.int32))
         for ij in range(np_W * np_H)]

    y = da.block([[y[i*np_W + j] for j in range(np_W)] for i in range(np_H)])
    y = y[..., :H, :W]

    if comp_label is not None:
        data_group = '/'.join(data_group.split(comp_label)[1].split('/')[1:])
    if len(seg_label):
        component = 'labels/%s/%s' % (seg_label, data_group)
    else:
        component = 'labels/%s' % data_group

    if 'zarr' in destination_format:
        y.to_zarr(output_filename, component=component, overwrite=True,
                  compressor=compressor)

        group = zarr.open(output_filename, mode="rw")
        if len(seg_label):
            seg_group = group['labels/%s' % seg_label]
        else:
            seg_group = group

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
        y = (y / y.max() * 255).astype(np.uint8)
        if classes > 1:
            im = Image.fromarray(y.squeeze().transpose(1, 2, 0).compute())
        else:
            im = Image.fromarray(y.squeeze().compute())
        im.save(output_filename, quality_opts={'compress_level': 9,
                                               'optimize': False})


def setup_network(state_args, pretrained_model=None, autoencoder_model=None,
                  use_gpu=False):
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
    segment_fun : function
        The function to be used as feed-forward step

    compressed_input : bool
        Whether the input requires to be compressed or not according to the
        segmentation model.
    """
    # When the model works on compressed representation, tell the dataloader to
    # obtain the compressed input and normal size target.
    if 'Decoder' in state_args['model_type']:
        compressed_input = True
        scale_input = 127.5
    else:
        compressed_input = False
        scale_input = 1.0

    if ('Decoder' in state_args['model_type']
       and autoencoder_model is not None):
        # If a decoder model is passed as argument, use the decoded step
        # version of the feed-forward step.
        checkpoint_state = torch.load(autoencoder_model,
                                      map_location=None if use_gpu else 'cpu')
        dec_model = models.SynthesizerInflate(rec_level=-1, color=False,
                                              **checkpoint_state['args'])
        dec_model.load_state_dict(checkpoint_state['decoder'], strict=False)

        dec_model = nn.DataParallel(dec_model)
        if use_gpu:
            dec_model.cuda()

        dec_model.eval()
        state_args['use_bridge'] = True
        state_args['autoencoder_channels_net'] = \
            checkpoint_state['args']['channels_net']

    elif ('Decoder' in state_args['model_type']
          and autoencoder_model is None):
        state_args['use_bridge'] = False
        dec_model = models.EmptyBridge(
            compression_level=state_args['compression_level'],
            compressed_input=True)

    elif 'NoBridge' in state_args['model_type']:
        state_args['use_bridge'] = False
        dec_model = models.EmptyBridge(compression_level=3,
                                       compressed_input=False)

    else:
        state_args['use_bridge'] = True
        dec_model = None

    seg_model_class = seg_model_types.get(state_args['model_type'], None)

    if seg_model_class is None:
        raise ValueError('Model type %s'
                         ' not supported' % state_args['model_type'])

    seg_model = seg_model_class(**state_args)
    if pretrained_model is not None:
        seg_model.load_state_dict(pretrained_model)

    seg_model = nn.DataParallel(seg_model)
    if use_gpu:
        seg_model.cuda()

    seg_model.eval()

    # Define what funtion use in the feed-forward step
    forward_fun = partial(forward_step_base, seg_model=seg_model,
                          dec_model=dec_model,
                          scale_input=scale_input)

    return seg_model, forward_fun, compressed_input


def segment(args):
    """Compress any supported file format (zarr, or any supported by PIL)
    into a compressed representation in zarr format.
    """
    logger = logging.getLogger(args.mode + '_log')

    # Open checkpoint from trained model state
    state = utils.load_state(args)

    # Find the size of the compressed patches in the checkpoint file
    classes = state['args']['classes']

    (_,
     forward_fun,
     compressed_input) = setup_network(
        state['args'],
        pretrained_model=state['model'],
        autoencoder_model=args.autoencoder_model,
        use_gpu=args.gpu)

    (transform,
     _,
     _) = utils.get_zarr_transform(normalize=True,
                                   compressed_input=compressed_input)

    if not args.destination_format.startswith('.'):
        args.destination_format = '.' + args.destination_format

    input_fn_list = utils.get_filenames(args.data_dir, args.source_format,
                                        data_mode="all")

    if args.destination_format.lower() not in args.output_dir[0].lower():
        # If the output path is a directory, append the input filenames to it,
        # so the compressed files have the same name as the original file.
        fn_list = map(lambda fn:
                      fn.split('.zarr')[0].replace('\\', '/').split('/')[-1],
                      input_fn_list)
        output_fn_list = [os.path.join(args.output_dir[0],
                                       '%s%s%s' % (fn, args.comp_identifier,
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

        if 'zarr' in args.destination_format:
            logger.info(
                'Image `%s` segmented succesfully. '
                'Output saved at `%s/%s/0`' % (in_fn, out_fn,
                                               args.task_label_identifier))
        else:
            logger.info(
                'Image `%s` segmented succesfully. '
                'Output saved at `%s`' % (in_fn, out_fn))


if __name__ == '__main__':
    args = utils.get_args(task='segmentation', mode='inference')

    utils.setup_logger(args)

    segment(args)
    logging.info('Image segmented successfully')

    logging.shutdown()
