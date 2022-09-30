import logging
import os

from itertools import product
import math
import numpy as np
import torch
import torch.nn as nn

import zarr
import dask
import dask.array as da

from numcodecs import Blosc

import models

import utils


COMP_VERSION = '0.1.2'


@dask.delayed
def encode(x, comp_model, transform, offset=0):
    x_t = transform(x.squeeze()).unsqueeze(0)

    with torch.set_grad_enabled(comp_model.training):
        y_q, _ = comp_model(x_t)

    y_q = y_q.detach().cpu() + 127.5
    y_q = y_q.round().to(torch.uint8)
    y_q = y_q.unsqueeze(2).numpy()

    h, w = y_q.shape[-2:]
    y_q = y_q[..., offset:h-offset, offset:w-offset]

    return y_q


def compress_image(comp_model, input_filename, output_filename, channels_bn,
                   compression_level,
                   patch_size=512,
                   add_offset=False,
                   transform=None,
                   source_format='zarr',
                   data_group='0/0',
                   data_axes='TCZYX',
                   seed=None,
                   comp_label='compressed'):

    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    z_arr, _, _ = utils.image_to_zarr(input_filename.split(';')[0], patch_size,
                                      source_format,
                                      data_group)

    in_channels, in_H, in_W = [z_arr.shape[data_axes.index(s)] for s in "CYX"]

    in_offset = 0
    out_offset = 0
    out_patch_size = patch_size // 2 ** compression_level
    if add_offset:
        in_offset = 2 ** compression_level
        out_offset = 1

    np_H_prior = utils.compute_num_patches(in_H, patch_size, 0, patch_size)
    np_H_prior += (np_H_prior * patch_size - in_H) < 0
    pad_y = np_H_prior * patch_size - in_H

    np_W_prior = utils.compute_num_patches(in_W, patch_size, 0, patch_size)
    np_W_prior += (np_W_prior * patch_size - in_W) < 0
    pad_x = np_W_prior * patch_size - in_W

    np_H = utils.compute_num_patches(in_H, patch_size + 2 * in_offset,
                                     pad_y + 2 * in_offset,
                                     patch_size)
    np_W = utils.compute_num_patches(in_W, patch_size + 2 * in_offset,
                                     pad_x + 2 * in_offset,
                                     patch_size)

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
                 [slice(ij[1]*patch_size,
                        ij[1]*patch_size + patch_size + 2 * in_offset, 1),
                  slice(ij[0]*patch_size,
                        ij[0]*patch_size + patch_size + 2 * in_offset, 1),
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

    y = da.block([[da.from_delayed(encode(np.transpose(z[slices[i*np_W + j]],
                                                       transpose_order),
                                          comp_model,
                                          transform,
                                          out_offset),
                                   shape=(1, channels_bn, 1,
                                          out_patch_size,
                                          out_patch_size),
                                   meta=np.empty((), dtype=np.uint8))
                   for j in range(np_W)]
                  for i in range(np_H)])

    comp_H = math.ceil(in_H / 2**compression_level)
    comp_W = math.ceil(in_W / 2**compression_level)

    y = y[..., :comp_H + 2 * out_offset, :comp_W + 2 * out_offset]

    if len(comp_label):
        component = '%s/%s' % (comp_label, data_group)
    else:
        component = data_group

    y.to_zarr(output_filename, component=component, overwrite=True,
              compressor=compressor)

    # Add metadata to the compressed zarr file
    group = zarr.open(output_filename)
    if len(comp_label):
        comp_group = group[comp_label]
    else:
        comp_group = group

    comp_group.attrs['compression_metadata'] = dict(
        height=in_H,
        width=in_W,
        compressed_height=comp_H,
        compressed_width=comp_W,
        channels=in_channels,
        compressed_channels=channels_bn,
        axes='TCZYX',
        compression_level=compression_level,
        patch_size=patch_size,
        offset=add_offset,
        model=str(comp_model),
        model_seed=seed,
        original=data_group,
        group=comp_label,
        version=COMP_VERSION
    )

    # Copy the labels of the original image
    # TODO: Copy the Metadata from the OME folder if any
    if 'zarr' in source_format and output_filename != input_filename:
        z_org = zarr.open(output_filename, mode="rw")
        if 'labels' in z_org.keys() and 'labels' not in group.keys():
            zarr.copy(z_org['labels'], group)


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


def compress(args):
    """Compress any supported file format (zarr, or any supported by PIL) into
    a compressed representation in zarr format.
    """
    logger = logging.getLogger(args.mode + '_log')

    # Open checkpoint from trained model state
    state = utils.load_state(args)

    comp_model = setup_network(state, args.gpu)
    transform, _, _ = utils.get_zarr_transform(normalize=True)

    # Get the compression level from the model checkpoint
    compression_level = state['args']['compression_level']

    if not args.source_format.startswith('.'):
        args.source_format = '.' + args.source_format

    input_fn_list = utils.get_filenames(args.data_dir, args.source_format,
                                        data_mode='all')

    if '.zarr' not in args.output_dir[0].lower():
        # If the output path is a directory, append the input filenames to it,
        # so the compressed files have the same name as the original file.
        fn_list = map(lambda fn:
                      fn.split(args.source_format)[0].replace('\\', '/').split('/')[-1],
                      input_fn_list)
        output_fn_list = [os.path.join(args.output_dir[0],
                                       '%s%s.zarr' % (fn, args.comp_identifier))
                          for fn in fn_list]

    else:
        output_fn_list = args.output_dir

    # Compress each file by separate. This allows to process large images    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        compress_image(
            comp_model=comp_model,
            input_filename=in_fn,
            output_filename=out_fn,
            channels_bn=state['args']['channels_bn'],
            compression_level=compression_level,
            patch_size=args.patch_size,
            add_offset=args.add_offset,
            transform=transform,
            source_format=args.source_format,
            data_axes=args.data_axes,
            data_group=args.data_group,
            seed=state['args']['seed'],
            comp_label=args.task_label_identifier)

        logger.info('Compressed image %s into %s' % (in_fn, out_fn))


if __name__ == '__main__':
    args = utils.get_args(task='encoder', mode='inference')

    utils.setup_logger(args)

    logger = logging.getLogger(args.mode + '_log')

    compress(args)

    logging.shutdown()
