import logging
import os

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


DECOMP_VERSION = '0.1.2'


@dask.delayed
def decode_pyr(y_q, decomp_model, transform, offset=0):
    y_q_t = transform(y_q.squeeze()).unsqueeze(0)

    x_rec = decomp_model(y_q_t)
    if not isinstance(x_rec, list):
        x_rec = [x_rec]

    num_pyr_scales = len(x_rec)
    _, channels, H, W = x_rec[-1].shape
    H -= 2 * offset
    W -= 2 * offset
    x_rec_pyr = np.empty((1, channels, 1, H,
                          W if num_pyr_scales == 1 else int(1.5 * W)))

    prev_h = 0
    for r in reversed(range(num_pyr_scales)):
        x = x_rec[r].detach().cpu()
        x.mul_(127.5)
        x.add_(127.5)
        x.round_()
        x.clip_(0, 255)
        x = x.to(torch.uint8).unsqueeze(2).numpy()

        h, w = x.shape[-2:]
        cl_offset = offset // 2**(num_pyr_scales - 1 - r)
        p_h = h - 2 * cl_offset
        p_w = w - 2 * cl_offset

        if r == 0:
            x_rec_pyr[..., :W] = x[..., cl_offset:cl_offset + p_h,
                                   cl_offset:cl_offset + p_w]
        else:
            x_rec_pyr[..., prev_h:prev_h + p_h, W:W + p_w] = \
                x[..., cl_offset:cl_offset + p_h, cl_offset:cl_offset + p_w]

            prev_h += p_h

    return x_rec_pyr


def decompress_image(decomp_model, input_filename, output_filename,
                     patch_size=512,
                     add_offset=False,
                     transform=None,
                     compute_pyramids=False,
                     destination_format='zarr',
                     data_group='0/0',
                     data_axes='TCZYX',
                     seed=None,
                     decomp_label='reconstruction'):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    src_group = zarr.open(input_filename, mode='r')
    z_arr = src_group[data_group]
    comp_metadata = src_group[data_group.split('/')[0]]
    comp_metadata = comp_metadata.attrs['compression_metadata']

    # Extract the compression level, original channels, and original shape from the compression metadata
    comp_level = comp_metadata['compression_level']
    in_channels = comp_metadata['channels']
    H = comp_metadata['height']
    W = comp_metadata['width']
    comp_label = comp_metadata['group']

    channels_bn, in_H, in_W = [z_arr.shape[data_axes.index(s)] for s in "CYX"]

    in_offset = 0
    out_offset = 0
    in_patch_size = patch_size // 2 ** comp_level
    if add_offset:
        out_offset = 2 ** comp_level
        in_offset = 1

    np_H_prior = utils.compute_num_patches(in_H, in_patch_size, 0,
                                           in_patch_size)
    np_H_prior += (np_H_prior * in_patch_size - in_H) < 0
    pad_y = np_H_prior * in_patch_size - in_H

    np_W_prior = utils.compute_num_patches(in_W, in_patch_size, 0,
                                           in_patch_size)
    np_W_prior += (np_W_prior * in_patch_size - in_W) < 0
    pad_x = np_W_prior * in_patch_size - in_W

    padding = [(in_offset, pad_x + in_offset), (in_offset, pad_y + in_offset),
               (0, 0),
               (0, 0),
               (0, 0)]

    padding = tuple([padding['XYZCT'.index(a)] for a in data_axes])

    np_H = utils.compute_num_patches(in_H, in_patch_size + 2 * in_offset,
                                     pad_y + 2 * in_offset,
                                     in_patch_size)
    np_W = utils.compute_num_patches(in_W, in_patch_size + 2 * in_offset,
                                     pad_x + 2 * in_offset,
                                     in_patch_size)

    z = da.from_zarr(z_arr)
    z = da.pad(z, pad_width=padding, mode='reflect')

    slices = map(lambda ij:
                 [slice(ij[1]*in_patch_size,
                        ij[1]*in_patch_size + in_patch_size + 2 * in_offset, 1),
                  slice(ij[0]*in_patch_size,
                        ij[0]*in_patch_size + in_patch_size + 2 * in_offset, 1),
                  slice(0, 1, 1),
                  slice(0, channels_bn, 1),
                  slice(0, 1, 1)],
                 product(range(np_H), range(np_W)))

    slices = [tuple([None]
              + [s['XYZCT'.index(a)] for a in data_axes])
              for s in slices]

    unused_axis = list(set(data_axes) - set('YXC'))
    transpose_order = [0]
    transpose_order += [data_axes.index(a) + 1 for a in unused_axis]
    transpose_order += [data_axes.index(a) + 1 for a in 'YXC']

    if compute_pyramids:
        rec_patch_size = int(1.5 * patch_size)
    else:
        rec_patch_size = patch_size

    y = da.block(
        [[da.from_delayed(
            decode_pyr(
                np.transpose(z[slices[i*np_W + j]], transpose_order),
                decomp_model,
                transform,
                out_offset),
            shape=(1, in_channels, 1, patch_size, rec_patch_size),
            meta=np.empty((), dtype=np.uint8))
          for j in range(np_W)] for i in range(np_H)])

    data_group = '/'.join(data_group.split(comp_label)[1].split('/')[1:])
    if len(decomp_label):
        component = '%s/%s' % (decomp_label, data_group)
    else:
        component = data_group

    if 'zarr' in destination_format:
        y.to_zarr(output_filename, component=component, overwrite=True,
                  compressor=compressor)

        group = zarr.open(output_filename)
        if len(decomp_label):
            decomp_group = group[decomp_label]
        else:
            decomp_group = group

        decomp_group.attrs['decompression_metadata'] = dict(
            compression_level=comp_level,
            axes='TCZYX',
            patch_size=patch_size,
            offset=add_offset,
            model=str(decomp_model),
            model_seed=seed,
            original=data_group,
            version=DECOMP_VERSION
        )

        if 'zarr' in destination_format and output_filename != input_filename:
            z_org = zarr.open(output_filename, mode="rw")
            if 'labels' in z_org.keys():
                zarr.copy(z_org['labels'], group)
    else:
        # Note that the image should have a number of classes that can be
        # interpreted as a GRAYSCALE image, RGB image or RBGA image.
        im = Image.fromarray(z.reshape(in_channels, H, W).compute())
        im.save(output_filename, destination_format)


def setup_network(state, rec_level=-1, compute_pyramids=False, use_gpu=False):
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
    if compute_pyramids:
        decomp_model = models.SynthesizerInflate(rec_level=rec_level,
                                                 **state['args'])
    else:
        decomp_model = models.Synthesizer(**state['args'])

    decomp_model.load_state_dict(state['decoder'])

    decomp_model = nn.DataParallel(decomp_model)
    if use_gpu:
        decomp_model.cuda()

    decomp_model.eval()

    return decomp_model


def decompress(args):
    """Decompress a compressed representation stored in zarr format with the
    same model used for compression.
    """
    logger = logging.getLogger(args.mode + '_log')

    # Open checkpoint from trained model state
    state = utils.load_state(args)

    decomp_model = setup_network(state, rec_level=args.rec_level, 
                                 use_gpu=args.gpu)
    transform, _, _ = utils.get_zarr_transform(normalize=True,
                                               compressed_input=True)

    if not args.destination_format.startswith('.'):
        args.destination_format = '.' + args.destination_format

    input_fn_list = utils.get_filenames(args.data_dir, source_format='zarr',
                                        data_mode='all')

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

    # Compress each file by separate. This allows to process large images.
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        decompress_image(
            decomp_model=decomp_model,
            input_filename=in_fn,
            output_filename=out_fn,
            patch_size=args.patch_size,
            add_offset=args.add_offset,
            transform=transform,
            compute_pyramids=args.compute_pyramids,
            destination_format=args.destination_format,
            data_axes=args.data_axes,
            data_group=args.data_group,
            seed=state['args']['seed'],
            decomp_label=args.task_label_identifier)

        logger.info('Compressed image %s into %s' % (in_fn, out_fn))


if __name__ == '__main__':
    args = utils.get_args(task='decoder', mode='inference')

    utils.setup_logger(args)

    logger = logging.getLogger(args.mode + '_log')

    decompress(args)

    logging.shutdown()
