from dask.diagnostics import ProgressBar

import logging
import os
import shutil
import requests

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
from compressai.entropy_models import EntropyBottleneck
import utils
from lc import Algorithm


COMP_VERSION = '0.1.3'


@dask.delayed
def encode(x, comp_model, transform, offset=0):

    x_t = transform(x.squeeze()).unsqueeze(0)
    H, W = x_t.shape[-2:]

    with torch.set_grad_enabled(comp_model.training):
        fx = comp_model['embedding'](x_t)
        y = comp_model['analysis'](fx)
        y_q, _ = comp_model['fact_ent'](y)
        y_cmp = comp_model['fact_ent'].module.compress(y)

        print('AE comp', len(y_cmp[0]), (8 * len(y_cmp[0])) / (H * W))

        y_q = y_q.cpu()
        h, w = y_q.shape[-2:]
        y_q = y_q[..., offset:h-offset, offset:w-offset]

        y_q = y_q - comp_model['fact_ent'].module.quantiles[..., 0].unsqueeze(-1)
        y_q = y_q.round().to(torch.int16)
        y_q = y_q.unsqueeze(2).numpy()

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
    fn, rois = utils.parse_roi(input_filename, source_format)

    s3_obj = utils.connect_s3(fn)
    z_arr, _, _ = utils.image_to_zarr(fn, patch_size, source_format,
                                      data_group, s3_obj=s3_obj)

    if not isinstance(z_arr, zarr.core.Array):
        z_arr = zarr.array(data=z_arr[:])

    a_ch, a_H, a_W = [data_axes.index(a) for a in "CYX"]
    in_channels = z_arr.shape[a_ch]
    if len(rois):
        in_H = (rois[0][a_H].stop - rois[0][a_H].start) // rois[0][a_H].step
        in_W = (rois[0][a_W].stop - rois[0][a_W].start) // rois[0][a_W].step
    else:
        in_H = z_arr.shape[a_H]
        in_W = z_arr.shape[a_W]

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

    if len(rois):
        z = da.from_zarr(z_arr)[rois[0]]
        rois = rois[0]
    else:
        z = da.from_zarr(z_arr)
        rois = None

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

    y_blocks = []
    for i in range(np_H):
        y_blocks.append([])
        for j in range(np_W):
            y_blocks[-1].append(
                da.from_delayed(
                    encode(
                        np.transpose(z[slices[i*np_W + j]], transpose_order),
                        comp_model,
                        transform,
                        out_offset),
                    shape=(1, channels_bn, 1, out_patch_size, out_patch_size),
                    meta=np.empty((), dtype=np.int16)))
    y = da.block(y_blocks)

    comp_H = math.ceil(in_H / 2**compression_level)
    comp_W = math.ceil(in_W / 2**compression_level)

    y = y[..., :comp_H + 2 * out_offset, :comp_W + 2 * out_offset]

    if len(comp_label):
        component = '%s/%s' % (comp_label, data_group)
    else:
        component = data_group

    with ProgressBar():
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
        rois=str(rois),
        version=COMP_VERSION
    )

    # Copy the labels of the original image
    if ('zarr' in source_format
       and (isinstance(z_arr.store, zarr.storage.FSStore)
            or not os.path.samefile(output_filename, fn))):
        z_org = zarr.open(output_filename, mode="rw")
        if 'labels' in z_org.keys() and 'labels' not in group.keys():
            zarr.copy(z_org['labels'], group)

        # If the source file has metadata (e.g. extracted by bioformats2raw)
        # copy that the destination zarr file.
        if isinstance(z_arr.store, zarr.storage.FSStore):
            metadata_resp = requests.get(fn + '/OME/METADATA.ome.xml')
            if metadata_resp.status_code == 200:
                os.mkdir(os.path.join(output_filename, 'OME'))
                # Download METADATA.ome.xml into the creted output dir
                with open(os.path.join(output_filename,
                                       'OME',
                                       'METADATA.ome.xml'),
                          'wb') as fp:
                    fp.write(metadata_resp.content)

        elif os.path.isdir(os.path.join(fn, 'OME')):
            shutil.copytree(os.path.join(fn, 'OME'),
                            os.path.join(output_filename, 'OME'),
                            dirs_exist_ok=True)


def setup_network(state, use_gpu=False, lc_pretrained_model=None,
                  ft_pretrained_model=None):
    """ Setup a neural network-based image compression model.

    Parameters
    ----------
    state : Dictionary
        A checkpoint state saved during the network training
    lc_pretrained_model : path to model or None
        Path to where the model that has been compressed using the LC algorithm is stored.
    ft_pretrained_model : path to model or None
        Path to where the model that has been fine tuned using the LC algorithm is stored.

    Returns
    -------
    comp_model : torch.nn.Module
        The compressor model
    """
    if state['args']['version'] in ['0.5.5', '0.5.6']:
        state['args']['act_layer_type'] = 'LeakyReLU'

    cae_model_base = models.AutoEncoder(**state['args'])

    cae_model_base.embedding.load_state_dict(state['embedding'])
    cae_model_base.analysis.load_state_dict(state['encoder'])
    cae_model_base.fact_ent.load_state_dict(state['fact_ent'], strict=False)
    cae_model_base.fact_ent.update()

    if lc_pretrained_model is not None and ft_pretrained_model is not None:

        # Load the model checkpoint from its compressed version
        lc_compressed_model_state = torch.load(lc_pretrained_model,
                                               map_location='cpu')
        ft_compressed_model_state = torch.load(ft_pretrained_model,
                                               map_location='cpu')['model_state']

        cae_model_base = nn.DataParallel(cae_model_base)
        cae_model_base = utils.load_compressed_dict(cae_model_base,
                                                    lc_compressed_model_state,
                                                    ft_compressed_model_state,
                                                    conv_scheme='scheme_2')
        cae_model_base = cae_model_base.module

    comp_model = nn.ModuleDict(
        dict(embedding=nn.DataParallel(cae_model_base.embedding),
             analysis=nn.DataParallel(cae_model_base.analysis),
             fact_ent=nn.DataParallel(cae_model_base.fact_ent)))

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

    comp_model = setup_network(state, args.gpu,
                               lc_pretrained_model=args.lc_pretrained_model,
                               ft_pretrained_model=args.ft_pretrained_model)
    transform, _, _ = utils.get_zarr_transform(**args.__dict__)

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
                      fn[:fn.lower().find(args.source_format)].replace('\\', '/').split('/')[-1],
                      input_fn_list)
        output_fn_list = [os.path.join(args.output_dir[0],
                                       '%s%s.zarr' % (fn, args.task_label_identifier))
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
