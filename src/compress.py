import dask
from dask.diagnostics import ProgressBar
import dask.array as da

import logging
import os
import shutil
import requests
from itertools import repeat

import aiohttp
import numpy as np
import torch
import zarr

from numcodecs import register_codec

import models
import utils

register_codec(models.ConvolutionalAutoencoder)
register_codec(models.ConvolutionalAutoencoderBottleneck)


def compress_fn_impl(chunk, model):
    with torch.no_grad():
        c, h, w = chunk.shape
        x = torch.from_numpy(chunk)
        x = x.view(1, c, h, w)
        x = x.float() / 255.0

        y = model['encoder'](x)

        y = y[0].cpu().detach().numpy()

    return y


def compress_image(checkpoint, input_filename, output_filename,
                   patch_size=512,
                   source_format='zarr',
                   data_group='0/0',
                   data_axes='TCZYX',
                   progress_bar=False,
                   save_as_bottleneck=False,
                   gpu=False):

    if save_as_bottleneck:
        model = models.autoencoder_from_state_dict(checkpoint=checkpoint,
                                                   gpu=gpu,
                                                   train=False)
        channels_bn = model["fact_ent"].module.channels
        compression_level = len(model["encoder"].module.analysis_track)

        compressor = models.ConvolutionalAutoencoderBottleneck(
            channels_bn=channels_bn,
            fact_ent=model["fact_ent"].module,
            gpu=gpu)
    else:
        compressor = models.ConvolutionalAutoencoder(checkpoint=checkpoint,
                                                     gpu=gpu)

    fn, rois = utils.parse_roi(input_filename, source_format)

    s3_obj = utils.connect_s3(fn)
    z = utils.image2array(fn, source_format, data_group, s3_obj=s3_obj,
                          use_dask=True)

    if len(rois):
        z = z[rois[0]].squeeze()
        rois = rois[0]
    else:
        z = z.squeeze()
        rois = None

    data_axes = [a for a in data_axes if a in 'YXC']
    tran_axes = [data_axes.index(a) for a in 'YXC']

    if save_as_bottleneck:
        z = z.rechunk(chunks=(3, patch_size, patch_size))

        comp_chunks = np.array([(ch // 2**compression_level,
                                 cw // 2**compression_level)
                                for ch, cw in zip(*z.chunks[1:])])

        comp_chunks = tuple([(channels_bn,)] +
                            [tuple(chk) for chk in comp_chunks.T])

        z_cmp = z.map_blocks(compress_fn_impl, model=model, dtype=np.float32,
                             chunks=comp_chunks,
                             meta=np.empty((0), dtype=np.float32))

        z_cmp = z_cmp.transpose(tran_axes)

    else:
        z = z.transpose(tran_axes)
        z = z.rechunk(chunks=(patch_size, patch_size, 3))
        z_cmp = z

    if progress_bar:
        with ProgressBar():
            z_cmp.to_zarr(output_filename, component=data_group,
                          overwrite=True,
                          compressor=compressor)
    else:
        z_cmp.to_zarr(output_filename, component=data_group, overwrite=True,
                      compressor=compressor)

    # Add metadata to the compressed zarr file
    group_dst = zarr.open(output_filename)

    # Copy the labels and metadata from the original image
    if ('zarr' in source_format):
        try:
            group_src = zarr.open_consolidated(fn, mode="r")
        except (KeyError, aiohttp.client_exceptions.ClientResponseError):
            group_src = zarr.open(fn, mode="r")

        if (isinstance(group_src.store, zarr.storage.FSStore)
          or not os.path.samefile(output_filename, fn)):
            z_org = zarr.open(output_filename, mode="rw")
            if 'labels' in z_org.keys() and 'labels' not in group_dst.keys():
                zarr.copy(z_org['labels'], group_dst)

            # If the source file has metadata (e.g. extracted by bioformats2raw)
            # copy that into the destination zarr file.
            if isinstance(group_src.store, zarr.storage.FSStore):
                metadata_resp = requests.get(fn + '/OME/METADATA.ome.xml')
                if metadata_resp.status_code == 200:
                    if not os.path.isdir(os.path.join(output_filename, 'OME')):
                        os.mkdir(os.path.join(output_filename, 'OME'))

                    # Download METADATA.ome.xml into the creted output dir
                    with open(os.path.join(output_filename,
                                           'OME',
                                           'METADATA.ome.xml'), 'wb') as fp:
                        fp.write(metadata_resp.content)

        elif os.path.isdir(os.path.join(fn, 'OME')):
            shutil.copytree(os.path.join(fn, 'OME'),
                            os.path.join(output_filename, 'OME'))


def compress(args):
    """Compress any supported file format (zarr, or any supported by PIL) into
    a compressed representation in zarr format.
    """
    logger = logging.getLogger(args.mode + '_log')

    if not args.source_format.startswith('.'):
        args.source_format = '.' + args.source_format

    input_fn_list = utils.get_filenames(args.data_dir, args.source_format,
                                        data_mode='all')

    if '.zarr' not in args.output_dir[0].lower():
        # If the output path is a directory, append the input filenames to it,
        # so the compressed files have the same name as the original file.
        output_fn_list = []
        for fn in input_fn_list:
            fn = fn[:fn.lower().find(args.source_format)]
            fn = fn.replace('\\', '/').split('/')[-1]
            output_fn_list.append(
                os.path.join(args.output_dir[0],
                             '%s%s.zarr' % (fn, args.task_label_identifier)))

    else:
        output_fn_list = args.output_dir

    # Compress each file by separate. This allows to process large images    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        compress_image(checkpoint=args.checkpoint, input_filename=in_fn,
                       output_filename=out_fn,
                       patch_size=args.patch_size,
                       source_format=args.source_format,
                       data_axes=args.data_axes,
                       data_group=args.data_group,
                       progress_bar=args.progress_bar,
                       save_as_bottleneck=args.save_as_bottleneck,
                       gpu=args.gpu)

        logger.info('Compressed image %s into %s' % (in_fn, out_fn))


if __name__ == '__main__':
    args = utils.get_args(task='encoder', mode='inference')

    utils.setup_logger(args)

    logger = logging.getLogger(args.mode + '_log')

    compress(args)

    logging.shutdown()
