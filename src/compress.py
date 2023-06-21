import dask
from dask.diagnostics import ProgressBar
import dask.array as da
import math

import logging
import os
import shutil
import requests
from itertools import cycle

import aiohttp
import numpy as np
import torch
import zarr

from imagecodecs.numcodecs import Jpeg2k, Jpeg
import numcodecs

import models
import utils

numcodecs.register_codec(Jpeg2k)
numcodecs.register_codec(Jpeg)
numcodecs.register_codec(models.ConvolutionalAutoencoder)
numcodecs.register_codec(models.ConvolutionalAutoencoderBottleneck)


def compress_image(codec, checkpoint, input_filename, output_filename,
                   patch_size=512,
                   source_format='zarr',
                   data_group='0/0',
                   data_axes='TCZYX',
                   progress_bar=False,
                   save_as_bottleneck=False,
                   gpu=False):

    if save_as_bottleneck and "CAE" in codec:
        model = models.autoencoder_from_state_dict(checkpoint=checkpoint,
                                                   gpu=gpu,
                                                   train=False)
        channels_bn = model["fact_ent"].module.channels
        compression_level = len(model["encoder"].module.analysis_track)

        compressor = models.ConvolutionalAutoencoderBottleneck(
            channels_bn=channels_bn,
            fact_ent=model["fact_ent"].module,
            gpu=gpu)

        def compress_fn_impl(chunk):
            with torch.no_grad():
                h, w, c = chunk.shape
                x = torch.from_numpy(chunk.transpose(2, 0, 1))
                x = x.view(1, c, h, w)
                x = x.float() / 255.0

                y = model['encoder'](x)

                y = y[0].cpu().detach().numpy()
                y = y.transpose(1, 2, 0)

            return y

    elif "CAE" in codec:
        compressor = models.ConvolutionalAutoencoder(checkpoint=checkpoint,
                                                     gpu=gpu)
    elif "Blosc" in codec:
        compressor = numcodecs.Blosc(clevel=9)
    elif "Jpeg2k" in codec:
        compressor = Jpeg2k(level=90)
    elif "Jpeg" in codec:
        compressor = Jpeg(level=90)
    elif "None" in codec:
        compressor = None
    else:
        raise ValueError("Codec %s not supported" % codec)

    fn, rois = utils.parse_roi(input_filename, source_format)

    s3_obj = utils.connect_s3(fn)
    z = utils.image2array(fn, source_format, data_group, s3_obj=s3_obj,
                          use_dask=True)

    if len(rois):
        rois = rois[0]
    else:
        rois = [slice(None) for _ in data_axes]

    rem_axes = "".join(set(data_axes) - set("YXC"))
    tran_axes = utils.map_axes_order(data_axes, target_axes=rem_axes + "YXC")

    z = z.transpose(tran_axes)
    rois = [rois[a] for a in tran_axes]
    
    # Select the index 0 from all non-spatial non-color axes
    # TODO: Allow to compress Time and Z axes without hard selecting index 0
    for a in range(len(rem_axes)):
        rois[a] = slice(0, 1, None)

    z = z[tuple(rois)].squeeze(axis=tuple(range(len(rem_axes))))
    z = z.rechunk(chunks=(patch_size, patch_size, 3))

    if save_as_bottleneck and "CAE" in codec:
        comp_chk_y = tuple(int(math.ceil(cs / 2**compression_level))
                           for cs in z.chunks[0])
        comp_chk_x = tuple(int(math.ceil(cs / 2**compression_level))
                           for cs in z.chunks[1])

        comp_chunks = (comp_chk_y, comp_chk_x, (channels_bn,))

        z_cmp = z.map_blocks(compress_fn_impl, dtype=np.float32,
                             chunks=comp_chunks,
                             meta=np.empty((0), dtype=np.float32))

    else:
        z_cmp = z

    if not len(data_group):
        data_group = "0/0"

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
            is_s3 = isinstance(group_src.store.store, zarr.storage.FSStore)

        except (KeyError, aiohttp.client_exceptions.ClientResponseError):
            group_src = zarr.open(fn, mode="r")
            is_s3 = isinstance(group_src.store, zarr.storage.FSStore)

        if is_s3 or not os.path.samefile(output_filename, fn):
            z_org = zarr.open(output_filename, mode="rw")

            if 'labels' in z_org.keys() and 'labels' not in group_dst.keys():
                zarr.copy(z_org['labels'], group_dst)

            if 'masks' in z_org.keys() and 'masks' not in group_dst.keys():
                zarr.copy(z_org['labels'], group_dst)

            # If the source file has metadata (e.g. extracted by bioformats2raw)
            # copy that into the destination zarr file.
            if is_s3:
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
            output_fn_list.append(os.path.join(args.output_dir[0],
                                               '%s.zarr' % fn))

    else:
        output_fn_list = args.output_dir

    # Compress each file by separate. This allows to process large images    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        compress_image(codec=args.codec, checkpoint=args.checkpoint,
                       input_filename=in_fn,
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
