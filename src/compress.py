from dask.diagnostics import ProgressBar
import dask.array as da

import logging
import os
import shutil
import requests

import aiohttp
import zarr

from numcodecs import register_codec

import models
import utils

register_codec(models.ConvolutionalAutoencoder)


def compress_image(checkpoint, input_filename, output_filename,
                   patch_size=512,
                   source_format='zarr',
                   data_group='0/0',
                   data_axes='TCZYX',
                   progress_bar=False,
                   gpu=False):

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

    z = z.transpose(tran_axes)
    z = z.rechunk(chunks=(patch_size, patch_size, 3))

    if progress_bar:
        with ProgressBar():
            z.to_zarr(output_filename, component=data_group, overwrite=True,
                      compressor=compressor)
    else:
        z.to_zarr(output_filename, component=data_group, overwrite=True,
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
                            os.path.join(output_filename, 'OME'),
                            dirs_exist_ok=True)


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
                       gpu=args.gpu)

        logger.info('Compressed image %s into %s' % (in_fn, out_fn))


if __name__ == '__main__':
    args = utils.get_args(task='encoder', mode='inference')

    utils.setup_logger(args)

    logger = logging.getLogger(args.mode + '_log')

    compress(args)

    logging.shutdown()
