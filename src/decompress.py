from dask.diagnostics import ProgressBar

import logging
import os
import shutil
import requests

import zarr
import dask.array as da

from PIL import Image
from numcodecs import Blosc, register_codec

import models
import utils

register_codec(models.ConvolutionalAutoencoder)


def decompress_image(input_filename, output_filename,
                     destination_format='zarr',
                     data_group='0/0',
                     decomp_label='reconstruction',
                     progress_bar=False):

    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    fn, rois = utils.parse_roi(input_filename, '.zarr')

    src_group = zarr.open(fn, mode='r')
    z_arr = src_group[data_group]

    if len(rois):
        z = da.from_zarr(z_arr)[rois[0]]
        rois = rois[0]
    else:
        z = da.from_zarr(z_arr)
        rois = None

    if len(decomp_label):
        component = '%s/%s' % (decomp_label, data_group)
    else:
        component = data_group

    if 'zarr' in destination_format:
        comp_pyr = '/'.join(component.split('/')[:-1])
        comp_r = comp_pyr + '/%i' % 0
        if progress_bar:
            with ProgressBar():
                z.to_zarr(output_filename, component=comp_r, overwrite=True,
                          compressor=compressor)
        else:
            z.to_zarr(output_filename, component=comp_r, overwrite=True,
                      compressor=compressor)

        group = zarr.open(output_filename)

        # Copy the labels of the original image
        if (isinstance(z_arr.store, zarr.storage.FSStore)
           or not os.path.samefile(output_filename, input_filename)):
            z_org = zarr.open(output_filename, mode="rw")
            if 'labels' in z_org.keys() and 'labels' not in group.keys():
                zarr.copy(z_org['labels'], group)

            # If the source file has metadata (e.g. extracted by
            # bioformats2raw) copy that the destination zarr file.
            if isinstance(z_arr.store, zarr.storage.FSStore):
                metadata_resp = requests.get(input_filename
                                             + '/OME/METADATA.ome.xml')
                if metadata_resp.status_code == 200:
                    os.mkdir(os.path.join(output_filename, 'OME'))
                    # Download METADATA.ome.xml into the creted output dir
                    with open(os.path.join(output_filename,
                                           'OME',
                                           'METADATA.ome.xml'),
                              'wb') as fp:
                        fp.write(metadata_resp.content)

            elif os.path.isdir(os.path.join(input_filename, 'OME')):
                shutil.copytree(os.path.join(input_filename, 'OME'),
                                os.path.join(output_filename, 'OME'),
                                dirs_exist_ok=True)
    else:
        # Note that the image should have a number of classes that can be
        # interpreted as a GRAYSCALE image, RGB image or RBGA image.
        fn_out_base = output_filename.split(destination_format)[0]

        fn_out = fn_out_base + destination_format
        if progress_bar:
            with ProgressBar():
                im = Image.fromarray(z.compute())
        else:
            im = Image.fromarray(z.compute())

        im.save(fn_out, quality_opts={'compress_level': 9,
                                        'optimize': False})


def decompress(args):
    """Decompress a compressed representation stored in zarr format with the
    same model used for compression.
    """
    logger = logging.getLogger(args.mode + '_log')

    if not args.destination_format.startswith('.'):
        args.destination_format = '.' + args.destination_format

    input_fn_list = utils.get_filenames(args.data_dir, source_format='.zarr',
                                        data_mode='all')

    if args.destination_format.lower() not in args.output_dir[0].lower():
        # If the output path is a directory, append the input filenames to it,
        # so the compressed files have the same name as the original file.
        output_fn_list = []
        for fn in input_fn_list:
            fn = fn.split('.zarr')[0].replace('\\', '/').split('/')[-1]
            output_fn_list.append(
                os.path.join(args.output_dir[0], 
                             '%s%s%s' % (fn,args.task_label_identifier,
                                         args.destination_format)))

    else:
        output_fn_list = args.output_dir

    # Compress each file by separate. This allows to process large images.
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        decompress_image(input_filename=in_fn, output_filename=out_fn,
                         destination_format=args.destination_format,
                         data_group=args.data_group,
                         decomp_label=args.task_label_identifier,
                         progress_bar=args.progress_bar)

        logger.info('Compressed image %s has been decompressed into %s' % (in_fn, out_fn))


if __name__ == '__main__':
    args = utils.get_args(task='decoder', mode='inference')

    utils.setup_logger(args)

    logger = logging.getLogger(args.mode + '_log')

    decompress(args)

    logging.shutdown()
