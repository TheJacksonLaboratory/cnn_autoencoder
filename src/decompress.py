from dask.diagnostics import ProgressBar

import logging
import os
import shutil
import requests

import numpy as np
import torch
import zarr
import dask
import dask.array as da

from PIL import Image
from numcodecs import Blosc, register_codec

import models
import utils

register_codec(models.ConvolutionalAutoencoderBottleneck)
register_codec(models.ConvolutionalAutoencoder)


def decompress_fn_impl(chunk, model):
    with torch.no_grad():
        h, w, c = chunk.shape

        y_q = torch.from_numpy(chunk).permute(2, 0, 1)
        y_q = y_q.view(1, c, h, w)

        x_r, _ = model['decoder'](y_q)

        x_r = x_r[0][0].cpu().detach() * 255.0
        x_r = x_r.clip(0, 255).to(torch.uint8)
        x_r = x_r.permute(1, 2, 0).numpy()

    return x_r


def decompress_image(input_filename, output_filename,
                     destination_format='zarr',
                     data_group='0/0',
                     decomp_label='reconstruction',
                     checkpoint=None,
                     progress_bar=False,
                     gpu=False):

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

    if (checkpoint is None
      or (isinstance(checkpoint, str) and not len(checkpoint))):
        z_r = z

    else:
        model = models.autoencoder_from_state_dict(checkpoint=checkpoint,
                                                   gpu=gpu,
                                                   train=False)

        compression_level = model["decoder"].module.rec_level

        decomp_chunks = np.array([(ch * 2**compression_level,
                                   cw * 2**compression_level)
                                   for ch, cw in zip(*z.chunks[:2])])

        decomp_chunks = tuple([tuple(chk) for chk in decomp_chunks.T] + [(3,)])

        z_r = z.map_blocks(decompress_fn_impl, model=model, dtype=np.uint8,
                           chunks=decomp_chunks,
                           meta=np.empty((0), dtype=np.uint8))

    if len(decomp_label):
        component = '%s/%s' % (decomp_label, data_group)
    else:
        component = data_group

    if 'zarr' in destination_format:
        comp_pyr = '/'.join(component.split('/')[:-1])
        comp_r = comp_pyr + '/%i' % 0

        if progress_bar:
            with ProgressBar():
                z_r.to_zarr(output_filename, component=comp_r, overwrite=True,
                            compressor=compressor)
        else:
            z_r.to_zarr(output_filename, component=comp_r, overwrite=True,
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
                                os.path.join(output_filename, 'OME'))
    else:
        # Note that the image should have a number of classes that can be
        # interpreted as a GRAYSCALE image, RGB image or RBGA image.
        fn_out_base = output_filename.split(destination_format)[0]

        fn_out = fn_out_base + destination_format
        if progress_bar:
            with ProgressBar():
                im = Image.fromarray(z_r.compute())
        else:
            im = Image.fromarray(z_r.compute())

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
        logger.info('Decompressing %s into %s' % (in_fn, out_fn))
        decompress_image(input_filename=in_fn, output_filename=out_fn,
                         destination_format=args.destination_format,
                         data_group=args.data_group,
                         decomp_label=args.task_label_identifier,
                         progress_bar=args.progress_bar,
                         checkpoint=args.checkpoint,
                         gpu=args.gpu)


if __name__ == '__main__':
    args = utils.get_args(task='decoder', mode='inference')

    utils.setup_logger(args)

    logger = logging.getLogger(args.mode + '_log')

    decompress(args)

    logging.shutdown()
