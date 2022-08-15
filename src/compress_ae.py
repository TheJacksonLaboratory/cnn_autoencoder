import argparse
import logging
import os

import numpy as np
import torch

import zarr

import utils


def compress_image(input_filename, output_filename, compressor, data_group='0/0'):
    if os.path.isdir(output_filename):
        group = zarr.open_group(output_filename, mode='rw')
    else:
        group = zarr.group(output_filename)

    comp_group = group.create_group('compressed', overwrite=True)

    # Add metadata to the compressed zarr file
    z_org = zarr.open(input_filename, mode='r')
    comp_group.attrs['compression_metadata'] = \
        z_org['compressed'].attrs['compression_metadata']

    z_comp = comp_group.create_dataset(data_group,
                                       shape=z_org[data_group].shape,
                                       chunks=z_org[data_group].chunks,
                                       dtype=z_org[data_group].dtype,
                                       compressor=compressor)

    if z_org is not None \
       and 'labels' in z_org.keys() and z_org.store.path != group.store.path:
        zarr.copy(z_org, group, 'labels')


def compress(args):
    """ Compress any supported file format (zarr, or any supported by PIL) into
    a compressed representation in zarr format.
    """
    logger = logging.getLogger(args.mode + '_log')

    # Open checkpoint from trained model state
    state = utils.load_state(args)

    cdfs = torch.load(args.cdfs).numpy()
    compressor = utils.define_buffer_compressor(cdfs,
                                                state['args']['channels_bn'],
                                                args.compression_method,
                                                args.compression_level,
                                                args.compression_precision)

    args.destination_format = '.zarr'

    if isinstance(args.data_dir, zarr.Group):
        input_fn_list = [args.data_dir]
    elif args.source_format.lower() not in args.data_dir[0].lower():
        # If a directory has been passed, get all image files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.data_dir[0], fn), filter(lambda fn: '.zarr' in fn.lower(), os.listdir(args.data_dir[0]))))
    else:
        input_fn_list = args.data_dir

    if args.destination_format.lower() not in args.output_dir[0].lower():
        # If the output path is a directory, append the input filenames to it, so the compressed files have the same name as the original file
        fn_list = map(lambda fn: fn.split('.zarr')[0].replace('\\', '/').split('/')[-1], input_fn_list)
        output_fn_list = [os.path.join(args.output_dir[0], '%s%s.zarr' % (fn, args.comp_identifier)) for fn in fn_list]
    else:
        output_fn_list = args.output_dir

    # Compress each file by separate. This allows to process large images    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        comp_group = compress_image(in_fn, out_fn, compressor,
                                    data_group=args.data_group)

        logger.info('Compressed image %s into %s' % (in_fn, out_fn))


if __name__ == '__main__':
    parser = utils.get_args(task='arithmetic_encoder', mode='inference', add_model=False, add_criteria=False, add_config=False, add_logging=False, add_data=True, parser_only=True)

    args = utils.override_config_file(parser)
    args.mode = 'inference'
    args.task = 'arithmetic_encoder'

    utils.setup_logger(args)

    logger = logging.getLogger(args.mode + '_log')

    for _ in compress(args):
        logger.info('Image compressed successfully')

    logging.shutdown()
