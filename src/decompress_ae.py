# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:51:49 2022

@author: mawes
"""
import numpy as np
import numcodecs as codecs
from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy, ensure_contiguous_ndarray
import tensorflow as tf
from tensorflow_compression.python.ops import gen_ops

import json
import argparse
import logging
import os

import numpy as np

import zarr

VER = '0.5.5'


class CNN_Comp(Codec):
    
    codec_id = 'cnn_compressor'
    
    def __init__(self,cdf_path,percision=16,num_channels=48,chunk_size=100):
        self.cdf_path=cdf_path
        self.percision=percision
        self.num_channels=num_channels
        self.chunk_size=chunk_size
        
    def encode(self,buf):
        #buf=ensure_contiguous_ndarray(buf)
        cdfs=np.load(self.cdf_path)
        cdfs=cdfs*(2**self.percision)
        cdfs=np.round(cdfs)
        cdfs=cdfs.astype(np.int32)
        p=[-self.percision]*self.num_channels
        p=np.array(p,dtype=np.int32)
        p=np.reshape(p, [self.num_channels,1])
        cdfs=np.concatenate([p,cdfs],axis=1)
        cdf=tf.convert_to_tensor(cdfs,tf.int32)
        buf=buf[0,:,0,:,:]
        buf = tf.convert_to_tensor(buf)
        buf=tf.transpose(buf,perm=[1,2,0])
        input_shape = tf.shape(buf)
        symbols = tf.cast(tf.round(buf), tf.int32)
        symbols = tf.reshape(symbols, tf.concat([input_shape, [-1]], 0))
        handle = gen_ops.create_range_encoder([], cdf)
        handle = gen_ops.entropy_encode_channel(handle, symbols)
        strings=gen_ops.entropy_encode_finalize(handle)
        return strings.numpy()
    
    def decode(self, buf, out=None):
        cdfs=np.load(self.cdf_path)
        cdfs=cdfs*(2**self.percision)
        cdfs=np.round(cdfs)
        cdfs=cdfs.astype(np.int32)
        p=[-self.percision]*self.num_channels
        p=np.array(p,dtype=np.int32)
        p=np.reshape(p, [self.num_channels,1])
        cdfs=np.concatenate([p,cdfs],axis=1)
        cdf=tf.convert_to_tensor(cdfs,tf.int32)
        if out is not None:
            out = ensure_contiguous_ndarray(out)
        buf = tf.convert_to_tensor(buf, dtype=tf.string)
        handle = gen_ops.create_range_decoder(buf, cdf)
        handle, symbols = gen_ops.entropy_decode_channel(handle, [self.chunk_size,self.chunk_size,self.num_channels,1],tf.int32)
        s_shape = tf.shape(symbols)
        symbols = tf.reshape(symbols, s_shape[0:3])
        symbols = tf.transpose(symbols,[2,0,1])
        s_shape = tf.shape(symbols)
        symbols=tf.reshape(symbols,[1,s_shape[0],1,s_shape[1],s_shape[2]])
        outputs = tf.cast(symbols, tf.uint8)
        outputs = outputs.numpy()
        return ndarray_copy(outputs, out)
    
    
        
class CNN_Comp2(Codec):
    
    codec_id = 'cnn_compressor'
        
    def __init__(self,cdf,num_channels=48,chunk_size=100):
        self.cdf=cdf
        self.num_channels=num_channels
        self.chunk_size=chunk_size
        
    def encode(self,buf):
        cdf=tf.convert_to_tensor(self.cdf,tf.int32)
        buf=buf[0,:,0,:,:]
        buf = tf.convert_to_tensor(buf)
        buf=tf.transpose(buf,perm=[1,2,0])
        input_shape = tf.shape(buf)
        symbols = tf.cast(tf.round(buf), tf.int32)
        symbols = tf.reshape(symbols, tf.concat([input_shape, [-1]], 0))
        handle = gen_ops.create_range_encoder([], cdf)
        handle = gen_ops.entropy_encode_channel(handle, symbols)
        strings=gen_ops.entropy_encode_finalize(handle)
        return strings.numpy()
        
    def decode(self, buf, out=None):
        cdf=tf.convert_to_tensor(self.cdf,tf.int32)
        if out is not None:
            out = ensure_contiguous_ndarray(out)
        buf = tf.convert_to_tensor(buf, dtype=tf.string)
        handle = gen_ops.create_range_decoder(buf, cdf)
        handle, symbols = gen_ops.entropy_decode_channel(handle, [self.chunk_size,self.chunk_size,self.num_channels,1],tf.int32)
        s_shape = tf.shape(symbols)
        symbols = tf.reshape(symbols, s_shape[0:3])
        symbols = tf.transpose(symbols,[2,0,1])
        s_shape = tf.shape(symbols)
        symbols=tf.reshape(symbols,[1,s_shape[0],1,s_shape[1],s_shape[2]])
        outputs = tf.cast(symbols, tf.uint8)
        outputs = outputs.numpy()
        return ndarray_copy(outputs, out)
    
    
    
class CNN_Comp_Gen(Codec):
    
    codec_id = 'cnn_compressor'
        
    def __init__(self,cdf,num_channels=48,chunk_size=100,nc_config={'id':'zlib','level':9}, org_shape=None):
        self.cdf=cdf
        self.num_channels=num_channels
        self.chunk_size=chunk_size
        self.nc_config=nc_config
        self.org_shape=org_shape
        
    def encode(self,buf):
        cdf=tf.convert_to_tensor(self.cdf,tf.int32)
        buf=buf[0,:,0,:,:]
        buf = tf.convert_to_tensor(buf)
        buf=tf.transpose(buf,perm=[1,2,0])
        input_shape = tf.shape(buf)
        symbols = tf.cast(tf.round(buf), tf.int32)
        symbols = tf.reshape(symbols, tf.concat([input_shape, [-1]], 0))
        #symbols=symbols-127
        handle = gen_ops.create_range_encoder([], cdf)
        handle = gen_ops.entropy_encode_channel(handle, symbols)
        strings=gen_ops.entropy_encode_finalize(handle)
        buf = strings.numpy() #ensure_contiguous_ndarray(strings.numpy())
        codec = codecs.get_codec(self.nc_config)
        return codec.encode(buf)
        
    def decode(self, buf, out=None):
        if out is not None:
            out = ensure_contiguous_ndarray(out)
        codec = codecs.get_codec(self.nc_config)
        dec = codec.decode(buf)
        print('Decoded buffer', len(dec))
        buf = tf.convert_to_tensor(dec, dtype=tf.string)
        cdf=tf.convert_to_tensor(self.cdf,tf.int32)
        handle = gen_ops.create_range_decoder(buf, cdf)
        handle, symbols = gen_ops.entropy_decode_channel(handle, [self.org_shape[0],self.org_shape[1],self.num_channels,1],tf.int32)
        s_shape = tf.shape(symbols)
        #symbols=symbols+127
        symbols = tf.reshape(symbols, s_shape[0:3])
        symbols = tf.transpose(symbols,[2,0,1])
        s_shape = tf.shape(symbols)
        print('Decompressing into original shape', self.org_shape, s_shape)
        symbols=tf.reshape(symbols,[1,s_shape[0],1,s_shape[1],s_shape[2]])
        outputs = tf.cast(symbols, tf.uint8)
        outputs = outputs.numpy()
        print('outputs size', outputs.shape, type(out))
        return ndarray_copy(outputs[:, :, :, :128, :128], out)
    
        
def prep_cdfs(cdf_path, num_channels, percision=16):
    cdfs=np.load(cdf_path)
    cdfs=cdfs*(2**percision)
    cdfs=np.round(cdfs)
    cdfs=cdfs.astype(np.int32)
    p=[-percision]*num_channels
    p=np.array(p,dtype=np.int32)
    p=np.reshape(p, [num_channels,1])
    cdfs=np.concatenate([p,cdfs],axis=1)
    return cdfs.tolist()


def define_buffer_compressor(cdfs, channels_bn,
                    compression_method='bz2',
                    compression_level=9,
                    compression_precision=16,
                    compression_chunk_size=128,
                    org_shape=None):
    # Prepare CDFs for each channel    
    cdfs = prep_cdfs(cdfs, num_channels=channels_bn,
                     percision=compression_precision)
    codecs.register_codec(CNN_Comp_Gen,codec_id="cnn_compressor")
    buffer_codec = codecs.get_codec(dict(id="cnn_compressor", cdf=cdfs,
                                         num_channels=channels_bn,
                                         chunk_size=compression_chunk_size,
                                         nc_config={'id': compression_method,
                                         'level': compression_level},
                                         org_shape=org_shape))
    return buffer_codec


def setup_logger(args):
    """ Sets up a logger for the diferent purposes.
    When training a model on a HPC cluster, it is better to save the logs into a file, rather than printing to a console.
    
    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. Only the code version and random seed are used from this.
    """
    args.version = VER
    
    # Create the logger
    logger = logging.getLogger(args.mode + '_log')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # By now, only if the model is training or testing, the logs are stored into a file
    if args.mode in ['training', 'test']:
        logger_fn = os.path.join(args.log_dir, '%s_ver%s%s.log' % (args.mode, args.version, args.log_identifier))
        fh = logging.FileHandler(logger_fn, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
        logger.setLevel(logging.DEBUG)

    if args.print_log:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    
    logger.info(args)


def get_args():
    task = 'arithmetic_encoder'
    mode = 'inference'
    parser = argparse.ArgumentParser('Arguments for running ' + task + ' in mode ' +  mode, conflict_handler='resolve')
    parser.add_argument('-c', '--config', dest='config_file', type=str, help='A configuration .json file')
    parser.add_argument('-ld', '--logdir', type=str, dest='log_dir', help='Directory where all logging and model checkpoints are stored', default='.')
    parser.add_argument('-li', '--logid', type=str, dest='log_identifier', help='Identifier added to the log file', default='')
    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)
    parser.add_argument('-ps', '--patchsize', type=int, dest='patch_size', help='Size of the patch taken from the orignal image', default=128)

    parser.add_argument('-dg', '--data-group', type=str, dest='data_group', help='For Zarr datasets, the group from where the data is retrieved', default='0/0')
    parser.add_argument('-dd', '--datadir', type=str, nargs='+', dest='data_dir', help='Directory, list of files, or text file with a list of files to be used as inputs.')
    parser.add_argument('-o', '--output', type=str, nargs='+', dest='output_dir', help='Output directory, or list of filenames where to store the compressed image')
    parser.add_argument('-ci', '--identifier', type=str, dest='comp_identifier', help='Identifier added  as suffix to the output filename of a compression/decompression process', default='')
    
    parser.add_argument('-cdf', '--model-cdf', type=str, dest='cdfs', help='CDFs per channel from the trained model')
    parser.add_argument('-cm', '--compression-method', type=str, dest='compression_method', help='Compression method', default='bz2')
    parser.add_argument('-cl', '--compression-level', type=int, dest='compression_level', help='Compression level', default=9)
    parser.add_argument('-cp', '--compression-precision', type=str, dest='compression_precision', help='Precision used for the arithmetic encoding algorithm', default=16)
    
    args = parser.parse_args()

    config_parser = argparse.ArgumentParser(parents=[parser], add_help=False)

    # Parse the arguments from a json configure file, when given
    if args.config_file is not None:
        if '.json' in args.config_file:
            config = json.load(open(args.config_file, 'r'))
            config_parser.set_defaults(**config)

        else:
            raise ValueError('The configure file must be a .json file')

    # The parameters passed through a json file are overridable from console instructions
    args = config_parser.parse_args()
    
    args.mode = 'inference'
    args.task = 'arithmetic_encoder'

    return args


def decompress_image(input_filename, output_filename, data_group='0/0', **kwargs):
    compressor = codecs.Blosc(cname='zlib', clevel=9, shuffle=codecs.Blosc.BITSHUFFLE)
    
    # This file should have been compressed using the Arithmetic encoding compressor
    z_org = zarr.open(input_filename, mode='r')
    comp_H = z_org['compressed'].attrs['compression_metadata']['compressed_height']
    comp_W = z_org['compressed'].attrs['compression_metadata']['compressed_width']

    cdfs = np.load(args.cdfs)
    channels_bn = cdfs.shape[0]
    _ = define_buffer_compressor(kwargs['cdfs'],
                                 channels_bn,
                                 kwargs['compression_method'],
                                 kwargs['compression_level'],
                                 kwargs['compression_precision'],
                                 kwargs['patch_size'],
                                 org_shape=(comp_H, comp_W))
    
    if os.path.isdir(output_filename):
        group = zarr.open_group(output_filename, mode='rw')
    else:
        group = zarr.group(output_filename)

    comp_group = group.create_group('compressed', overwrite=True)

    print('Source\n', z_org.info)
    comp_group.attrs['compression_metadata'] = \
        z_org['compressed'].attrs['compression_metadata']
    print('Compressed shape\n', z_org['compressed/' + data_group].shape, z_org['compressed/' + data_group].chunks)
    z_comp = comp_group.create_dataset(data_group,
                                       shape=z_org['compressed/' + data_group].shape,
                                       chunks=z_org['compressed/' + data_group].chunks,
                                       dtype=z_org['compressed/' + data_group].dtype,
                                       compressor=compressor)

    # Copying pxels from compressed with AE to intermediate compressed representation
    print('Decompress', z_org['compressed/' + data_group].shape, 'into:', z_comp.shape)
    z_comp[:]=z_org['compressed/' + data_group][:]

    if z_org is not None \
       and 'labels' in z_org.keys() and z_org.store.path != group.store.path:
        zarr.copy(z_org['labels'], group)


def decompress(args):
    """ Decompress any supported file format (zarr, or any supported by PIL) into
    a compressed representation in zarr format.
    """
    logger = logging.getLogger(args.mode + '_log')
    args.destination_format = '.zarr'

    if isinstance(args.data_dir, zarr.Group):
        input_fn_list = [args.data_dir]
    elif '.zarr' not in args.data_dir[0].lower():
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
        decompress_image(in_fn, out_fn, **args.__dict__)

        logger.info('Decompressed image %s into %s' % (in_fn, out_fn))


if __name__ == '__main__':
    args = get_args()

    setup_logger(args)

    logger = logging.getLogger(args.mode + '_log')

    decompress(args)

    logging.shutdown()
