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
        
    def __init__(self,cdf,num_channels=48,chunk_size=100,nc_config={'id':'zlib','level':9}):
        self.cdf=cdf
        self.num_channels=num_channels
        self.chunk_size=chunk_size
        self.nc_config=nc_config
        
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
        buf = tf.convert_to_tensor(dec, dtype=tf.string)
        cdf=tf.convert_to_tensor(self.cdf,tf.int32)
        handle = gen_ops.create_range_decoder(buf, cdf)
        handle, symbols = gen_ops.entropy_decode_channel(handle, [self.chunk_size,self.chunk_size,self.num_channels,1],tf.int32)
        s_shape = tf.shape(symbols)
        #symbols=symbols+127
        symbols = tf.reshape(symbols, s_shape[0:3])
        symbols = tf.transpose(symbols,[2,0,1])
        s_shape = tf.shape(symbols)
        symbols=tf.reshape(symbols,[1,s_shape[0],1,s_shape[1],s_shape[2]])
        outputs = tf.cast(symbols, tf.uint8)
        outputs = outputs.numpy()
        return ndarray_copy(outputs, out)
    
        
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
                    compression_precision=16):
    # Prepare CDFs for each channel    
    cdfs = prep_cdfs(cdfs, num_channels=channels_bn,
                     percision=compression_precision)

    buffer_codec = codecs.get_codec(dict(id="cnn_compressor", cdf=cdfs,
                                         num_channels=channels_bn,
                                         chunk_size=128,
                                         nc_config={'id': compression_method,
                                         'level': compression_level}))
    return buffer_codec
