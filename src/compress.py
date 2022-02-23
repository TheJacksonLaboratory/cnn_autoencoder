import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import zarr
from numcodecs import Blosc

import models

import utils
import sys

def compress_zarr(args):
    """ Compress a whole image into a compressed zarr format.
    The image must be provided as a zarr file
    """
    compressor = Blosc(cname='zlib', clevel=0, shuffle=Blosc.BITSHUFFLE)
    
    logger = logging.getLogger(args.mode + '_log')

    state = utils.load_state(args)

    embedding = models.ColorEmbedding(**state['args'])
    comp_model = models.Analyzer(**state['args'])

    embedding.load_state_dict(state['embedding'])
    comp_model.load_state_dict(state['encoder'])

    if torch.cuda.is_available() and args.gpu:
        embedding = nn.DataParallel(embedding)
        comp_model = nn.DataParallel(comp_model)
        embedding.cuda()
        comp_model.cuda()
    
    embedding.eval()
    comp_model.eval()

    logger.debug('Color embedding\n', embedding)
    logger.debug('Model\n', comp_model)
    
    # Conver the single zarr file into a dataset to be iterated
    transform = utils.get_histo_transform(normalize=True)

    comp_level = state['args']['compression_level']
    offset = (2 ** comp_level) if args.add_offset else 0

    logger.info('Openning zarr file from {}'.format(args.input))
    
    if not args.input[0].endswith('.zarr'):
        # If a directory has been passed, get all zarr files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: fn.endswith('.zarr'), os.listdir(args.input[0]))))
    else:
        input_fn_list = args.input
        
    output_fn_list = list(map(lambda fn: os.path.join(args.output_dir, fn + '_comp.zarr'), map(lambda fn: os.path.splitext(os.path.basename(fn))[0], input_fn_list)))
    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        histo_ds = utils.Histology_seg_zarr(root=in_fn, patch_size=args.patch_size, offset=offset, transform=transform)
        data_queue = DataLoader(histo_ds, batch_size=1, num_workers=args.workers, shuffle=False, pin_memory=True)
        
        H, W = histo_ds._z_list[0].shape[-2:]
        comp_patch_size = args.patch_size//2**comp_level

        # Output dir is actually the absolute path to the file where to store the compressed representation
        group = zarr.group(out_fn, overwrite=True)
        comp_group = group.create_group('0', overwrite=True)

        z_comp = zarr.zeros((1, state['args']['channels_bn'], int(np.ceil(H/2**comp_level)), int(np.ceil(W/2**comp_level))), chunks=(1, state['args']['channels_bn'], comp_patch_size, comp_patch_size), dtype='u1', compressor=compressor)

        with torch.no_grad():
            for i, (x, _) in enumerate(data_queue):
                fx = embedding(x)
                y_q, _ = comp_model(fx)
                logger.info('Compress: {}, ({}, {})'.format(y_q.size(), y_q.min(), y_q.max()))
            
                y_q = y_q + 127.5

                y_q = y_q.round().to(torch.uint8)
                
                y_q = y_q.detach().cpu().numpy()
                if offset > 0:
                    y_q = y_q[..., 1:-1, 1:-1]
            
                _, tl_y, tl_x = histo_ds._compute_grid(i)
                tl_y *= comp_patch_size
                tl_x *= comp_patch_size
                z_comp[..., tl_y:(tl_y+comp_patch_size), tl_x:(tl_x+comp_patch_size)] = y_q

        comp_group.create_dataset('0', data=z_comp, dtype='u1', compression=compressor)
        
        if args.is_labeled:
            label_group = group.create_group('1', overwrite=True)
            z_org = zarr.open( histo_ds._filenames[0], 'r')
            zarr.copy(z_org['1/0'], label_group)
        
        logger.info(' Compressed file of size{} into {}, saving in: {}'.format(histo_ds._z_list[0].shape, z_comp.shape, out_fn))
        

def compress(args):
    """ Compress a list of images into binary files.
    The images can be provided as a lis of tensors, or a tensor stacked in the first dimension.
    """
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)
    
    logger = logging.getLogger(args.mode + '_log')

    state = utils.load_state(args)

    embedding = models.ColorEmbedding(**state['args'])
    comp_model = models.Analyzer(**state['args'])

    embedding.load_state_dict(state['embedding'])
    comp_model.load_state_dict(state['encoder'])

    if torch.cuda.is_available() and args.gpu:
        embedding = nn.DataParallel(embedding)
        comp_model = nn.DataParallel(comp_model)
    
        embedding.cuda()
        comp_model.cuda()
    
    embedding.eval()
    comp_model.eval()
    
    logger.debug('Color embedding')
    logger.debug(embedding)
    logger.debug('Model')
    logger.debug(comp_model)
    
    for i, fn in enumerate(args.input):
        x = utils.open_image(fn, state['args']['compression_level'])

        with torch.no_grad():           
            fx = embedding(x)
            y_q, _ = comp_model(fx)
            y_q = y_q + 127.5
            torch.save(y_q, os.path.join(args.output_dir, '{:03d}_comp.pth'.format(i)))
            y_q = y_q.round().to(torch.uint8)

        logger.info('Compressed representation: {} in [{}, {}], from [{}, {}]'.format(y_q.size(), y_q.min(), y_q.max(), x.min(), x.max()))

        # Save the compressed representation as the output of the cnn autoencoder
        y_q = y_q.detach().cpu().numpy()
                
        _, channels_bn, H, W = y_q.shape
        
        out_fn = os.path.join(args.output_dir, '{:03d}_comp.zarr'.format(i))
        
        group = zarr.group(out_fn, overwrite=True)
        comp_group = group.create_group('0', overwrite=True)

        comp_group.create_dataset('0', data=y_q, chunks=(1, channels_bn, H, W), dtype='u1', compressor=compressor)
        

if __name__ == '__main__':
    args = utils.get_compress_args()
    
    utils.setup_logger(args)
    
    compress(args)
    
    logging.shutdown()