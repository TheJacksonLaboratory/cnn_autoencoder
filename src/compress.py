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


def compress_image(comp_model, filename, output_dir, channels_bn, comp_level, patch_size, offset, transform, workers, source_format, is_labeled=False):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    histo_ds = utils.ZarrDataset(root=filename, patch_size=patch_size, offset=offset, transform=transform, source_format=source_format)
    data_queue = DataLoader(histo_ds, batch_size=1, num_workers=workers, shuffle=False, pin_memory=True)
    
    H, W = histo_ds.get_shape()
    comp_patch_size = patch_size//2**comp_level

    # Output dir is actually the absolute path to the file where to store the compressed representation
    group = zarr.group(output_dir, overwrite=True)
    comp_group = group.create_group('0', overwrite=True)
    
    comp_group.create_dataset('0', shape=(1, channels_bn, int(np.ceil(H/2**comp_level)), int(np.ceil(W/2**comp_level))), chunks=(1, channels_bn, comp_patch_size, comp_patch_size), dtype='u1', compressor=compressor)

    z_comp = zarr.open('%s/0/0' % output_dir, mode='a')

    with torch.no_grad():
        for i, (x, _) in enumerate(data_queue):
            y_q, _ = comp_model(x)
            y_q = y_q + 127.5

            y_q = y_q.round().to(torch.uint8)
            y_q = y_q.detach().cpu().numpy()

            if offset > 0:
                y_q = y_q[..., 1:-1, 1:-1]
                
            _, tl_y, tl_x = utils.compute_grid(i, 1, H, W, patch_size)
            tl_y *= comp_patch_size
            tl_x *= comp_patch_size
            z_comp[..., tl_y:(tl_y+comp_patch_size), tl_x:(tl_x+comp_patch_size)] = y_q
    
    if is_labeled:
        label_group = group.create_group('1', overwrite=True)
        z_org = zarr.open(filename, 'r')
        zarr.copy(z_org['1/0'], label_group)
    

def compress(args):
    """ Compress any supported file format (zarr, or any supported by PIL) into a compressed representation in zarr format.
    """    
    logger = logging.getLogger(args.mode + '_log')

    # Open checkpoint from trained model state
    state = utils.load_state(args)

    embedding = models.ColorEmbedding(**state['args'])
    comp_model = models.Analyzer(**state['args'])

    embedding.load_state_dict(state['embedding'])
    comp_model.load_state_dict(state['encoder'])

    comp_model = nn.Sequential(embedding, comp_model)

    if torch.cuda.is_available() and args.gpu:
        comp_model = nn.DataParallel(comp_model)
        comp_model.cuda()
    
    comp_model.eval()

    logger.debug('Model')
    logger.debug(comp_model)
    
    # Conver the single zarr file into a dataset to be iterated
    transform = utils.get_zarr_transform(normalize=True)

    # Get the compression level from the model checkpoint
    comp_level = state['args']['compression_level']
    offset = (2 ** comp_level) if args.add_offset else 0

    if not args.input[0].endswith('zarr'):
        # If a directory has been passed, get all image files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: fn.endswith(args.source_format), os.listdir(args.input[0]))))
    else:
        input_fn_list = args.input
        
    output_fn_list = list(map(lambda fn: os.path.join(args.output_dir, fn + '_comp.zarr'), map(lambda fn: os.path.splitext(os.path.basename(fn))[0], input_fn_list)))

    # Compress each file by separate. This allows to process large images    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        compress_image(comp_model, in_fn, out_fn, state['args']['channels_bn'], comp_level, args.patch_size, offset, transform, args.workers, source_format=args.source_format, is_labeled=args.is_labeled)


'''
def compress_zarr(args):
    """ Compress a whole image into a compressed zarr format.
    The image must be provided as a zarr file
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
    
    # Conver the single zarr file into a dataset to be iterated
    transform = utils.get_histology_transform(normalize=True)

    comp_level = state['args']['compression_level']
    offset = (2 ** comp_level) if args.add_offset else 0

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
                y_q = y_q + 127.5

                y_q = y_q.round().to(torch.uint8)                
                y_q = y_q.detach().cpu().numpy()

                if offset > 0:
                    y_q = y_q[..., 1:-1, 1:-1]
            
                _, tl_y, tl_x = utils.compute_grid(i, )
                tl_y *= comp_patch_size
                tl_x *= comp_patch_size
                z_comp[..., tl_y:(tl_y+comp_patch_size), tl_x:(tl_x+comp_patch_size)] = y_q

        comp_group.create_dataset('0', data=z_comp, dtype='u1', compression=compressor)
        
        if args.is_labeled:
            label_group = group.create_group('1', overwrite=True)
            z_org = zarr.open( histo_ds._filenames[0], 'r')
            zarr.copy(z_org['1/0'], label_group)
        
        logger.debug(' Compressed file of size{} into {}, saving in: {}'.format(histo_ds._z_list[0].shape, z_comp.shape, out_fn))
        

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

        logger.debug('Compressed representation: {} in [{}, {}], from [{}, {}]'.format(y_q.size(), y_q.min(), y_q.max(), x.min(), x.max()))

        # Save the compressed representation as the output of the cnn autoencoder
        y_q = y_q.detach().cpu().numpy()
                
        _, channels_bn, H, W = y_q.shape
        
        out_fn = os.path.join(args.output_dir, '{:03d}_comp.zarr'.format(i))
        
        group = zarr.group(out_fn, overwrite=True)
        comp_group = group.create_group('0', overwrite=True)

        comp_group.create_dataset('0', data=y_q, chunks=(1, channels_bn, H, W), dtype='u1', compressor=compressor)
'''

if __name__ == '__main__':
    args = utils.get_compress_args()
    
    utils.setup_logger(args)
    
    compress(args)
    
    logging.shutdown()