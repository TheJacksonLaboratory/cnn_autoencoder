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


def decompress_image(decomp_model, filename, output_dir, channels_org, comp_level, patch_size, offset, workers):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    comp_patch_size = patch_size//2**comp_level

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    histo_ds = utils.ZarrDataset(root=filename, patch_size=comp_patch_size, offset=1 if offset > 0 else 0, transform=None, source_format='zarr')
    data_queue = DataLoader(histo_ds, batch_size=1, num_workers=workers, shuffle=False, pin_memory=True)
    
    H_comp, W_comp = histo_ds.get_shape()
    H = H_comp * 2**comp_level
    W = W_comp * 2**comp_level

    # Output dir is actually the absolute path to the file where to store the compressed representation
    group = zarr.group(output_dir, overwrite=True)
    comp_group = group.create_group('0', overwrite=True)
    
    comp_group.create_dataset('0', shape=(1, channels_org, H, W), chunks=(1, channels_org, patch_size, patch_size), dtype='u1', compressor=compressor)

    z_decomp = zarr.open('%s/0/0' % output_dir, mode='a')

    with torch.no_grad():
        for i, (y_q, _) in enumerate(data_queue):
            y_q = y_q.float() - 127.5
            x = decomp_model(y_q)
            x = 0.5 * x + 0.5

            x = (x * 255).round().to(torch.uint8)
            x = x.detach().cpu().numpy()

            if offset > 0:
                x = x[..., offset:-offset, offset:-offset]
            
            _, tl_y, tl_x = utils.compute_grid(i, 1, H_comp, W_comp, comp_patch_size)
            tl_y *= patch_size
            tl_x *= patch_size
            z_decomp[..., tl_y:(tl_y+patch_size), tl_x:(tl_x+patch_size)] = x


def decompress(args):
    """ Decompress a compressed representation stored in zarr format with the same model used for decompression.
    """    
    logger = logging.getLogger(args.mode + '_log')

    # Open checkpoint from trained model state
    state = utils.load_state(args)

    decomp_model = models.Synthesizer(**state['args'])

    decomp_model.load_state_dict(state['decoder'])

    if torch.cuda.is_available() and args.gpu:
        decomp_model = nn.DataParallel(decomp_model)
        decomp_model.cuda()
    
    decomp_model.eval()

    logger.debug('Model')
    logger.debug(decomp_model)
    
    # Get the compression level from the model checkpoint
    comp_level = state['args']['compression_level']
    offset = (2 ** comp_level) if args.add_offset else 0

    if not args.input[0].endswith('zarr'):
        # If a directory has been passed, get all image files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: fn.endswith(args.source_format), os.listdir(args.input[0]))))
    else:
        input_fn_list = args.input
        
    output_fn_list = list(map(lambda fn: os.path.join(args.output_dir, fn + '_rec.zarr'), map(lambda fn: os.path.splitext(os.path.basename(fn))[0], input_fn_list)))

    # Compress each file by separate. This allows to process large images    
    for in_fn, out_fn in zip(input_fn_list, output_fn_list):
        decompress_image(decomp_model, in_fn, out_fn, state['args']['channels_org'], comp_level, args.patch_size, offset, args.workers)

'''
def decompress_zarr(args):
    """ Decmpress a compressed representation stored in zarr format.
    """
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    logger = logging.getLogger(args.mode + '_log')

    state = utils.load_state(args)

    decomp_model = models.Synthesizer(**state['args'])
    decomp_model.load_state_dict(state['decoder'])

    decomp_model = nn.DataParallel(decomp_model)
    if torch.cuda.is_available() and args.gpu:
        decomp_model.cuda()
    
    decomp_model.eval()
    
    logger.debug('Model')
    logger.debug(decomp_model)

    # Get the compression level from the model checkpoint
    comp_level = state['args']['compression_level']
    offset = (2**comp_level) if args.add_offset else 0
    comp_patch_size = args.patch_size//2**comp_level
    
    if not args.input[0].endswith('.zarr'):
        # If a directory has been passed, get all zarr files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: fn.endswith('.zarr'), os.listdir(args.input[0])))) 
    else:
        input_fn_list = args.input
    
    output_fn_list = list(map(lambda fn: os.path.join(args.output_dir, fn + '_rec.zarr'), map(lambda fn: os.path.splitext(os.path.basename(fn))[0], input_fn_list)))

    for fn_in, fn_out in zip(input_fn_list, output_fn_list):
        histo_ds = utils.Histology_zarr(root=fn_in, patch_size=comp_patch_size, offset=1 if args.add_offset else 0)
        data_queue = DataLoader(histo_ds, batch_size=1, num_workers=args.workers, shuffle=False, pin_memory=True)

        _, _, H, W = histo_ds._z_list[0].shape
        H *= 2**comp_level
        W *= 2**comp_level

        group = zarr.group(fn_out, overwrite=True)
        decomp_group = group.create_group('0', overwrite=True)

        z_decomp = zarr.zeros((1, state['args']['channels_org'], H, W), chunks=(1, state['args']['channels_org'], args.patch_size, args.patch_size), dtype='u1', compressor=compressor)

        with torch.no_grad():
            for i, (y_b, _) in enumerate(data_queue):            
                y_b = y_b.to(torch.float32)
                y_b = y_b - 127.5

                x = decomp_model(y_b)
                x = 255 * (0.5*x + 0.5)

                x = x.round().astype(np.uint8)
                x = x.detach().cpu().numpy()

                if offset > 0:
                    x = x[..., offset:-offset, offset:-offset]

                _, tl_y, tl_x = histo_ds._compute_grid(i)
                tl_y *= args.patch_size
                tl_x *= args.patch_size
                z_decomp[..., tl_y:(tl_y+args.patch_size), tl_x:(tl_x+args.patch_size)] = x

        # Output dir is actually the absolute path to the file where to store the decompressed image
        decomp_group.create_dataset('0', data=z_decomp, dtype='u1', compression=compressor)

    logger.info('Decompressed file from size {} into {}, [{}, {}]'.format(histo_ds._z_list[0].shape, z_decomp.shape, z_decomp[:].min(), z_decomp[:].max()))


def decompress(args):
    """ Decompress a list of pytorch pickled files into images.
    """
    logger = logging.getLogger(args.mode + '_log')
    
    if hasattr(args, 'dataset'):
        if args.dataset == 'MNIST':
            img_ext = 'pgm'
        elif args.dataset == 'ImageNet':
            img_ext = 'jpg'
        elif args.dataset == 'Histology':
            img_ext = 'png'
        else:
            raise ValueError('The dataset \'%s\' is not supported.' % args.dataset)
    else:
        img_ext = args.format
    
    state = utils.load_state(args)

    decomp_model = models.Synthesizer(**state['args'])
    decomp_model.load_state_dict(state['decoder'])
    
    if torch.cuda.is_available() and args.gpu:
        decomp_model = nn.DataParallel(decomp_model)
        decomp_model.cuda()

    decomp_model.eval()
    
    for i, fn in enumerate(args.input):
        y_q = zarr.open(fn, 'r')
        y_q = torch.from_numpy(y_q['0/0'][:].astype(np.float32))
        
        logger.debug('Reconstructing {}'.format(fn))
        
        # y_q =  torch.load(fn)
        y_q = y_q - 127.5
        
        with torch.no_grad():
            x = decomp_model(y_q)
            x = 0.5*x + 0.5
        
        logger.debug('Reconstruction in [{}, {}]'.format(x.min(), x.max()))
        
        utils.save_image(os.path.join(args.output_dir, '{:03d}_rec.{}'.format(i, img_ext)), x)
'''

if __name__ == '__main__':
    args = utils.get_decompress_args()
    
    utils.setup_logger(args)
    
    decompress(args)

    logging.shutdown()