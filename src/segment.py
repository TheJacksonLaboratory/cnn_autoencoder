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


def segment(args):
    """ Segment the objects in the images into a set of learned classes.    
    """
    logger = logging.getLogger(args.mode + '_log')

    state = utils.load_state(args)
    
    if state['args']['model_type'] == 'DecoderUNet':
        seg_model = models.DecoderUNet(**state['args'])
        transform = utils.get_histo_transform(normalize=False)
    
    elif state['args']['model_type'] == 'UNet':
        seg_model = models.UNet(**state['args'])
        transform = utils.get_histo_transform(normalize=True)
    
    seg_model.load_state_dict(state['model'])

    seg_model = nn.DataParallel(seg_model)
    if torch.cuda.is_available():
        seg_model.cuda()
    
    seg_model.eval()
    
    # Conver the single zarr file into a dataset to be iterated    
    logger.info('Openning zarr file from {}'.format(args.input))

    offset = 2**state['args']['compression_level']
    histo_ds = utils.Histology_seg_zarr(root=args.input, patch_size=args.patch_size, offset=offset, transform=transform)
    data_queue = DataLoader(histo_ds, batch_size=1, num_workers=args.workers, shuffle=False, pin_memory=True)
    
    H, W = histo_ds._z_list[0].shape[-2:]
    compressor = Blosc(cname='zlib', clevel=1, shuffle=Blosc.BITSHUFFLE)
    
    # Output dir is actually the absolute path to the file where to store the compressed representation
    group = zarr.group(args.output_dir, overwrite=True)
    seg_group = group.create_group('0', overwrite=True)

    z_seg = zarr.zeros((1, state['args']['classes'], H, W), chunks=(1, state['args']['classes'], state['args']['patch_size'], state['args']['patch_size']), compressor=compressor, dtype=np.float32)
    
    with torch.no_grad():
        for i, (x, _) in enumerate(data_queue):
            y = seg_model(x)

            y = y.detach().cpu().numpy()
            if offset > 0:
                y = y[..., offset:-offset, offset:-offset]
            
            _, tl_y, tl_x = histo_ds._compute_grid(i)
            z_seg[..., tl_y:(tl_y+state['args']['patch_size']), tl_x:(tl_x+state['args']['patch_size'])] = y

    logger.info('Segmentation of file of size {} into {}'.format(histo_ds._z_list[0].shape, z_seg.shape))
    seg_group.create_dataset('0', data=z_seg, compression=compressor)
    
    # Save the zarr metadata
    store = group.store
    for key in filter(lambda key: key.endswith(('.zarray', '.zgroup')), store.keys()):
        json_struct = store[key].decode()
        with open(os.path.join(args.output_dir, key), 'w') as f:
            f.write(json_struct)


if __name__ == '__main__':
    args = utils.get_compress_args()
    
    utils.setup_logger(args)
    
    segment(args)
    
    logging.shutdown()