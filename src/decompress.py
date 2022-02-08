import zarr
import logging
import os

import numpy as np
import torch
import torch.nn as nn

import models

import utils


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
    
    decomp_model = nn.DataParallel(decomp_model)

    if torch.cuda.is_available():
        decomp_model.cuda()

    decomp_model.eval()
    
    for i, fn in enumerate(args.input):        
        y_q = zarr.open(fn, 'r')
        y_q = torch.from_numpy(y_q['0/0'][:].astype(np.float32))
        y_q = y_q - 127.5
        
        with torch.no_grad():
            x = decomp_model(y_q)
            x = 0.5*x + 0.5
        
        logger.info('Reconstruction in [{}, {}]'.format(x.min(), x.max()))
        utils.save_image(os.path.join(args.output_dir, '{:03d}_rec.{}'.format(i, img_ext)), x)


if __name__ == '__main__':
    args = utils.get_decompress_args()
    
    utils.setup_logger(args)
    
    decompress(args)

    logging.shutdown()