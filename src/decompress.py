import struct
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from .models import Synthesizer, Decoder

from .utils import get_decompress_args, load_state, setup_logger, open_compressed, save_image


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
    
    state = load_state(args)

    decomp_model = Synthesizer(**state['args'])

    decomp_model.load_state_dict(state['decoder'])
    
    decomp_model = nn.DataParallel(decomp_model)

    if torch.cuda.is_available():
        decomp_model.cuda()

    decomp_model.eval()

    decoder = Decoder(512)
    
    for i, fn in enumerate(args.input):
        y_q = torch.load(os.path.join(args.output_dir, '{:03d}.pth'.format(i)))
        
        logger.info('Decompressed representation: {} in [{}, {}]'.format(y_q.size(), y_q.min(), y_q.max()))
        
        x = decomp_model(y_q)
        
        logger.info('Reconstruction in [{}, {}]'.format(x.min(), x.max()))

        x = 0.5*x + 0.5

        save_image(os.path.join(args.output_dir, '{:03d}_rec.{}'.format(i, img_ext)), x)

        with open(os.path.join(args.output_dir, '{:03d}.comp'.format(i)), mode='rb') as f:
            # Write the size of the image:
            size_b = f.read(16)
            size = struct.unpack('IIII', size_b)

            # Write the compressed bitstream
            y_b = f.read()
        
        y_q = decoder(y_b, size)

        logger.info('Decompressed representation (AE): {} in [{}, {}]'.format(y_q.size(), y_q.min(), y_q.max()))
        
        x = decomp_model(y_q)

        logger.info('Reconstruction (AE) in [{}, {}]'.format(x.min(), x.max()))

        x = 0.5*x + 0.5
        
        save_image(os.path.join(args.output_dir, '{:03d}_ae_rec.{}'.format(i, img_ext)), x)


if __name__ == '__main__':
    args = get_decompress_args()
    
    setup_logger(args)
    
    decompress(args)

    logging.shutdown()