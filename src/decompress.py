import logging
import os

import numpy as np
import torch
import torch.nn as nn

from models import Synthesizer

from utils import get_decompress_args, load_state, setup_logger, open_compressed, save_image


def decompress(args):
    """" Deompress a list of pytorch pickled files into images.
    """
    
    if hasattr(args, 'dataset'):
        if args.dataset == 'MNIST':
            img_ext = 'pgm'
        elif args.dataset == 'ImageNet':
            img_ext = 'jpg'
        else:
            raise ValueError('The dataset \'%s\' is not supported.' % args.dataset)
    else:
        img_ext = args.format
    
    state = load_state(args)

    decomp_model = Synthesizer(**state['args'])

    # Load only the analysis track from the trained model:
    decomp_state_dict = {}
    for k in filter(lambda k: 'synthesis' in k, state['cae_model'].keys()):
        decomp_module_name = '.'.join(filter(lambda m: m != 'synthesis', k.split('.')))
        decomp_state_dict[decomp_module_name] = state['cae_model'][k]

    decomp_model = nn.DataParallel(decomp_model)
    if torch.cuda.is_available():
        decomp_model.cuda()
    
    decomp_model.load_state_dict(decomp_state_dict)
    
    decomp_model.eval()
    
    for i, fn in enumerate(args.input):
        y_q = open_compressed(fn)
        x = decomp_model(y_q)
        save_image(os.path.join(args.output_dir, '{:03d}_rec.{}'.format(i, img_ext)), x)


if __name__ == '__main__':
    args = get_decompress_args()
    
    setup_logger(args)
    
    decompress(args)
