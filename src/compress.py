import logging
import os

import numpy as np
import torch
import torch.nn as nn

from models import Analyzer

from utils import get_compress_args, load_state, setup_logger, open_image, save_compressed


def compress(args):
    """" Compress a list of images into pytorch pickled files.
    The images can be provided as a lis of tensors, or a tensor stacked in the first dimension.
    """
    logger = logging.getLogger(args.mode + '_log')

    state = load_state(args)

    comp_model = Analyzer(**state['args'])

    # Load only the analysis track from the trained model:
    comp_state_dict = {}
    for k in filter(lambda k: 'analysis' in k, state['cae_model'].keys()):
        comp_module_name = '.'.join(filter(lambda m: m != 'analysis', k.split('.')))
        comp_state_dict[comp_module_name] = state['cae_model'][k]

    comp_model = nn.DataParallel(comp_model)
    if torch.cuda.is_available():
        comp_model.cuda()
    
    comp_model.load_state_dict(comp_state_dict)

    comp_model.eval()

    for i, fn in enumerate(args.input):
        x = open_image(fn, state['args']['compression_level'])

        y_q, _ = comp_model(x)
        
        logger.info('Compressed representation: {} in [{}, {}]'.format(y_q.size(), y_q.min(), y_q.max()))
        
        save_compressed(os.path.join(args.output_dir, '{:03d}.pth'.format(i)), y_q)


if __name__ == '__main__':
    args = get_compress_args()
    
    setup_logger(args)
    
    compress(args)