import logging

import numpy as np
import torch
import torch.nn as nn

from models import Analyzer, Quantizer

from utils import get_compress_args, load_state, setup_logger
from datasets import open_image, save_compressed


def main(args):
    logger = logging.getLogger(args.mode + '_log')

    state = load_state(args)

    comp_model = Analyzer(**state['args'])
    quantizer = Quantizer()

    # Load only the analysis track from the trained model:
    comp_state_dict = {}
    for k in filter(lambda k: k.split('.')[0] == 'analysis', state['cae_model'].keys()):
        comp_state_dict['.'.join(k.split('.')[1:])] = state['cae_model'][k]
    
    comp_model.load_state_dict(comp_state_dict)

    if torch.cuda.is_available():
        comp_model = nn.DataParallel(comp_model).cuda()

    comp_model.eval()
    quantizer.eval()

    x = open_image(args.input, state['args']['compression_level'])
    y = comp_model(x)
    y_q = quantizer(y).to(torch.uint8)

    logger.info('Image of shape {}, compressed into a pytorch tensor of size {} in the range of [{}, {}]'.format(x.size(), y_q.size(), y_q.min(), y_q.max()))

    save_compressed(args.output, y_q)


if __name__ == '__main__':
    args = get_compress_args()
    
    setup_logger(args)
    
    main(args)