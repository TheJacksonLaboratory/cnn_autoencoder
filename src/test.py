import logging

import numpy as np
import torch
import torch.nn as nn

from models import AutoEncoder, RateDistorsion
from utils import load_state, get_testing_args, setup_logger, get_data


def valid(cae_model, data, criterion):
    cae_model.eval()
    sum_loss = 0

    for x, _ in data:
        x_r, y, p_y = cae_model(x)

        synthesis_model = nn.DataParallel(cae_model.module.synthesis)
        if args.gpu:
            synthesis_model.cuda()

        loss = criterion(x=x, y=y, x_r=x_r, p_y=p_y, synth_net=synthesis_model)

        sum_loss += loss.item()

    mean_loss = sum_loss / len(data)

    return mean_loss


def main(args):
    logger = logging.getLogger(args.mode + '_log')

    state = load_state(args)

    # The autoencoder model contains all the modules
    cae_model = AutoEncoder(**state['args'])
    cae_model.load_state_dict(state['cae_model'])

    # Loss function
    criterion = RateDistorsion(**state['args'])

    cae_model = nn.DataParallel(cae_model)
    criterion = nn.DataParallel(criterion)
    if torch.cuda.is_available():
        cae_model = cae_model.cuda()
        criterion = criterion.cuda()
        args.gpu = True
    else:
        args.gpu = False

    test_data = get_data(args)

    test_loss = valid(cae_model, test_data, criterion)

    logger.info('[Testing] Test loss {:0.4f}'.format(test_loss))


if __name__ == '__main__':
    args = get_testing_args()
    
    setup_logger(args)
    
    main(args)