import logging

import numpy as np
import torch
import torch.nn as nn

from models import AutoEncoder
from criterions import RateDistorsion
from utils import load_state, get_testing_args, setup_logger
from datasets import get_data


def valid(cae_model, data, criterion):
    cae_model.eval()
    sum_loss = 0

    for x, _ in data:
        x_r, p_y = cae_model(x)

        loss = criterion(x, x_r, p_y)

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

    if torch.cuda.is_available():
        cae_model = nn.DataParallel(cae_model).cuda()
        criterion = nn.DataParallel(criterion).cuda()

    test_data = get_data(args)

    test_loss = valid(cae_model, test_data, criterion)

    logger.info('[Testing] Test loss {:0.4f}'.format(test_loss))


if __name__ == '__main__':
    args = get_testing_args()
    
    setup_logger(args)
    
    main(args)