import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import AutoEncoder
from criterions import RateDistorsion
from utils import save_state, get_training_args, setup_logger
from datasets import get_data

from itertools import chain
from inspect import getfullargspec


def train(cae_model, data, criterion, optimizer):
    cae_model.train()

    sum_loss = 0

    for x, _ in data:
        
        optimizer.zero_grad()

        x_r, p_y = cae_model(x)

        loss = criterion(x, x_r, p_y)

        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    mean_loss = sum_loss / len(data)

    return mean_loss


def valid(cae_model, data, criterion):
    cae_model.train()
    sum_loss = 0

    for x, _ in data:
        x_r, p_y = cae_model(x)

        loss = criterion(x, x_r, p_y)

        sum_loss += loss.item()

    mean_loss = sum_loss / len(data)

    return mean_loss


def main(args):
    logger = logging.getLogger('training_log')

    # The autoencoder model contains all the modules
    cae_model = AutoEncoder(args.input_channels, args.net_channels, args.bn_channels, args.compression_level, args.channels_expansion, K=args.factorized_entropy_K, r=args.factorized_entropy_r)

    # Loss function
    criterion = RateDistorsion(args.distorsion_lambda)

    if torch.cuda.is_available():
        cae_model = nn.DataParallel(cae_model).cuda()
        criterion = nn.DataParallel(criterion).cuda()

    optimizer = optim.Adam(params=cae_model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min')

    train_data, valid_data = get_data(args)

    best_valid_loss = float('inf')
    train_loss_history = []
    valid_loss_history = []

    for e in range(args.epochs):        
        train_loss = train(cae_model, train_data, criterion, optimizer)
        valid_loss = valid(cae_model, valid_data, criterion)

        logger.info('[Epoch {:3d}] Training loss {:0.4f}, validation loss {:.4f}, best validation loss {:.4f}'.format(
            e, train_loss, valid_loss, best_valid_loss)
            )

        if scheduler is not None:
            if 'metrics' in getfullargspec(scheduler.step).args:
                scheduler.step(valid_loss)
            else:
                scheduler.step()

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
   
        if valid_loss < best_valid_loss or e % args.checkpoint_epochs == 0:
            # Create a dictionary with the current state as checkpoint
            training_state = dict(
                cae_model=cae_model.state_dict(),
                optimizer=optimizer.state_dict(),
                args=args,
                best_val=best_valid_loss,
                epoch=e,
                train_loss=train_loss_history,
                valid_loss=valid_loss_history,
                code_version=args.version
            )
            
            if scheduler is not None:
                training_state['scheduler'] = scheduler.state_dict()
            else:
                training_state['scheduler'] = None            
            
            if e % args.checkpoint_epochs == 0:
                save_state('last', training_state, args)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_state('best', training_state, args)


if __name__ == '__main__':
    args = get_training_args()
    setup_logger(args)
   
    main(args)