import logging

import numpy as np
import torch
from torch._C import Value
import torch.nn as nn
import torch.optim as optim

from models import AutoEncoder
from criterions import RateDistorsion, RateDistorsionPenaltyA, RateDistorsionPenaltyB
from utils import save_state, get_training_args, setup_logger
from datasets import get_data

from itertools import chain
from inspect import getfullargspec


def train(cae_model, data, criterion, optimizer):
    logger = logging.getLogger(args.mode + '_log')

    cae_model.train()

    sum_loss = 0

    for i, (x, _) in enumerate(data):
        optimizer.zero_grad()

        x_r, y, p_y = cae_model(x)

        loss = criterion(x=x, y=y, x_r=x_r, p_y=p_y, synth_net=cae_model.synthesis)

        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

        if i % int(0.1 * len(data)) == 0:
            logger.debug('[{:04d}/{:04d}] Training Loss {:.4f} ({:.4f})'.format(i, len(data), loss.item(), sum_loss / (i+1)))

    mean_loss = sum_loss / len(data)

    return mean_loss


def valid(cae_model, data, criterion):
    logger = logging.getLogger(args.mode + '_log')

    cae_model.eval()
    sum_loss = 0

    with torch.no_grad():
        for i, (x, _) in enumerate(data):
            x_r, y, p_y = cae_model(x)

            loss = criterion(x=x, y=y, x_r=x_r, p_y=p_y, synth_net=cae_model.synthesis)

            sum_loss += loss.item()

            if i % int(0.1 * len(data)) == 0:
                logger.debug('[{:04d}/{:04d}] Validation Loss {:.4f} ({:.4f})'.format(i, len(data), loss.item(), sum_loss / (i+1)))

    mean_loss = sum_loss / len(data)

    return mean_loss


def main(args):
    logger = logging.getLogger(args.mode + '_log')

    # The autoencoder model contains all the modules
    cae_model = AutoEncoder(**args.__dict__)

    # Loss function
    if args.criterion == 'RD_PA':
        criterion = RateDistorsionPenaltyA(**args.__dict__)
    elif args.criterion == 'RD_PB':
        criterion = RateDistorsionPenaltyB(**args.__dict__)
    elif args.criterion == 'RD':
        criterion = RateDistorsion(**args.__dict__)
    else:
        raise ValueError('Criterion \'%s\' not supported' % args.criterion)

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
                args=args.__dict__,
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