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
from inspect import getfullargspec, signature


def valid(cae_model, data, criterion):
    """ Validation step.
    Evaluates the performance of the network in its current state using the full set of validation elements.

    Parameters
    ----------
    cae_model : torch.nn.Module
        The network model in the current state
    data : torch.utils.data.DataLoader or list[tuple]
        The validation dataset. Because the target is recosntruct the input, the label associated is ignored
    criterion : function or torch.nn.Module
        The loss criterion to evaluate the network's performance
    
    Returns
    -------
    mean_loss : float
        Mean value of the criterion function over the full set of validation elements
    """
    logger = logging.getLogger(args.mode + '_log')

    cae_model.eval()
    sum_loss = 0

    with torch.no_grad():
        for i, (x, _) in enumerate(data):
            x_r, y, p_y = cae_model(x)

            if isinstance(cae_model, nn.DataParallel):
                synthesis_model = nn.DataParallel(cae_model.module.synthesis).cuda()                
            else:
                synthesis_model = cae_model.synthesis

            loss = criterion(x=x, y=y, x_r=x_r, p_y=p_y, synth_net=synthesis_model)

            sum_loss += loss.item()

            if i % max(1, int(0.1 * len(data))) == 0:
                logger.debug('\t[{:04d}/{:04d}] Validation Loss {:.4f} ({:.4f})'.format(i, len(data), loss.item(), sum_loss / (i+1)))

    mean_loss = sum_loss / len(data)

    return mean_loss


def train(cae_model, train_data, valid_data, criterion, optimizer, scheduler, args):
    """ Training loop by steps

    Parameters
    ----------
    cae_model : torch.nn.Module
        The model to be trained
    train_data : torch.utils.data.DataLoader or list[tuple]
        The training data. Must contain the input and respective label; however, only the input is used because the target is reconstructing the input
    valid_data : torch.utils.data.DataLoader or list[tuple]
        The validation data.
    criterion : function or torch.nn.Module
        The loss criterion to evaluate the network's performance
    optimizer : torch.optim.Optimizer
        The parameter's optimizer method
    scheduler : torch.optim.lr_scheduler or None
        If provided, a learning rate scheduler for the optimizer
    args : dict or Namespace
        The dictionary of input arguments passed at running time
    
    Returns
    -------
    completed : bool
        Whether the training was sucessfully completed or it was interrupted
    """
    logger = logging.getLogger(args.mode + '_log')

    completed = False
    step = 1
    sum_loss = 0

    best_valid_loss = float('inf')
    train_loss_history = []
    valid_loss_history = []

    while step <= args.steps:
        for i, (x, _) in enumerate(train_data):
            # Training step
            optimizer.zero_grad()

            x_r, y, p_y = cae_model(x)

            if isinstance(cae_model, nn.DataParallel):
                synthesis_model = nn.DataParallel(cae_model.module.synthesis).cuda()                
            else:
                synthesis_model = cae_model.synthesis

            loss = criterion(x=x, y=y, x_r=x_r, p_y=p_y, synth_net=synthesis_model)

            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            # Log the training performance every 10% of the training set
            if i % max(1, int(0.1 * len(train_data))) == 0:
                logger.debug('\t[{:04d}/{:04d}] Training Loss {:.4f} ({:.4f})'.format(i, len(train_data), loss.item(), sum_loss / (i+1)))
    
            # Checkpoint step
            if step == args.steps or step % args.checkpoint_steps == 0:
                train_loss = sum_loss / step

                # Evaluate the model in the validation set
                valid_loss = valid(cae_model, valid_data, criterion)
                cae_model.train()

                # If there is a learning rate scheduler, perform a step
                # Log the overall network performance every checkpoint step
                logger.info('[Step {:6d}] Training loss {:0.4f}, validation loss {:.4f}, best validation loss {:.4f}'.format(
                    step, train_loss, valid_loss, best_valid_loss)
                )

                train_loss_history.append(train_loss)
                valid_loss_history.append(valid_loss)

                # Create a dictionary with the current state as checkpoint
                training_state = dict(
                    cae_model=cae_model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    args=args.__dict__,
                    best_val=best_valid_loss,
                    step=step,
                    train_loss=train_loss_history,
                    valid_loss=valid_loss_history,
                    code_version=args.version
                )
                
                if scheduler is not None:
                    if 'metrics' in dict(signature(scheduler.step).parameters).keys():
                        scheduler.step(metrics=valid_loss)
                    else:
                        scheduler.step()

                    training_state['scheduler'] = scheduler.state_dict()
                else:
                    training_state['scheduler'] = None
                
                save_state('last', training_state, args)
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    save_state('best', training_state, args)

            step += 1
            if step > args.steps:
                break
    else:
        completed = True

    # Return True if the training finished sucessfully
    return completed
   

def main(args):
    """ Set up the training environment

    Parameters
    ----------
    args : dict or Namespace

    """
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

    # TODO: Different schedulers can be used here, or even let it as None.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min')

    if args.resume is not None:
        if torch.cuda.is_available():
            checkpoint_state = torch.load(args.resume, map_location=torch.device('cpu'))
        else:
            checkpoint_state = torch.load(args.resume)
        
        cae_model.load_state_dict(checkpoint_state['cae_model'])
        optimizer.load_state_dict(checkpoint_state['optimizer'])

        if scheduler is not None and checkpoint_state['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint_state['scheduler'])

    # Log the training setup
    logger.info('Network architecture:')
    logger.info(cae_model)
    
    logger.info('Criterion:')
    logger.info(criterion)

    logger.info('Optimization parameters:')
    logger.info(optimizer)
    
    logger.info('Scheduler parameters:')
    logger.info(optimizer)

    train_data, valid_data = get_data(args)
    
    train(cae_model, train_data, valid_data, criterion, optimizer, scheduler, args)


if __name__ == '__main__':
    args = get_training_args()

    setup_logger(args)

    main(args)