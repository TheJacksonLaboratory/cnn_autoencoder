import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel
import torch.optim as optim

from models import AutoEncoder, RateDistorsion, RateDistorsionPenaltyA, RateDistorsionPenaltyB, EarlyStoppingPatience, EarlyStoppingTarget
from utils import save_state, get_training_args, setup_logger, get_data

from collections import namedtuple
from itertools import chain
from functools import reduce
from inspect import signature

scheduler_options = {"ReduceOnPlateau": optim.lr_scheduler.ReduceLROnPlateau}


def checkpoint(step, cae_model, optimizer, scheduler, valid_loss, best_valid_loss, train_loss_history, valid_loss_history):
    # Create a dictionary with the current state as checkpoint
    training_state = dict(
        encoder=cae_model.module.analysis.state_dict(),
        decoder=cae_model.module.synthesis.state_dict(),
        fact_ent=cae_model.module.fact_entropy.state_dict(),
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

        training_state['scheduler'] = scheduler.state_dict()
    else:
        training_state['scheduler'] = None
    
    save_state('last', training_state, args)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        save_state('best', training_state, args)
    
    return best_valid_loss


def valid(cae_model, data, criterion, args):
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
            
            synthesizer = DataParallel(cae_model.module.synthesis)
            if args.gpu:
                synthesizer.cuda()

            loss, _ = criterion(x=x, y=y, x_r=x_r, p_y=p_y, synth_net=synthesizer)

            sum_loss += loss.item()

            if i % max(1, int(0.1 * len(data))) == 0:
                logger.debug('\t[{:04d}/{:04d}] Validation Loss {:.4f} ({:.4f}). Quantized compressed representation in [{:.4f}, {:.4f}], reconstruction in [{:.4f}, {:.4f}]'.format(i, len(data), loss.item(), sum_loss / (i+1), y.detach().min(), y.detach().max(), x_r.detach().min(), x_r.detach().max()))

    mean_loss = sum_loss / len(data)

    return mean_loss


def train(cae_model, train_data, valid_data, criterion, stopping_criteria, optimizer, scheduler, args):
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
    stopping_criteria : list[StoppingCriterion]
        Stopping criteria tracker for different problem statements
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

    best_valid_loss = float('inf')
    train_loss_history = []
    valid_loss_history = []

    step = 0
    while stopping_criteria[0].check():
        # Reset the average loss computation every epoch
        sum_loss = 0

        for i, (x, _) in enumerate(train_data):
            step += 1

            # Training step
            optimizer.zero_grad()

            x_r, y, p_y = cae_model(x)
            
            synthesizer = DataParallel(cae_model.module.synthesis)
            if args.gpu:
                synthesizer.cuda()
            
            loss, extra_info = criterion(x=x, y=y, x_r=x_r, p_y=p_y, synth_net=synthesizer)
            
            loss.backward()

            optimizer.step()
            sum_loss += loss.item()

            if scheduler is not None and 'metrics' not in dict(signature(scheduler.step).parameters).keys():
                scheduler.step()

            # When training with penalty on the energy of the compression representation, 
            # update the respective stopping criterion
            if len(stopping_criteria) > 1:
                stopping_criteria[1].update(iteration=step, metric=extra_info.item())

            # Log the training performance every 10% of the training set
            if i % max(1, int(0.01 * len(train_data))) == 0:
                logger.debug('\t[{:04d}/{:04d}] Training Loss {:.4f} ({:.4f}). Quantized compressed representation in [{:.4f}, {:.4f}], reconstruction in [{:.4f}, {:.4f}]'.format(i, len(train_data), loss.item(), sum_loss / (i+1), y.detach().min(), y.detach().max(), x_r.detach().min(), x_r.detach().max()))

            # Checkpoint step
            keep_training = reduce(lambda sc1, sc2: sc1 & sc2, map(lambda sc: sc.check(), stopping_criteria), True)

            if not keep_training or step % args.checkpoint_steps == 0:
                train_loss = sum_loss / step

                # Evaluate the model in the validation set
                valid_loss = valid(cae_model, valid_data, criterion, args)
                
                cae_model.train()

                stopping_info = ';'.join(map(lambda sc: sc.__repr__(), stopping_criteria))

                # If there is a learning rate scheduler, perform a step
                # Log the overall network performance every checkpoint step
                logger.info('[Step {:06d} ({})] Training loss {:0.4f}, validation loss {:.4f}, best validation loss {:.4f}, learning rate {:e}, stopping criteria: {}'.format(
                    step, 'training' if keep_training else 'stopping', train_loss, valid_loss, best_valid_loss, optimizer.param_groups[0]['lr'], stopping_info)
                )

                train_loss_history.append(train_loss)
                valid_loss_history.append(valid_loss)

                best_valid_loss = checkpoint(step, cae_model, optimizer, scheduler, valid_loss, best_valid_loss, train_loss_history, valid_loss_history)

                stopping_criteria[0].update(iteration=step, metric=valid_loss)
            else:
                stopping_criteria[0].update(iteration=step)
            
            if not keep_training:
                logging.info('\n**** Stopping criteria met: Interrupting training ****')
                break
        
    else:
        completed = True

    # Return True if the training finished sucessfully
    return completed


def setup_network(args):
    # The autoencoder model contains all the modules
    cae_model = AutoEncoder(**args.__dict__)

    cae_model = nn.DataParallel(cae_model)

    if args.gpu:
        cae_model.cuda()

    return cae_model


def setup_criteria(args):
    # Early stopping criterion:
    stopping_criteria = [EarlyStoppingPatience(max_iterations=args.steps, **args.__dict__)]

    # Loss function
    if args.criterion == 'RD_PA':
        criterion = RateDistorsionPenaltyA(**args.__dict__)
        stopping_criteria.append(EarlyStoppingTarget(mode='le', max_iterations=args.steps, target=args.energy_limit))

    elif args.criterion == 'RD_PB':
        criterion = RateDistorsionPenaltyB(**args.__dict__)
        stopping_criteria.append(EarlyStoppingTarget(mode='ge', max_iterations=args.steps, target=args.energy_limit))

    elif args.criterion == 'RD':
        criterion = RateDistorsion(**args.__dict__)

    else:
        raise ValueError('Criterion \'%s\' not supported' % args.criterion)

    criterion = nn.DataParallel(criterion)
    if args.gpu:
        criterion = criterion.cuda()

    return criterion, stopping_criteria


def setup_optim(cae_model, args):
    # Otpimizer:
    optimizer = optim.Adam(params=cae_model.parameters(), lr=args.learning_rate)
    
    if args.scheduler == 'None':
        scheduler = None
    elif args.scheduler in scheduler_options.keys():
        scheduler = scheduler_options[args.scheduler](optimizer=optimizer, mode='min')
    else:
        raise ValueError('Scheduler \"%s\" is not implemented' % args.scheduler)

    return optimizer, scheduler


def resume_checkpoint(cae_model, optimizer, scheduler, args):
    if not args.gpu:
        checkpoint_state = torch.load(args.resume, map_location=torch.device('cpu'))
    else:
        checkpoint_state = torch.load(args.resume)
    
    cae_model.module.analysis.load_state_dict(checkpoint_state['encoder'])
    cae_model.module.synthesis.load_state_dict(checkpoint_state['decoder'])
    cae_model.module.fact_entropy.load_state_dict(checkpoint_state['fact_ent'])

    optimizer.load_state_dict(checkpoint_state['optimizer'])

    if scheduler is not None and checkpoint_state['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint_state['scheduler'])


def main(args):
    """ Set up the training environment

    Parameters
    ----------
    args : dict or Namespace

    """
    logger = logging.getLogger(args.mode + '_log')

    cae_model = setup_network(args)
    criterion, stopping_criteria = setup_criteria(args)
    optimizer, scheduler = setup_optim(cae_model, args)

    if args.resume is not None:
        resume_checkpoint(cae_model, optimizer, scheduler, args)

    # Log the training setup
    logger.info('Network architecture:')
    logger.info(cae_model)
    
    logger.info('\nCriterion:')
    logger.info(criterion)
    
    logger.info('\nStopping criterion:')
    logger.info(stopping_criteria[0])
    
    if len(stopping_criteria) > 1:
        logger.info('\nAdditinal stopping criterions:')
        logger.info(stopping_criteria[1])

    logger.info('\nOptimization parameters:')
    logger.info(optimizer)
    
    logger.info('\nScheduler parameters:')
    logger.info(scheduler)

    train_data, valid_data = get_data(args)
    
    train(cae_model, train_data, valid_data, criterion, stopping_criteria, optimizer, scheduler, args)


if __name__ == '__main__':
    args = get_training_args()

    setup_logger(args)

    main(args)