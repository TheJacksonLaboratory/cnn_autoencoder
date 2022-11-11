import logging
from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (accuracy_score,
                             recall_score,
                             precision_score,
                             f1_score)

import models
import utils
import segment

from functools import reduce
from inspect import signature

scheduler_options = {
    "ReduceOnPlateau": partial(optim.lr_scheduler.ReduceLROnPlateau,
                               mode='min',
                               patience=2)}
seg_model_types = {"UNetNoBridge": models.UNet,
                   "UNet": models.UNet,
                   "DecoderUNet": models.DecoderUNet}


def valid(seg_model, data, criterion, logger, args, forward_fun=None):
    """Validation step.
    Evaluates the performance of the network in its current state using the full set of validation elements.

    Parameters
    ----------
    seg_model : torch.nn.Module
        The network model in the current state
    data : torch.utils.data.DataLoader or list[tuple]
        The validation dataset. Because the target is recosntruct the input, the label associated is ignored
    criterion : function or torch.nn.Module
        The loss criterion to evaluate the network's performance
    logger: logger
        Current logger used to track thre model performance during training
    args : Namespace
        The input arguments passed at running time
    forward_fun: function
        Function used to perform the feed-forward step

    Returns
    -------
    mean_loss : float
        Mean value of the criterion function over the full set of validation elements
    """

    seg_model.eval()
    sum_loss = 0
    if args.print_log:
        q = tqdm(total=len(data), desc='Validating', position=1, leave=None)
    with torch.no_grad():
        for i, (x, t) in enumerate(data):
            y = forward_fun(x)

            loss = criterion(y, t)
            # In case that distributed computation of the criterion ouptuts a vector instead of a scalar
            loss = torch.mean(loss)
            sum_loss += loss.item()

            if args.print_log:
                t_flat = t[:, -1, ...].numpy().flatten()
                y_flat = y.detach().cpu().numpy().flatten() > 0.5
                acc = accuracy_score(t_flat, y_flat)
                recall = recall_score(t_flat, y_flat, zero_division=0)
                prec = precision_score(t_flat, y_flat, zero_division=0)
                f1 = f1_score(t_flat, y_flat, zero_division=0)
                q.set_description('Validation Loss {:.4f} ({:.4f}: acc={:0.4f}, prec={:0.4f}, recall={:0.4f}, f1={:.4f}).'.format(
                    loss.item(), sum_loss / (i+1), acc, prec, recall, f1))
                q.update()
            elif i % max(1, int(0.1 * len(data))) == 0:
                t_flat = t[:, -1, ...].numpy().flatten()
                y_flat = y.detach().cpu().numpy().flatten() > 0.5
                acc = accuracy_score(t_flat, y_flat)
                recall = recall_score(t_flat, y_flat, zero_division=0)
                prec = precision_score(t_flat, y_flat, zero_division=0)
                f1 = f1_score(t_flat, y_flat, zero_division=0)
                logger.debug('\t[{:04d}/{:04d}] Validation Loss {:.4f} ({:.4f}: acc={:0.4f}, prec={:0.4f}, recall={:0.4f}, f1={:.4f}).'.format(
                    i, len(data), loss.item(), sum_loss / (i+1), acc, prec, recall, f1))

    if args.print_log:
        q.close()
    mean_loss = sum_loss / len(data)

    return mean_loss


def train(seg_model, train_data, valid_data, criterion, stopping_criteria, optimizer, scheduler, args, forward_fun=None):
    """Training loop by steps.
    This loop involves validation and network training checkpoint creation.

    Parameters
    ----------
    seg_model : torch.nn.Module
        The model to be trained
    train_data : torch.utils.data.DataLoader or list[tuple]
        The training data. Must contain the input and respective label; however, only the input is used because the target is reconstructing the input
    valid_data : torch.utils.data.DataLoader or list[tuple]
        The validation data.
    criterion : function or torch.nn.Module
        The loss criterion to evaluate the network's performance
    stopping_criteria : StoppingCriterion
        Stopping criteria tracker for different problem statements
    optimizer : torch.optim.Optimizer
        The parameter's optimizer method
    scheduler : torch.optim.lr_scheduler or None
        If provided, a learning rate scheduler for the optimizer
    args : Namespace
        The input arguments passed at running time
    forward_fun: function
        Function used to perform the feed-forward step

    Returns
    -------
    completed : bool
        Whether the training was sucessfully completed or it was interrupted
    """
    logger = logging.getLogger(args.mode + '_log')

    seg_model.train()
    completed = False
    keep_training = True

    best_valid_loss = float('inf')
    train_loss_history = []
    valid_loss_history = []

    step = 0
    if args.print_log:
        q = tqdm(total=stopping_criteria['early_stopping']._max_iterations,
                 desc="Training", position=0)
    while keep_training:
        # Reset the average loss computation every epoch
        sum_loss = 0

        for i, (x, t) in enumerate(train_data):
            step += 1

            # Start of training step
            optimizer.zero_grad()

            y = forward_fun(x)
            loss = criterion(y, t)
            loss = torch.mean(loss)
            loss.backward()

            optimizer.step()
            sum_loss += loss.item()

            if scheduler is not None and 'metrics' not in dict(signature(scheduler.step).parameters).keys():
                scheduler.step()
            # End of training step

            # Log the training performance every 10% of the training set
            if args.print_log:
                if i % 100 == 0:
                    t_flat = t[:, -1, ...].numpy().flatten()
                    y_flat = y.detach().cpu().numpy().flatten() > 0.5
                    acc = accuracy_score(t_flat, y_flat)
                    recall = recall_score(t_flat, y_flat, zero_division=0)
                    prec = precision_score(t_flat, y_flat, zero_division=0)
                    f1 = f1_score(t_flat, y_flat, zero_division=0)
                    q.set_description('Training Loss {:0.4f} ({:.4f}: acc={:0.4f}, prec={:0.4f}, recall={:0.4f}, f1={:0.4f}).'.format(
                        loss.item(), sum_loss / (i+1), acc, prec, recall, f1))
                q.update()
            elif i % max(1, int(0.1 * len(train_data))) == 0:
                t_flat = t[:, -1, ...].numpy().flatten()
                y_flat = y.detach().cpu().numpy().flatten() > 0.5
                acc = accuracy_score(t_flat, y_flat)
                recall = recall_score(t_flat, y_flat, zero_division=0)
                prec = precision_score(t_flat, y_flat, zero_division=0)
                f1 = f1_score(t_flat, y_flat, zero_division=0)
                logger.debug('\t[Step {:06d} {:04d}/{:04d}] Training Loss {:0.4f} ({:0.4f}: acc={:0.4f}, prec={:0.4f}, recall={:0.4f}, f1={:0.4f})'.format(
                    step, i, len(train_data), loss.item(), sum_loss / (i+1), acc, prec, recall, f1))

            # Checkpoint step
            keep_training = reduce(lambda sc1, sc2: sc1 & sc2, map(lambda sc: sc[1].check(), stopping_criteria.items()), True)

            if not keep_training or step % args.checkpoint_steps == 0:
                train_loss = sum_loss / (i+1)

                # Evaluate the model with the validation set
                valid_loss = valid(seg_model, valid_data, criterion, logger, args, forward_fun=forward_fun)

                seg_model.train()

                stopping_info = ';'.join(map(lambda sc: sc.__repr__(), stopping_criteria))

                # If there is a learning rate scheduler, perform a step
                # Log the overall network performance every checkpoint step
                logger.info('[Step {:06d} ({})] Training loss {:0.4f}, validation loss {:0.4f}, best validation loss {:0.4f}, learning rate {:e}, stopping criteria: {}'.format(
                    step, 'training' if keep_training else 'stopping', train_loss, valid_loss, best_valid_loss, optimizer.param_groups[0]['lr'], stopping_info)
                )

                train_loss_history.append(train_loss)
                valid_loss_history.append(valid_loss)

                # Save the current training state in a checkpoint file
                best_valid_loss = utils.checkpoint(step, seg_model, optimizer, scheduler, best_valid_loss, train_loss_history, valid_loss_history, args)

                stopping_criteria['early_stopping'].update(iteration=step, metric=valid_loss)
            else:
                stopping_criteria['early_stopping'].update(iteration=step)

            if not keep_training:
                logging.info('\n**** Stopping criteria met: Interrupting training ****')
                break

    else:
        completed = True

    if args.print_log:
        q.close()

    # Return True if the training finished sucessfully
    return completed


def setup_criteria(args):
    """Setup a loss function for the neural network optimization, and training
    stopping criteria.

    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. All the parameters are passed directly to the criteria constructors.

    Returns
    -------
    criterion : nn.Module
        The loss function that is used as target to optimize the parameters of the nerual network.

    stopping_criteria : list[StoppingCriterion]
        A list of stopping criteria. The first element is always set to stop the training after a fixed number of iterations.
        Depending on the criterion used, additional stopping criteria is set.        
    """

    # Early stopping criterion:
    stopping_criteria = {
        'early_stopping': models.EarlyStoppingPatience(
            max_iterations=args.steps, **args.__dict__)
    }

    # Loss function
    if args.criterion == 'CE':
        if args.classes == 1:
            criterion = nn.BCEWithLogitsLoss(reduction='none',
                                             pos_weight=torch.FloatTensor([args.pos_weight]))
        else:
            criterion = models.CrossEnropy2D()
    elif args.criterion == 'CEW':
        criterion = models.CrossEnropyDistance(**args.__dict__)

    else:
        raise ValueError('Criterion \'%s\' not supported' % args.criterion)

    criterion = nn.DataParallel(criterion)
    if args.gpu:
        criterion.cuda()

    return criterion, stopping_criteria


def setup_optim(seg_model, args):
    """Setup a loss function for the neural network optimization, and training
    stopping criteria.

    Parameters
    ----------
    cae_model : torch.nn.Module
        The convolutional autoencoder model to be optimized
    scheduler_type : str
        The type of learning rate scheduler used during the neural network training

    Returns
    -------
    optimizer : torch.optim.Optimizer
        The neurla network optimizer method
    scheduler : torch.optim.lr_scheduler
        The learning rate scheduler for the optimizer
    """

    # By now, only the ADAM optimizer is used
    optimizer = optim.Adam(params=seg_model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    # Only the the reduce on plateau, or none at all scheduler are used
    if args.scheduler_type == 'None':
        scheduler = None
    elif args.scheduler_type in scheduler_options.keys():
        scheduler = scheduler_options[args.scheduler_type](optimizer=optimizer)
    else:
        raise ValueError('Scheduler \"%s\" is not implemented' % args.scheduler_type)

    return optimizer, scheduler


def resume_checkpoint(seg_model, optimizer, scheduler, checkpoint, gpu=True):
    """Resume training from a previous checkpoint

    Parameters
    ----------
    seg_model : torch.nn.Module
        The convolutional autoencoder model to be optimized
    optimizer : torch.optim.Optimizer
        The neurla network optimizer method
    scheduler : torch.optim.lr_scheduler or None
        The learning rate scheduler for the optimizer
    checkpoint : str
        Path to a previous training checkpoint
    gpu : bool
        Wether use GPUs to train the neural network or not    
    """

    if not gpu:
        checkpoint_state = torch.load(checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint_state = torch.load(checkpoint)

    seg_model.module.load_state_dict(checkpoint_state['model'])

    optimizer.load_state_dict(checkpoint_state['optimizer'])

    if scheduler is not None and checkpoint_state['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint_state['scheduler'])


def main(args):
    """Set up the training environment

    Parameters
    ----------
    args : dict or Namespace
        The set of parameters passed to the different constructors to set up the convolutional autoencoder training
    """
    logger = logging.getLogger(args.mode + '_log')

    (seg_model,
     forward_fun,
     args.compressed_input) = segment.setup_network(
        args.__dict__,
        pretrained_model=None,
        autoencoder_model=args.autoencoder_model,
        use_gpu=args.gpu)

    criterion, stopping_criteria = setup_criteria(args)
    optimizer, scheduler = setup_optim(seg_model, args)

    if args.resume is not None:
        resume_checkpoint(seg_model, optimizer, scheduler, args.resume, gpu=args.gpu)

    # Log the training setup
    logger.info('Network architecture:')
    logger.info(seg_model)

    logger.info('\nCriterion:')
    logger.info(criterion)

    logger.info('\nStopping criterion:')
    for k in stopping_criteria.keys():
        logger.info(stopping_criteria[k])

    logger.info('\nOptimization parameters:')
    logger.info(optimizer)

    logger.info('\nScheduler parameters:')
    logger.info(scheduler)

    train_data, valid_data = utils.get_data(args)

    train(seg_model, train_data, valid_data, criterion, stopping_criteria,
          optimizer,
          scheduler,
          args,
          forward_fun)


if __name__ == '__main__':
    args = utils.get_args(task='segmentation', mode='training')

    utils.setup_logger(args)

    main(args)

    logging.shutdown()
