import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import EarlyStoppingPatience, UNet, UNetNoBridge, DecoderUNet, Synthesizer
from utils import checkpoint, get_training_args, setup_logger, get_data, load_state

from functools import reduce
from inspect import signature

scheduler_options = {"ReduceOnPlateau": partial(optim.lr_scheduler.ReduceLROnPlateau, mode='min', patience=2)}
seg_model_types = {"UNetNoBridge": UNetNoBridge, "UNet": UNet, "DecoderUNet": DecoderUNet}


# Variation of the forward step can be implemented here and used with 'partial' to be used inside training and validation steps.
def forward_undecoded_step(x, seg_model, decoder_model=None):
    y = seg_model(x)
    return y


def forward_decoded_step(x, seg_model, decoder_model=None):
    with torch.no_grad():
        x_brg = decoder_model.inflate(x, color=False)
        
    y = seg_model(x, x_brg[:0:-1])

    return y


def forward_parallel_decoded_step(x, seg_model, decoder_model=None):
    with torch.no_grad():
        x_brg = decoder_model.module.inflate(x, color=False)
        
    y = seg_model(x, x_brg[:0:-1])

    return y


def valid(seg_model, data, criterion, logger, forward_fun=None):
    """ Validation step.
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
    forward_fun: function
        Function used to perform the feed-forward step
    
    Returns
    -------
    mean_loss : float
        Mean value of the criterion function over the full set of validation elements
    """

    seg_model.eval()
    sum_loss = 0

    with torch.no_grad():
        for i, (x, t) in enumerate(data):
            y = forward_fun(x)
            
            loss = criterion(y, t)
            # In case that distributed computation of the criterion ouptuts a vector instead of a scalar
            loss = torch.mean(loss)
            sum_loss += loss.item()

            if i % max(1, int(0.1 * len(data))) == 0:
                logger.debug('\t[{:04d}/{:04d}] Validation Loss {:.4f} ({:.4f})'.format(i, len(data), loss.item(), sum_loss / (i+1)))

    mean_loss = sum_loss / len(data)

    return mean_loss


def train(seg_model, train_data, valid_data, criterion, stopping_criteria, optimizer, scheduler, args, forward_fun=None):
    """ Training loop by steps.
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

    completed = False
    keep_training = True

    best_valid_loss = float('inf')
    train_loss_history = []
    valid_loss_history = []

    step = 0
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
            if i % max(1, int(0.1 * len(train_data))) == 0:
                logger.debug('\t[Step {:06d} {:04d}/{:04d}] Training Loss {:.4f} ({:.4f})'.format(step, i, len(train_data), loss.item(), sum_loss / (i+1)))

            # Checkpoint step
            keep_training = reduce(lambda sc1, sc2: sc1 & sc2, map(lambda sc: sc.check(), stopping_criteria), True)

            if not keep_training or step % args.checkpoint_steps == 0:
                train_loss = sum_loss / (i+1)

                # Evaluate the model with the validation set
                valid_loss = valid(seg_model, valid_data, criterion, logger, forward_fun=forward_fun)
                
                seg_model.train()

                stopping_info = ';'.join(map(lambda sc: sc.__repr__(), stopping_criteria))

                # If there is a learning rate scheduler, perform a step
                # Log the overall network performance every checkpoint step
                logger.info('[Step {:06d} ({})] Training loss {:0.4f}, validation loss {:.4f}, best validation loss {:.4f}, learning rate {:e}, stopping criteria: {}'.format(
                    step, 'training' if keep_training else 'stopping', train_loss, valid_loss, best_valid_loss, optimizer.param_groups[0]['lr'], stopping_info)
                )

                train_loss_history.append(train_loss)
                valid_loss_history.append(valid_loss)

                # Save the current training state in a checkpoint file
                best_valid_loss = checkpoint(step, seg_model, optimizer, scheduler, best_valid_loss, train_loss_history, valid_loss_history, args)

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
    """ Setup a nerual network for object segmentation.

    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. All the parameters are passed directly to the model constructor.
        This way, the constructor can take the parameters needed that have been passed by the user.
    
    Returns
    -------
    seg_model : nn.Module
        The segmentation mode implemented by a convolutional neural network
    
    forward_function : function
        The function to be used as feed-forward step
    """
    # When the model works on compressed representation, tell the dataloader to obtain the compressed input and normal size target
    if 'Decoder' in args.model_type:
        args.compressed_input = True

    # If a decoder model is passed as argument, use the decoded step version of the feed-forward step
    if args.autoencoder_model is not None:
        if not args.gpu:
            checkpoint_state = torch.load(args.autoencoder_model, map_location=torch.device('cpu'))
        
        else:
            checkpoint_state = torch.load(args.autoencoder_model)
       
        decoder_model = Synthesizer(**checkpoint_state['args'])
        decoder_model.load_state_dict(checkpoint_state['decoder'])

        if args.gpu:
            decoder_model = nn.DataParallel(decoder_model)
            decoder_model.cuda()

        decoder_model.eval()
        args.use_bridge = True
    else:
        args.use_bridge = False
    
    seg_model_class = seg_model_types.get(args.model_type, None)
    if seg_model_class is None:
        raise ValueError('Model type %s not supported' % args.model_type)
    
    seg_model = seg_model_class(**args.__dict__)

    # If there are more than one GPU, DataParallel handles automatically the distribution of the work
    seg_model = nn.DataParallel(seg_model)
    if args.gpu:
        seg_model.cuda()

    # Define what funtion use in the feed-forward step
    if args.autoencoder_model is not None:
        if args.gpu:
            forward_function = partial(forward_parallel_decoded_step, seg_model=seg_model, decoder_model=decoder_model)
        else:
            forward_function = partial(forward_decoded_step, seg_model=seg_model, decoder_model=decoder_model)

    else:
        if 'Decoder' in args.model_type:
            # If no decoder is loaded, use the inflate function inside the segmentation model
            if args.gpu:
                forward_function = partial(forward_parallel_decoded_step, seg_model=seg_model, decoder_model=seg_model)
            else:
                forward_function = partial(forward_decoded_step, seg_model=seg_model, decoder_model=seg_model)
        else:
            forward_function = partial(forward_undecoded_step, seg_model=seg_model, decoder_model=None)

    return seg_model, forward_function


def setup_criteria(args):
    """ Setup a loss function for the neural network optimization, and training stopping criteria.

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
    stopping_criteria = [EarlyStoppingPatience(max_iterations=args.steps, **args.__dict__)]

    # Loss function
    if args.criterion == 'CE':
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    else:
        raise ValueError('Criterion \'%s\' not supported' % args.criterion)

    criterion = nn.DataParallel(criterion)
    if args.gpu:
        criterion.cuda()
        
    return criterion, stopping_criteria


def setup_optim(seg_model, scheduler_type='None'):
    """ Setup a loss function for the neural network optimization, and training stopping criteria.

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
    optimizer = optim.Adam(params=seg_model.parameters(), lr=args.learning_rate)
    
    # Only the the reduce on plateau, or none at all scheduler are used
    if scheduler_type == 'None':
        scheduler = None
    elif scheduler_type in scheduler_options.keys():
        scheduler = scheduler_options[scheduler_type](optimizer=optimizer)
    else:
        raise ValueError('Scheduler \"%s\" is not implemented' % scheduler_type)

    return optimizer, scheduler


def resume_checkpoint(seg_model, optimizer, scheduler, checkpoint, gpu=True):
    """ Resume training from a previous checkpoint

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
    """ Set up the training environment

    Parameters
    ----------
    args : dict or Namespace
        The set of parameters passed to the different constructors to set up the convolutional autoencoder training
    """
    logger = logging.getLogger(args.mode + '_log')

    seg_model, forward_function = setup_network(args)
    criterion, stopping_criteria = setup_criteria(args)
    optimizer, scheduler = setup_optim(seg_model, scheduler_type=args.scheduler)

    if args.resume is not None:
        resume_checkpoint(seg_model, optimizer, scheduler, args.resume, gpu=args.gpu)

    # Log the training setup
    logger.info('Network architecture:')
    logger.info(seg_model)
    
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
    
    train(seg_model, train_data, valid_data, criterion, stopping_criteria, optimizer, scheduler, args, forward_function)


if __name__ == '__main__':
    args = get_training_args(task='segmentation')

    setup_logger(args)

    main(args)
    
    logging.shutdown()