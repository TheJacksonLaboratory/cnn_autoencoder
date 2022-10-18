import logging
import sys

import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel
import torch.optim as optim
import math

import models
import utils

from inspect import signature

from tqdm import tqdm


optimization_algorithms = {"Adam": optim.Adam,
                           "SGD": optim.SGD}

scheduler_options = {"ReduceOnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
                     "StepLR": optim.lr_scheduler.StepLR}

model_options = {"AutoEncoder": models.AutoEncoder,
                 "MaskedAutoEncoder": models.MaskedAutoEncoder}


def forward_step_base(x, cae_model):
    x_r, y, p_y = cae_model(x)
    return x_r, y, p_y


def forward_step_pyramid(x, cae_model):
    x_r, y, p_y = cae_model.module.forward_steps(x)
    return x_r, y, p_y


def valid(forward_fun, cae_model, data, criterion, args):
    """ Validation step.
    Evaluates the performance of the network in its current state using the full set of validation elements.

    Parameters
    ----------
    forward_fun : function
        The function used to make the forward pass to the input batch
    cae_model : torch.nn.Module
        The network model in the current state
    data : torch.utils.data.DataLoader or list[tuple]
        The validation dataset. Because the target is recosntruct the input, the label associated is ignored
    criterion : function or torch.nn.Module
        The loss criterion to evaluate the network's performance
    args: Namespace
        The input arguments passed at running time

    Returns
    -------
    mean_loss : float
        Mean value of the criterion function over the full set of validation elements
    """
    logger = logging.getLogger(args.mode + '_log')

    cae_model.eval()
    sum_loss = 0

    if args.print_log:
        q = tqdm(total=len(data), desc='Validating', position=1, leave=None)
    with torch.no_grad():
        for i, (x, _) in enumerate(data):
            x_r, y, p_y = forward_fun(x, cae_model)

            loss, _ = criterion(x=x, y=y, x_r=x_r, p_y=p_y, net=cae_model)
            loss = torch.mean(loss)
            sum_loss += loss.item()

            dist, rate = criterion.compute_distortion(x=x, x_r=x_r, p_y=p_y)
            if not isinstance(dist, list):
                dist = [dist]

            if args.print_log:
                q.set_description('Validation Loss {:.4f} ({:.4f}: dist=[{}], rate:{:0.4f}).'.format(
                    loss.item(), sum_loss / (i+1), ','.join(['%0.4f' % d.item() for d in dist]), rate.item()))
                q.update()
            else:
                if i % max(1, int(0.1 * len(data))) == 0:
                    logger.debug('\t[{:04d}/{:04d}] Validation Loss {:.4f} ({:.4f}: dist=[{}], rate:{:0.4f}). Quantized compressed representation in [{:.4f}, {:.4f}], reconstruction in [{:.4f}, {:.4f}]'.format(
                    i, len(data), loss.item(), sum_loss / (i+1), ','.join(['%0.4f' % d.item() for d in dist]), rate.item(), y.detach().min(), y.detach().max(), x_r[0].detach().min(), x_r[0].detach().max()))

    if args.print_log:
        q.close()
    mean_loss = sum_loss / len(data)

    return mean_loss


def train(forward_fun, cae_model, train_data, valid_data, criterion, stopping_criteria, optimizer, scheduler, args):
    """Training loop by steps.
    This loop involves validation and network training checkpoint creation.

    Parameters
    ----------
    forward_fun : function
        The function used to make the forward pass to the input batch
    cae_model : torch.nn.Module
        The model to be trained
    train_data : torch.utils.data.DataLoader or list[tuple]
        The training data. Must contain the input and respective label; however, only the input is used because the target is reconstructing the input
    valid_data : torch.utils.data.DataLoader or list[tuple]
        The validation data.
    criterion : function or torch.nn.Module
        The loss criterion to evaluate the network's performance
    stopping_criteria : dict
        Stopping criteria tracker for different problem statements
    optimizer : torch.optim.Optimizer
        The parameter's optimizer method
    scheduler : torch.optim.lr_scheduler or None
        If provided, a learning rate scheduler for the optimizer
    args : Namespace
        The input arguments passed at running time

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
    if args.print_log:
        q = tqdm(total=stopping_criteria['early_stopping']._max_iterations,
                 desc="Training", position=0)
    while keep_training:
        # Reset the average loss computation every epoch
        sum_loss = 0
        for i, (x, _) in enumerate(train_data):
            step += 1
            q_penalty = None
            if 'penalty' in stopping_criteria.keys():
                stopping_criteria['penalty'].reset()

            while True:
                # Start of training step
                optimizer.zero_grad()

                x_r, y, p_y = forward_fun(x, cae_model)

                synthesizer = DataParallel(cae_model.module.synthesis)
                if args.gpu:
                    synthesizer.cuda()

                loss, extra_info = criterion(x=x, y=y, x_r=x_r, p_y=p_y, net=cae_model)
                if extra_info is not None:
                    extra_info = torch.mean(extra_info)
                loss = torch.mean(loss)
                loss.backward()

                # Clip the gradients to prevent from exploding gradients problems
                nn.utils.clip_grad_norm_(cae_model.parameters(), max_norm=50.0)
                optimizer.step()
                step_loss = loss.item()

                # When training with penalty on the energy of the compression
                # representation, update the respective stopping criterion
                if 'penalty' in stopping_criteria.keys():
                    if args.print_log:
                        param_nan = False
                        for param_name, param in cae_model.named_parameters():
                            if (param.grad is not None
                               and (math.isnan(param.data.detach().norm(2))
                                    or math.isnan(param.grad.data.detach().norm(2)))):
                                print('Paramater set to nan\n', param_name, param.data.detach().min(), param.data.detach().max(), param.data.detach().std())
                                print('Paramater gradient\n', param_name, param.grad.detach().data.min(), param.grad.detach().data.max(), param.grad.detach().data.std())
                                param_nan = True

                        if param_nan:
                            print('Org', x.detach().min(), x.detach().max(), x.detach().std())
                            print('Rec', x_r.detach().min(), x_r.detach().max(), x_r.detach().std())
                            print('Comp', y.detach().min(), y.detach().max(), y.detach().std())
                            print('Prob', p_y.detach().min(), p_y.detach().max(), p_y.detach().std())
                            print('Loss', step_loss, extra_info, torch.sum(-torch.log2(p_y.detach())))
                            sys.exit()

                        if q_penalty is None:
                            q_penalty = tqdm(total=stopping_criteria['penalty']._max_iterations, position=2, leave=None)
                        q_penalty.set_description('Sub-iter loss=%0.4f, energy=%0.4f' % (step_loss, extra_info))
                        q_penalty.update()
                    stopping_criteria['penalty'].update(iteration=step,
                                                        metric=extra_info.item())

                    if not stopping_criteria['penalty'].check():
                        if args.print_log:
                            q_penalty.close()
                        break
                else:
                    break

            sum_loss += step_loss

            if (scheduler is not None
               and 'metrics' not in signature(scheduler.step).parameters):
                scheduler.step()
            # End of training step

            dist, rate = criterion.compute_distortion(x=x, y=y, x_r=x_r, p_y=p_y)
            if not isinstance(dist, list):
                dist = [dist]
                x_r = [x_r]

            if args.print_log:
                q.set_description('Training Loss {:.4f} ({:.4f}: dist=[{}], rate={:.4f}). Quant bn [{:.4f}, {:.4f}], rec [{:.4f}, {:.4f}]'.format(
                    step_loss, sum_loss / (i+1), ','.join(['%0.4f' % d.item() for d in dist]), rate.item(), y.detach().min(), y.detach().max(), x_r[0].detach().min(), x_r[0].detach().max()))
                q.update()

            else:
                # Log the training performance every 10% of the training set
                if i % max(1, int(0.01 * len(train_data))) == 0:
                    logger.debug('\n\t[Step {:06d} {:04d}/{:04d}] Training Loss {:.4f} ({:.4f}: dist=[{}], rate={:.4f}). Quantized compressed representation in [{:.4f}, {:.4f}], reconstruction in [{:.4f}, {:.4f}]'.format(
                    step, i, len(train_data), loss.item(), sum_loss / (i+1), ','.join(['%0.4f' % d.item() for d in dist]), rate.item(), y.detach().min(), y.detach().max(), x_r[0].detach().min(), x_r[0].detach().max()))

            # Checkpoint step
            keep_training = stopping_criteria['early_stopping'].check()

            if not keep_training or (step >= args.early_warmup and (step-args.early_warmup) % args.checkpoint_steps == 0):
                train_loss = sum_loss / (i+1)

                # Evaluate the model with the validation set
                valid_loss = valid(forward_fun, cae_model, valid_data, criterion, args)

                cae_model.train()

                stopping_info = ';'.join(map(lambda k_sc: k_sc[0] + ": " + k_sc[1].__repr__(), stopping_criteria.items()))

                # If there is a learning rate scheduler, perform a step
                # Log the overall network performance every checkpoint step
                if not args.print_log:
                    logger.info('[Step {:06d} ({})] Training loss {:0.4f}, validation loss {:.4f}, best validation loss {:.4f}, learning rate {:e}, stopping criteria: {}'.format(
                        step, 'training' if keep_training else 'stopping', train_loss, valid_loss, best_valid_loss, optimizer.param_groups[0]['lr'], stopping_info)
                    )

                train_loss_history.append(train_loss)
                valid_loss_history.append(valid_loss)

                # Save the current training state in a checkpoint file
                best_valid_loss = utils.checkpoint(step, cae_model, optimizer, scheduler, best_valid_loss, train_loss_history, valid_loss_history, args)

                stopping_criteria['early_stopping'].update(iteration=step,
                                                           metric=valid_loss)
            else:
                stopping_criteria['early_stopping'].update(iteration=step)

            if not keep_training:
                logging.info('\n**** Stopping criteria met: '
                             'Interrupting training ****')
                break
    else:
        completed = True

    if args.print_log:
        q.close()
    # Return True if the training finished sucessfully
    return completed


def setup_network(args):
    """Setup a nerual network for image compression/decompression.

    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. All the parameters are passed directly to the model constructor.
        This way, the constructor can take the parameters needed that have been passed by the user.

    Returns
    -------
    cae_model : nn.Module
        The convolutional neural network autoencoder model.
    """

    # The autoencoder model contains all the modules
    cae_model = model_options[args.model_type](**args.__dict__)

    # If there are more than one GPU, DataParallel handles automatically the distribution of the work
    cae_model = nn.DataParallel(cae_model)
    if args.gpu:
        cae_model.cuda()

    return cae_model


def setup_criteria(cae_model, args):
    """Setup a loss function for the neural network optimization, and training stopping criteria.

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
            max_iterations=args.steps,
            **args.__dict__)
    }

    # Loss function
    if args.criterion == 'RD_PA':
        forward_fun = forward_step_base
        criterion = models.RateDistortionPenaltyA(**args.__dict__)
        stopping_criteria['penalty'] = \
            models.EarlyStoppingTarget(comparison='le',
                                       max_iterations=100,
                                       target=args.energy_limit,
                                       **args.__dict__)

    elif args.criterion == 'RD_PB':
        forward_fun = forward_step_base
        criterion = models.RateDistortionPenaltyB(**args.__dict__)
        stopping_criteria['penalty'] = \
            models.EarlyStoppingTarget(comparison='ge',
                                       max_iterations=100,
                                       target=args.energy_limit,
                                       **args.__dict__)

    elif args.criterion == 'RMS-SSIM_PA':
        forward_fun = forward_step_base
        criterion = models.RateMSSSIMPenaltyA(**args.__dict__)
        stopping_criteria['penalty'] = \
            models.EarlyStoppingTarget(comparison='le',
                                       max_iterations=100,
                                       target=args.energy_limit,
                                       **args.__dict__)

    elif args.criterion == 'RMS-SSIM_PB':
        forward_fun = forward_step_base
        criterion = models.RateMSSSIMPenaltyB(**args.__dict__)
        stopping_criteria['penalty'] = \
            models.EarlyStoppingTarget(comparison='ge',
                                       max_iterations=100,
                                       target=args.energy_limit,
                                       **args.__dict__)

    elif args.criterion == 'RD':
        forward_fun = forward_step_base
        criterion = models.RateDistortion(**args.__dict__)

    elif args.criterion == 'RMS-SSIM':
        forward_fun = forward_step_base
        criterion = models.MultiScaleSSIM(**args.__dict__)

    elif args.criterion == 'RD_MS_PA':
        forward_fun = forward_step_pyramid
        criterion = models.RateDistortionPyramidPenaltyA(**args.__dict__)
        stopping_criteria['penalty'] = \
            models.EarlyStoppingTarget(comparison='le',
                                       max_iterations=100,
                                       target=args.energy_limit,
                                       **args.__dict__)

    elif args.criterion == 'RD_MS_PB':
        forward_fun = forward_step_pyramid
        criterion = models.RateDistortionPyramidPenaltyB(**args.__dict__)
        stopping_criteria['penalty'] = \
            models.EarlyStoppingTarget(comparison='ge',
                                       max_iterations=100,
                                       target=args.energy_limit,
                                       **args.__dict__)

    elif args.criterion == 'RMS-SSIM_MS_PA':
        forward_fun = forward_step_pyramid
        criterion = models.RateMSSSIMPyramidPenaltyA(**args.__dict__)
        stopping_criteria['penalty'] = \
            models.EarlyStoppingTarget(comparison='le',
                                       max_iterations=100,
                                       target=args.energy_limit,
                                       **args.__dict__)

    elif args.criterion == 'RMS-SSIM_MS_PB':
        forward_fun = forward_step_pyramid
        criterion = models.RateMSSSIMPyramidPenaltyB(**args.__dict__)
        stopping_criteria['penalty'] = \
            models.EarlyStoppingTarget(comparison='ge',
                                       max_iterations=100,
                                       target=args.energy_limit,
                                       **args.__dict__)

    elif args.criterion == 'RD_MS':
        forward_fun = forward_step_pyramid
        criterion = models.RateDistortionPyramid(**args.__dict__)

    elif args.criterion == 'RMS-SSIM_MS':
        forward_fun = forward_step_pyramid
        criterion = models.MultiScaleSSIMPyramid(**args.__dict__)

    else:
        raise ValueError('Criterion \'%s\' not supported' % args.criterion)

    # criterion = nn.DataParallel(criterion)
    # if args.gpu:
    #    criterion = criterion.cuda()

    return forward_fun, criterion, stopping_criteria


def setup_optim(cae_model, args):
    """Setup a loss function for the neural network optimization, and training
    stopping criteria.

    Parameters
    ----------
    cae_model : torch.nn.Module
        The convolutional autoencoder model to be optimized
    scheduler_type : str
        The type of learning rate scheduler used during the neural network
        training

    Returns
    -------
    optimizer : torch.optim.Optimizer
        The neurla network optimizer method
    scheduler : torch.optim.lr_scheduler
        The learning rate scheduler for the optimizer
    """

    # By now, only the ADAM optimizer is used
    optim_algo = optimization_algorithms[args.optim_algo]
    optimizer = optim_algo(params=cae_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Only the the reduce on plateau, or none at all scheduler are used
    if args.scheduler_type == 'None':
        scheduler = None
    elif args.scheduler_type in scheduler_options.keys():
        scheduler = scheduler_options[args.scheduler_type](optimizer=optimizer, mode='min', patience=2)
    else:
        raise ValueError('Scheduler \"%s\" is not implemented' % args.scheduler_type)

    return optimizer, scheduler


def resume_checkpoint(cae_model, optimizer, scheduler, checkpoint, gpu=True):
    """Resume training from a previous checkpoint

    Parameters
    ----------
    cae_model : torch.nn.Module
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

    cae_model.module.embedding.load_state_dict(checkpoint_state['embedding'])
    cae_model.module.analysis.load_state_dict(checkpoint_state['encoder'])
    cae_model.module.synthesis.load_state_dict(checkpoint_state['decoder'])
    cae_model.module.fact_entropy.load_state_dict(checkpoint_state['fact_ent'])

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

    cae_model = setup_network(args)
    forward_fun, criterion, stopping_criteria = setup_criteria(cae_model, args)
    optimizer, scheduler = setup_optim(cae_model, args)

    if args.resume is not None:
        resume_checkpoint(cae_model, optimizer, scheduler, args.resume, gpu=args.gpu)

    # Log the training setup
    logger.info('Network architecture:')
    logger.info(cae_model)

    logger.info('\nCriterion:')
    logger.info(criterion)

    logger.info('\nStopping criterion:')
    for k, crit in stopping_criteria.items():
        logger.info('\n' + k)
        logger.info(crit)

    logger.info('\nOptimization parameters:')
    logger.info(optimizer)

    logger.info('\nScheduler parameters:')
    logger.info(scheduler)

    train_data, valid_data = utils.get_data(args)

    train(forward_fun, cae_model, train_data, valid_data, criterion, stopping_criteria, optimizer, scheduler, args)


if __name__ == '__main__':
    args = utils.get_args(task='autoencoder', mode='training')

    utils.setup_logger(args)

    main(args)

    logging.shutdown()
