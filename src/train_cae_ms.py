import logging

import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel
import torch.optim as optim

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

            synthesizer = DataParallel(cae_model.module.synthesis)
            if args.gpu:
                synthesizer.cuda()

            entropy_model = DataParallel(cae_model.module.fact_entropy)
            if args.gpu:
                entropy_model.cuda()

            loss_dict = criterion(x=x, y=y, x_r=x_r, p_y=p_y, net=synthesizer,
                                entropy_model=entropy_model)
            loss = torch.mean(loss_dict['dist_rate_loss'])
            sum_loss += loss.item()
            
            aux_loss = torch.mean(loss_dict['entropy_loss'])

            dist = criterion.compute_dist(x=x, x_r=x_r)
            rate = criterion.compute_rate(x=x, p_y=p_y)
            if not isinstance(dist, list):
                dist = [dist]

            if args.print_log:
                q.set_description(
                    'Validation Loss {:.4f} (dist=[{}], rate={:.2f}, '
                    'aux={:.2f}, energy={:.3f}). Quant bn [{:.2f}, {:.2f}] '
                    '({:.2f}, {:.2f}, {:.2f}), '
                    'rec [{:.2f}, {:.2f}]'.format(
                        sum_loss / (i+1),
                        ','.join(['%0.4f' % d.item() for d in dist]),
                        rate.item(),
                        aux_loss.item(),
                        loss_dict.get('energy', 0.0),
                        y.detach().min(),
                        y.detach().max(),
                        *entropy_model.module.tails.detach().mean(dim=(0, 1, 3)),
                        x_r[0].detach().min(),
                        x_r[0].detach().max()))
                q.update()
            elif i % max(1, int(0.1 * len(data))) == 0:
                logger.debug(
                    '\t[{:04d}/{:04d}] Validation Loss {:.4f} (dist=[{}], '
                    'rate={:.2f}, aux={:.2f}, energy={:.3f}). '
                    'Quant bn [{:.2f}, {:.2f}] '
                    '({:.2f}, {:.2f}, {:.2f}), '
                    'rec [{:.2f}, {:.2f}]'.format(
                        i,
                        len(data),
                        sum_loss / (i+1),
                        ','.join(['%0.4f' % d.item() for d in dist]),
                        rate.item(),
                        aux_loss.item(),
                        loss_dict.get('energy', 0.0),
                        y.detach().min(),
                        y.detach().max(),
                        *entropy_model.module.tails.detach().mean(dim=(0, 1, 3)),
                        x_r[0].detach().min(),
                        x_r[0].detach().max()))

    if args.print_log:
        q.close()
    mean_loss = sum_loss / len(data)

    return mean_loss


def train(forward_fun, cae_model, train_data, valid_data, criterion, stopping_criteria, optimizer, aux_optimizer, scheduler, args):
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
    aux_optimizer : torch.optim.Optimizer
        The entropy model symbols range optimizer method
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
                aux_optimizer.zero_grad()

                x_r, y, p_y = forward_fun(x, cae_model)

                synthesizer = DataParallel(cae_model.module.synthesis)
                if args.gpu:
                    synthesizer.cuda()

                entropy_model = DataParallel(cae_model.module.fact_entropy)
                if args.gpu:
                    entropy_model.cuda()

                loss_dict = criterion(x=x, y=y, x_r=x_r, p_y=p_y,
                                      net=synthesizer,
                                      entropy_model=entropy_model)
                if 'energy' in loss_dict:
                    extra_info = torch.mean(loss_dict['energy'])
                loss = torch.mean(loss_dict['dist_rate_loss'])
                loss.backward()

                # Clip the gradients to prevent from exploding gradients problems
                nn.utils.clip_grad_norm_(cae_model.parameters(), max_norm=1.0)
                optimizer.step()
                step_loss = loss.item()

                aux_loss = torch.mean(loss_dict['entropy_loss'])
                aux_loss.backward()
                aux_optimizer.step()

                # When training with penalty on the energy of the compression
                # representation, update the respective stopping criterion
                if 'penalty' in stopping_criteria.keys():
                    if args.print_log:
                        if q_penalty is None:
                            q_penalty = tqdm(total=stopping_criteria['penalty']._max_iterations, position=2, leave=None)

                        with torch.no_grad():
                            dist = criterion.compute_dist(x=x, x_r=x_r)
                            rate = criterion.compute_rate(x=x, p_y=p_y)

                        if not isinstance(dist, list):
                            dist = [dist]

                        q_penalty.set_description(
                            'Sub-iter Loss {:.4f} (dist=[{}], rate={:.2f}, '
                            'aux={:.2f}, energy={:.3f}) '
                            'Quant bn [{:.2f}, {:.2f}] '
                            '({:.2f}, {:.2f}, {:.2f}), '
                            'rec [{:.2f}, {:.2f}]'.format(
                                step_loss,
                                ','.join(['%0.4f' % d.item() for d in dist]),
                                rate.item(),
                                aux_loss.item(),
                                loss_dict.get('energy', 0.0),
                                y.detach().min(),
                                y.detach().max(),
                                *entropy_model.module.tails.detach().mean(dim=(0, 1, 3)),
                                x_r[0].detach().min(),
                                x_r[0].detach().max()))

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

            with torch.no_grad():
                dist = criterion.compute_dist(x=x, x_r=x_r)
                rate = criterion.compute_rate(x=x, p_y=p_y)

            if not isinstance(dist, list):
                dist = [dist]
                x_r = [x_r]

            if args.print_log:
                q.set_description(
                    'Training Loss {:.4f} (dist=[{}], rate={:.2f}, aux={:.2f},'
                    ' energy={:3f}). Quant bn [{:.2f}, {:.2f}] '
                    '({:.2f}, {:.2f}, {:.2f}), '
                    'rec [{:.2f}, {:.2f}]'.format(
                        sum_loss / (i+1),
                        ','.join(['%0.4f' % d.item() for d in dist]),
                        rate.item(),
                        aux_loss.item(),
                        loss_dict.get('energy', 0.0),
                        y.detach().min(),
                        y.detach().max(),
                        *entropy_model.module.tails.detach().mean(dim=(0, 1, 3)),
                        x_r[0].detach().min(),
                        x_r[0].detach().max()))
                q.update()

            else:
                # Log the training performance every 10% of the training set
                if i % max(1, int(0.01 * len(train_data))) == 0:
                    logger.debug(
                        '\n\t[Step {:06d} {:04d}/{:04d}] Training Loss {:.4f} '
                        '(dist=[{}], rate={:.2f}, aux={:.2f}'
                        'energy={:3f} ). Quant bn [{:.2f}, {:.2f}] '
                        '({:.2f}, {:.2f}, {:.2f}), '
                        'rec [{:.2f}, {:.2f}]'.format(
                            step, i,
                            sum_loss / (i+1),
                            ','.join(['%0.4f' % d.item() for d in dist]),
                            rate.item(),
                            aux_loss.item(),
                            loss_dict.get('energy', 0.0),
                            y.detach().min(),
                            y.detach().max(),
                            *entropy_model.module.tails.detach().mean(dim=(0, 1, 3)),
                            x_r[0].detach().min(),
                            x_r[0].detach().max()))

            # Checkpoint step
            keep_training = stopping_criteria['early_stopping'].check()

            if (not keep_training
              or (step >= args.early_warmup
                  and (step-args.early_warmup) % args.checkpoint_steps == 0)):
                train_loss = sum_loss / (i+1)

                # Evaluate the model with the validation set
                valid_loss = valid(forward_fun, cae_model, valid_data, criterion, args)

                cae_model.train()

                stopping_info = ';'.join(map(lambda k_sc:\
                                             k_sc[0] + ": " + k_sc[1].__repr__(),
                                             stopping_criteria.items()))

                # If there is a learning rate scheduler, perform a step
                # Log the overall network performance every checkpoint step
                if not args.print_log:
                    logger.info(
                        '[Step {:06d} ({})] Training loss {:0.4f}, validation '
                        'loss {:.4f}, best validation loss {:.4f}, learning '
                        'rate {:e}, stopping criteria: {}'.format(
                            step, 'training' if keep_training else 'stopping',
                            train_loss,
                            valid_loss,
                            best_valid_loss,
                            optimizer.param_groups[0]['lr'],
                            stopping_info)
                    )

                train_loss_history.append(train_loss)
                valid_loss_history.append(valid_loss)

                # Save the current training state in a checkpoint file
                best_valid_loss = utils.checkpoint(step, cae_model, optimizer,
                                                   scheduler,
                                                   best_valid_loss,
                                                   train_loss_history,
                                                   valid_loss_history,
                                                   args)

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
        The input arguments passed at running time. All the parameters are
        passed directly to the model constructor. This way, the constructor can
        take the parameters needed that have been passed by the user.

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


def setup_criteria(args):
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
    args_dict = args.__dict__

    # Early stopping criterion:
    stopping_criteria = {
        'early_stopping': models.EarlyStoppingPatience(
            max_iterations=args.steps,
            **args_dict)
    }

    criterion_name = ''

    args_dict['max_iterations'] = args.sub_iter_steps
    args_dict['target'] = args_dict['energy_limit']
    if 'PA' in args_dict['criterion']:
        args_dict['comparison'] = 'le'
        stopping_criteria['penalty'] = models.EarlyStoppingTarget(**args_dict)
        criterion_name = 'PenaltyA'

    elif 'PB' in args_dict['criterion']:
        args_dict['comparison'] = 'ge'
        stopping_criteria['penalty'] = models.EarlyStoppingTarget(**args_dict)
        criterion_name = 'PenaltyB'

    if 'RD' in args_dict['criterion']:
        criterion_name = 'RateMSE' + criterion_name
    elif 'RMS-SSIM' in args_dict['criterion']:
        criterion_name = 'RateMSSSIM' + criterion_name

    # Loss function
    if 'Multiscale' in args_dict['criterion']:
        forward_fun = forward_step_pyramid
        criterion_name = 'Multiscale' + criterion_name
    else:
        forward_fun = forward_step_base

    if criterion_name not in models.LOSS_LIST:
        raise ValueError(
            'Criterion \'%s\' not supported' % args_dict['criterion'])

    criterion = models.LOSS_LIST[criterion_name](**args_dict)

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
        The optimizer for the neural network
    aux_optimizer : torch.optim.Optimizer
        The optimizer for the entropy model range of symbols
    scheduler : torch.optim.lr_scheduler
        The learning rate scheduler for the optimizer
    """

    # By now, only the ADAM optimizer is used
    optim_algo = optimization_algorithms[args.optim_algo]

    net_pars = []
    aux_pars = []
    for par_name, par in cae_model.named_parameters():
        if 'tails' in par_name:
            aux_pars.append(par)
        else:
            net_pars.append(par)

    optimizer = optim_algo(params=net_pars,
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    aux_optimizer = optim_algo(params=aux_pars,
                               lr=args.aux_learning_rate,
                               weight_decay=args.aux_weight_decay)

    # Only the the reduce on plateau, or none at all scheduler are used
    if args.scheduler_type == 'None':
        scheduler = None
    elif args.scheduler_type in scheduler_options.keys():
        scheduler = scheduler_options[args.scheduler_type](optimizer=optimizer,
                                                           mode='min',
                                                           patience=2)
    else:
        raise ValueError('Scheduler \"%s\" is not implemented' % args.scheduler_type)

    return optimizer, aux_optimizer, scheduler


def resume_checkpoint(cae_model, optimizer, scheduler, checkpoint, gpu=True,
                      resume_optimizer=False):
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
    resume_optimizer : bool
        Wether use the optimizer from the checkpoint or not. This only works
        for resume training rather than starting from pre-trained models
    """

    if not gpu:
        checkpoint_state = torch.load(checkpoint,
                                      map_location=torch.device('cpu'))
    else:
        checkpoint_state = torch.load(checkpoint)

    cae_model.module.embedding.load_state_dict(checkpoint_state['embedding'])
    cae_model.module.analysis.load_state_dict(checkpoint_state['encoder'])
    cae_model.module.synthesis.load_state_dict(checkpoint_state['decoder'],
                                               strict=False)
    if checkpoint_state['args']['version'] == '0.5.5':
        for color_layer in cae_model.module.synthesis.color_layers:
            color_layer[0].weight.data.copy_(
                checkpoint_state['decoder']['synthesis_track.4.weight'])
    cae_model.module.fact_entropy.load_state_dict(checkpoint_state['fact_ent'])

    if resume_optimizer:
        optimizer.load_state_dict(checkpoint_state['optimizer'])

        if scheduler is not None and checkpoint_state['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint_state['scheduler'])


def main(args):
    """Set up the training environment

    Parameters
    ----------
    args : dict or Namespace
        The set of parameters passed to the different constructors to set up
        the convolutional autoencoder training.
    """
    logger = logging.getLogger(args.mode + '_log')

    cae_model = setup_network(args)
    forward_fun, criterion, stopping_criteria = setup_criteria(args)
    optimizer, aux_optimizer, scheduler = setup_optim(cae_model, args)

    if args.resume is not None:
        resume_checkpoint(cae_model, optimizer, scheduler, args.resume,
                          gpu=args.gpu)

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

    train(forward_fun, cae_model, train_data, valid_data, criterion,
          stopping_criteria,
          optimizer,
          aux_optimizer,
          scheduler,
          args)


if __name__ == '__main__':
    args = utils.get_args(task='autoencoder', mode='training')

    utils.setup_logger(args)

    main(args)

    logging.shutdown()
