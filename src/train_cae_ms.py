import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import models
import utils

from inspect import signature
from itertools import chain

from tqdm import tqdm


optimization_algorithms = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SGD": optim.SGD}

scheduler_algorithms = {
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "StepLR": optim.lr_scheduler.StepLR,
    "LinearLR": optim.lr_scheduler.LinearLR,
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR}


def valid(model, data, criterion, args):
    """ Validation step.
    Evaluates the performance of the network in its current state using the
    full set of validation elements.

    Parameters
    ----------
    model : dict
        The network model in the current state
    data : torch.utils.data.DataLoader or list[tuple]
        The validation dataset. Because the target is recosntruct the input,
        the label associated is ignored
    criterion : function or torch.nn.Module
        The loss criterion to evaluate the network's performance
    args: Namespace
        The input arguments passed at running time

    Returns
    -------
    mean_loss : float
        Mean value of the criterion function over the full set of validation
        elements
    """
    logger = logging.getLogger(args.mode + '_log')

    valid_forward_step = models.decorate_trainable_modules(
        trainable_modules=None,
        enabled_modules=args.enabled_modules)

    for k in model.keys():
        model[k].eval()

    sum_loss = 0
    rec_metrics = None
    channel_e_history = []

    if args.progress_bar:
        q = tqdm(total=len(data), desc='Validating', position=1, leave=None)

    for i, (x, t) in enumerate(data):
        output = valid_forward_step(x, model)
        t = t.to(output['y_q'].device)

        loss_dict = criterion(inputs=x, outputs=output, targets=t, net=model)
        loss = torch.mean(loss_dict['loss'])
        sum_loss += loss.item()

        channel_e_history.append(loss_dict['channel_e'])
        channel_e = int(torch.median(torch.LongTensor(channel_e_history)))

        if args.progress_bar:
            log_str, _ = utils.log_info(None, i + 1, None, model, x, t, output,
                                        sum_loss / (i + 1),
                                        loss_dict,
                                        channel_e,
                                        step_type='Validation',
                                        lr=None,
                                        progress_bar=True)
            q.set_description(log_str)
            q.update()

        if i % max(1, int(0.1 * len(data))) == 0:
            (log_str,
             curr_rec_metrics) = utils.log_info(None, i + 1, None, model, x, t,
                                                output,
                                                sum_loss / (i + 1),
                                                loss_dict,
                                                channel_e,
                                                step_type='Validation',
                                                lr=None,
                                                progress_bar=False)

            logger.debug(log_str)
            if rec_metrics is None:
                rec_metrics = dict((m, []) for m in curr_rec_metrics.keys())
            
            for m, v in curr_rec_metrics.items():
                rec_metrics[m].append(v)

    if args.progress_bar:
        q.close()

    avg_rec_metrics = {}
    for m, v in rec_metrics.items():
        avg_rec_metrics['val_' + m] = np.nanmean(v)

    mean_loss = sum_loss / len(data)

    return mean_loss, avg_rec_metrics


def train(model, train_data,
          valid_data,
          criterion,
          stopping_criteria,
          mod_optimizers,
          mod_schedulers,
          mod_grad_accumulate,
          args):
    """Training loop by steps.
    This loop involves validation and network training checkpoint creation.

    Parameters
    ----------
    model : dict
        The modules of the model to be trained as a dictionary
    train_data : torch.utils.data.DataLoader or list[tuple]
        The training data. Must contain the input and respective label;
        however, only the input is used because the target is reconstructing
        the input
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

    train_forward_step = models.decorate_trainable_modules(
        trainable_modules=args.trainable_modules,
        enabled_modules=args.enabled_modules)

    completed = False
    keep_training = True

    best_valid_loss = float('inf')
    train_loss_history = []
    valid_loss_history = []
    channel_e_history = []

    step = 0
    train_data_size = len(train_data)

    if args.progress_bar:
        q = tqdm(total=stopping_criteria['early_stopping']._max_iterations,
                 desc="Training", position=0)

    for k in model.keys():
        if k in args.trainable_modules:
            model[k].train()
        else:
            model[k].eval()

    for k, opt in mod_optimizers.items():
        # Accumulate gradients on different steps according to the network
        # module type.
        opt.zero_grad()

    rec_metrics = None
    extra_metrics = None
    while keep_training:
        # Reset the average loss computation every epoch
        sum_loss = 0
        for i, (x, t) in enumerate(train_data):
            step += 1

            if 'penalty' in stopping_criteria.keys():
                stopping_criteria['penalty'].reset()
                if args.progress_bar:
                    q_penalty = tqdm(
                        total=stopping_criteria['penalty']._max_iterations,
                        position=2,
                        leave=None)

            sub_step = 0
            sub_step_loss = 0
            while True:
                sub_step += 1
                # Start of training step

                output = train_forward_step(x, model)
                t = t.to(output['y_q'].device)

                loss_dict = criterion(inputs=x, outputs=output, targets=t,
                                      net=model)

                loss = torch.mean(loss_dict['loss'])
                loss.backward()

                if 'entropy_loss' in loss_dict:
                    aux_loss = torch.mean(loss_dict['entropy_loss'])
                    aux_loss.backward()

                for k, opt in mod_optimizers.items():
                    if step % mod_grad_accumulate[k] == 0:
                        # Clip the gradients to prevent the exploding gradients
                        # problem
                        nn.utils.clip_grad_norm_(opt.param_groups[0]['params'],
                                                 max_norm=1.0)

                        # Update each network's module by separate
                        opt.step()
                        opt.zero_grad()

                step_loss = loss.item()
                sub_step_loss += step_loss
                channel_e_history.append(loss_dict.get('channel_e', -1))

                # When training with penalty on the energy of the compression
                # representation, update the respective stopping criterion
                if 'penalty' in stopping_criteria.keys():
                    if args.progress_bar:
                        channel_e = int(torch.median(torch.LongTensor(channel_e_history)))
                        log_str, _ = utils.log_info(sub_step, None, model, x,
                                                    t,
                                                    output,
                                                    sub_step_loss / sub_step,
                                                    loss_dict,
                                                    channel_e,
                                                    step_type='Sub-iter',
                                                    lr=None,
                                                    progress_bar=True)

                        q_penalty.set_description(log_str)
                        q_penalty.update()

                    stopping_criteria['penalty'].update(iteration=sub_step,
                                                        metric=torch.mean(loss_dict['energy']).item())

                    if not stopping_criteria['penalty'].check():
                        if args.progress_bar:
                            q_penalty.close()
                        break
                else:
                    break

            sum_loss += sub_step_loss / sub_step

            # End of training step
            if args.progress_bar:
                current_lr = ''
                for k, sched in mod_schedulers.items():
                    if hasattr(sched, '_last_lr'):
                        current_lr += '{}={:.2e} '.format(k, sched._last_lr[0])
                    else:
                        current_lr += '{}=None '.format(k)

                channel_e = int(torch.median(torch.LongTensor((channel_e_history))))
                log_str, _ = utils.log_info(None, i + 1, None, model, x, t,
                                            output,
                                            sum_loss / (i + 1),
                                            loss_dict,
                                            channel_e,
                                            step_type='Training',
                                            lr=current_lr,
                                            progress_bar=True)

                q.set_description(log_str)
                q.update()

            # Log the training performance every 10% of the training set
            if i % max(1, int(0.01 * len(train_data))) == 0:
                current_lr = ''
                for k, sched in mod_schedulers.items():
                    if hasattr(sched, '_last_lr'):
                        current_lr += '{}={:.2e} '.format(k, sched._last_lr[0])
                    else:
                        current_lr += '{}=None '.format(k)

                channel_e = int(torch.median(torch.LongTensor(channel_e_history)))

                (log_str,
                 curr_rec_metrics) = utils.log_info(step, i + 1,
                                                    train_data_size,
                                                    model,
                                                    x,
                                                    t,
                                                    output,
                                                    sum_loss / (i + 1),
                                                    loss_dict,
                                                    channel_e,
                                                    step_type='Training',
                                                    lr=current_lr,
                                                    progress_bar=False)

                logger.debug(log_str)

                if rec_metrics is None:
                    rec_metrics = dict((m, [])
                                       for m in curr_rec_metrics.keys())
                
                for m, v in curr_rec_metrics.items():
                    rec_metrics[m].append(v)

            # Checkpoint step
            keep_training = stopping_criteria['early_stopping'].check()

            if (not keep_training
              or (step >= args.early_warmup
                  and (step-args.early_warmup) % args.checkpoint_steps == 0)
                  and step > 1):
                train_loss = sum_loss / (i+1)

                # Evaluate the model with the validation set
                (valid_loss,
                 val_avg_metrics) = valid(model, valid_data, criterion, args)

                # Update the learning rate of the trainable modules
                for k in args.trainable_modules:
                    model[k].train()
                    sched = mod_schedulers.get(k, None)
                    if sched is not None:
                        if 'metrics' in signature(sched.step).parameters:
                            sched.step(valid_loss)
                        else:
                            sched.step()

                    aux_sched = mod_schedulers.get(k + '_aux', None)
                    if aux_sched is not None:
                        if 'metrics' in signature(aux_sched.step).parameters:
                            aux_sched.step(valid_loss)
                        else:
                            aux_sched.step()

                # If there is a learning rate scheduler, perform a step
                # Log the overall network performance every checkpoint step
                stopping_info = ';'.join(map(lambda k_sc:\
                                             k_sc[0] + ": " + k_sc[1].__repr__(),
                                             stopping_criteria.items()))
                current_lr = ''
                trn_avg_metrics = {}
                for k, sched in mod_schedulers.items():
                    if hasattr(sched, '_last_lr'):
                        current_lr += '{}={:.2e} '.format(k, sched._last_lr[0])
                        trn_avg_metrics[k] = sched._last_lr[0]
                    else:
                        current_lr += '{}=None '.format(k)
                        trn_avg_metrics[k] = float('nan')

                train_loss_history.append(train_loss)
                valid_loss_history.append(valid_loss)

                # Compute the mean value of the metrics recorded every 10% of
                # the steps within each epoch.
                if rec_metrics is not None:
                    for m, v in rec_metrics.items():
                        trn_avg_metrics['trn_' + m] = np.nanmean(v)

                channel_e = int(torch.median(torch.LongTensor(channel_e_history)))
                if extra_metrics is None:
                    extra_metrics = {'channel_e': []}
                    for m in chain(trn_avg_metrics.keys(),
                                   val_avg_metrics.keys()):
                        extra_metrics[m] = []

                extra_metrics['channel_e'].append(channel_e)
                for m, v in chain(trn_avg_metrics.items(),
                                  val_avg_metrics.items()):
                    extra_metrics[m].append(v)

                # Save the current training state in a checkpoint file
                best_valid_loss = utils.checkpoint(step, model, mod_optimizers,
                                                   mod_schedulers,
                                                   best_valid_loss,
                                                   train_loss_history,
                                                   valid_loss_history,
                                                   args,
                                                   extra_metrics)
                channel_e_history = []
                rec_metrics = None

                logger.info(
                    '[Step {:06d} ({})] Training loss {:0.4f}, validation '
                    'loss {:.4f}, best validation loss {:.4f}, learning '
                    'rate {}, stopping criteria: {}'.format(
                        step, 'training' if keep_training else 'stopping',
                        train_loss,
                        valid_loss,
                        best_valid_loss,
                        current_lr,
                        stopping_info)
                )

                # Update the state of the stopping criteria
                stopping_criteria['early_stopping'].update(iteration=step,
                                                           metric=valid_loss)
            else:
                stopping_criteria['early_stopping'].update(iteration=step)

            # Update the warming learning rate of the trainable modules
            if step <= args.early_warmup:
                for k in args.trainable_modules:
                    if step < mod_grad_accumulate[k]:
                        continue

                    sched = mod_schedulers.get(k + '_warmup', None)
                    if sched is not None:
                        sched.step()

                    aux_sched = mod_schedulers.get(k + '_aux_warmup', None)
                    if aux_sched is not None:
                        aux_sched.step()

            if not keep_training:
                logging.info('\n**** Stopping criteria met: '
                             'Interrupting training ****')
                break

    else:
        completed = True

    if args.progress_bar:
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
    if isinstance(args, dict):
        args_dict = args
    else:
        args_dict = args.__dict__

    args_dict['multiscale_analysis'] = 'Multiscale' in args_dict['criterion']

    if args.checkpoint is not None:
        checkpoint_state = torch.load(args.checkpoint, map_location='cpu')
        checkpoint_state.update(args_dict)
        args_dict = checkpoint_state

    model = models.autoencoder_from_state_dict(args_dict, gpu=args.gpu,
                                               train=True)

    if 'class_model' in args_dict['enabled_modules']:
        model['class_model'] = models.classifier_from_state_dict(args_dict,
                                                                 gpu=args.gpu,
                                                                 train=True)
    if 'seg_model' in args_dict['enabled_modules']:
        model['seg_model'] = models.segmenter_from_state_dict(args_dict,
                                                              gpu=args.gpu,
                                                              train=True)

    return model


def setup_criteria(args, checkpoint=None):
    """Setup a loss function for the neural network optimization, and training
    stopping criteria.

    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. All the parameters are
        passed directly to the criteria constructors.
    checkpoint: Path or None
        Path to a pretrained model. Only used when the Penalty B is active, 
        to extract the channel index with highest energy.
    Returns
    -------
    criterion : nn.Module
        The loss function that is used as target to optimize the parameters of
        the nerual network.

    stopping_criteria : list[StoppingCriterion]
        A list of stopping criteria. The first element is always set to stop
        the training after a fixed number of iterations.
        Depending on the criterion used, additional stopping criteria is set.
    """

    # Early stopping criterion:
    if 'PB' in args.criterion:
        args.channel_e = 0
        if checkpoint is not None:
            checkpoint_state = torch.load(checkpoint, map_location='cpu')
            args.channel_e = int(checkpoint_state.get('channel_e', 0))

    stopping_criteria = models.setup_stopping_criteria(**args.__dict__)

    criterion = models.setup_loss(**args.__dict__)

    return criterion, stopping_criteria


def setup_optim(model, args):
    """Setup a loss function for the neural network optimization, and training
    stopping criteria.

    Parameters
    ----------
    model : torch.nn.Module
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
    optim_algos = utils.parse_typed_arguments(args.mod_optim_algo)

    scheduler_algos = {}
    for mod_pars in args.mod_scheduler_algo:
        mod = mod_pars[:mod_pars.find('=')]
        sched_type_args = mod_pars[mod_pars.find('=') + 1:]
        sched_type = sched_type_args.split(',')[0]
        if sched_type.lower() == 'none':
            sched_type = None
        sched_args = sched_type_args.split(',')[1:]
        scheduler_algos[mod] = (sched_type,
                                utils.parse_typed_arguments(sched_args))

    # Parse the values of learning rate and weight decay for each module. These
    # must be passed in the form `--mod-lrate class_model=0.1`, e.g. to assign
    # an initial learning rate to the weights update of the `class` module of
    # the neural network.
    mod_grad_accumulate = utils.parse_typed_arguments(
        args.mod_grad_accumulate)

    mod_learning_rate = utils.parse_typed_arguments(
        args.mod_learning_rate)

    mod_weight_decay = utils.parse_typed_arguments(args.mod_weight_decay)

    mod_aux_learning_rate = utils.parse_typed_arguments(
        args.mod_aux_learning_rate)

    mod_aux_weight_decay = utils.parse_typed_arguments(
        args.mod_aux_weight_decay)

    mod_optimizers = {}
    mod_schedulers = {}
    for k in args.trainable_modules:
        if mod_grad_accumulate.get(k, None) is None:
            mod_grad_accumulate[k] = 1

        optim_aux_pars = {}
        optim_pars = {}
        pars = []
        aux_pars = []

        for par_name, par in model[k].named_parameters():
            if 'quantiles' in par_name.lower() or 'aux' in par_name.lower():
                aux_pars.append(par)
            else:
                pars.append(par)

        optim_pars['params'] = pars
        module_lr = mod_learning_rate.get(k, None)
        module_wd = mod_weight_decay.get(k, None)

        if module_lr is not None:
            optim_pars['lr'] = module_lr
            optim_pars['initial_lr'] = module_lr

        if module_wd is not None:
            optim_pars['weight_decay'] = module_wd

        mod_optimizers[k] = optimization_algorithms[optim_algos[k]](
            [optim_pars])

        (mod_scheduler_algo,
         mod_scheduler_args) = scheduler_algos.get(k, (None, None))

        if mod_scheduler_algo is not None:
            mod_schedulers[k] = scheduler_algorithms[mod_scheduler_algo](
                optimizer=mod_optimizers[k],
                **mod_scheduler_args)

        if len(aux_pars) > 0:
            mod_grad_accumulate[k + '_aux'] = mod_grad_accumulate[k]

            optim_aux_pars['params'] = aux_pars
            module_aux_lr = mod_aux_learning_rate.get(k, None)
            module_aux_wd = mod_aux_weight_decay.get(k, None)

            if module_aux_lr is not None:
                optim_aux_pars['lr'] = module_aux_lr
                optim_aux_pars['initial_lr'] = module_aux_lr

            if module_wd is not None:
                optim_aux_pars['weight_decay'] = module_aux_wd

            mod_optimizers[k + '_aux'] = \
                optimization_algorithms[optim_algos[k]]([optim_aux_pars])

            if mod_scheduler_algo is not None:
                mod_schedulers[k + '_aux'] = \
                    scheduler_algorithms[mod_scheduler_algo](
                        optimizer=mod_optimizers[k + '_aux'],
                        **mod_scheduler_args)

    # Setup warmup schedulers:
    if args.early_warmup > 0:
        warmup_schedulers = {}
        for k in mod_schedulers.keys():
            warmup_schedulers[k + '_warmup'] = optim.lr_scheduler.LinearLR(
                optimizer=mod_optimizers[k],
                start_factor=1/args.early_warmup,
                end_factor=1.0,
                total_iters=args.early_warmup)

        mod_schedulers.update(warmup_schedulers)

    return mod_optimizers, mod_schedulers, mod_grad_accumulate


def resume_optimizer(mod_optimizers, mod_schedulers, checkpoint, gpu=True):
    """Resume training optimizers and schedulers from a previous checkpoint

    Parameters
    ----------
    mod_optimizers : list of torch.optim.Optimizer
        The neurla network optimizer method
    mod_schedulers : list of torch.optim.lr_scheduler or None
        The learning rate scheduler for the optimizer
    checkpoint : str
        Path to a previous training checkpoint
    gpu : bool
        Wether use GPUs to train the neural network or not
    """
    if not gpu:
        checkpoint_state = torch.load(checkpoint, map_location='cpu')
    else:
        checkpoint_state = torch.load(checkpoint)

    for k in mod_optimizers.keys():
        mod_optimizers.load_state_dict(checkpoint_state[k + '_optimizer'])

    for k in mod_schedulers.keys():
        mod_schedulers.load_state_dict(checkpoint_state[k + '_scheduler'])


def main(args):
    """Set up the training environment

    Parameters
    ----------
    args : dict or Namespace
        The set of parameters passed to the different constructors to set up
        the convolutional autoencoder training.
    """
    logger = logging.getLogger(args.mode + '_log')

    train_data, valid_data, num_classes = utils.get_data(args)
    args.num_classes = num_classes

    model = setup_network(args)
    criterion, stopping_criteria = setup_criteria(args,
                                                  checkpoint=args.checkpoint)

    (mod_optimizers,
     mod_schedulers,
     mod_grad_accumulate) = setup_optim(model, args)

    if args.resume_optimizer:
        resume_optimizer(mod_optimizers, mod_schedulers, args.checkpoint,
                         gpu=args.gpu)

    # Log the training setup
    logger.info('Network architecture:')
    for k in model.keys():
        logger.info('\n{} (is trainable: {})'.format(
            k,
            k in args.trainable_modules))
        logger.info(model[k])

    logger.info('\nCriterion:')
    logger.info(criterion)

    logger.info('\nStopping criterion:')
    for k, crit in stopping_criteria.items():
        logger.info('\n' + k)
        logger.info(crit)

    logger.info('\nOptimization parameters:')
    for k in mod_optimizers.keys():
        logger.info('\n{}'.format(k))
        logger.info(mod_optimizers[k])

    logger.info('\nScheduler parameters:')
    for k in mod_schedulers.keys():
        logger.info('\n{}'.format(k))
        logger.info(mod_schedulers[k])

    train(model, train_data, valid_data, criterion, stopping_criteria,
          mod_optimizers,
          mod_schedulers,
          mod_grad_accumulate,
          args)


if __name__ == '__main__':
    args = utils.get_args(task='autoencoder', mode='training')

    utils.setup_logger(args)

    main(args)

    logging.shutdown()
