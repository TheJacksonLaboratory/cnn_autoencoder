import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import models
import utils

from inspect import signature

from tqdm import tqdm


optimization_algorithms = {"Adam": optim.Adam,
                           "SGD": optim.SGD}

scheduler_options = {"ReduceOnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
                     "StepLR": optim.lr_scheduler.StepLR,
                     "LinearLR": optim.lr_scheduler.LinearLR}


def log_info(step, sub_step, len_data, model, inputs, targets, output,
             avg_loss,
             loss_dict,
             channel_e,
             step_type='Training',
             progress_bar=False):
    if step is not None:
        log_string = '[{:06d}]'.format(step)
    else:
        log_string = ''

    if not progress_bar:
        if len_data is None:
            log_string += '[{:04d}] '.format(sub_step)
        else:
            log_string += '[{:04d}/{:04d}] '.format(sub_step, len_data)

    log_string += '{} Loss {:.4f}'.format(step_type, avg_loss)

    if 'dist' in loss_dict:
        log_string += ' D=[{}]'.format(','.join(['%0.4f' % d.item()
                                                for d in loss_dict['dist']]))
        log_string += ' Xo={:.2f},{:.2f},std={:.2f}'.format(
            inputs.min(),
            inputs.max(),
            inputs.std())

        if isinstance(output['x_r'], list):
            log_string += ' Xr={:.2f},{:.2f},std={:.2f}'.format(
                output['x_r'][0].detach().min(),
                output['x_r'][0].detach().max())
        else:
            log_string += ' Xr={:.2f},{:.2f},std={:.2f}'.format(
                output['x_r'].detach().min(),
                output['x_r'].detach().max(),
                output['x_r'].detach().std())

    if 'rate_loss' in loss_dict:
        log_string += ' R={:.2f}'.format(loss_dict['rate_loss'].item())

        log_string += ' BN={:.2f},{:.2f} P={:.2f},{:.2f}'.format(
            output['y'].detach().min(),
            output['y'].detach().max(),
            output['p_y'].detach().min(),
            output['p_y'].detach().max())

    if 'entropy_loss' in loss_dict:
        log_string += ' A={:.3f}'.format(loss_dict['entropy_loss'].item())
        quantiles = model['fact_ent'].module.quantiles.detach()
        quantiles_info = (quantiles[:, 0, 0].median(),
                        quantiles[:, 0, 1].median(),
                        quantiles[:, 0, 2].median())
        log_string += ' QP={:.2f},{:.2f},{:.2f}'.format(*quantiles_info)

    if 'energy' in loss_dict:
        log_string += ' E={:.3f}'.format(loss_dict['energy'].item())

    if 'class_error' in loss_dict:
        log_string += ' C={:.3f}'.format(loss_dict['class_error'].item())
        num_classes = output['t_pred'].size(1)
        class_metrics = utils.compute_class_metrics(output['t_pred'],
                                                    targets,
                                                    top_k=5,
                                                    num_classes=num_classes)
        if progress_bar:
            log_string += ' acc:{:.3f} top5:{:.3f}'.format(
                class_metrics['acc'],
                class_metrics['acc_top'])
        else:
            for k, m in class_metrics.items():
                log_string += ' {}:{:.3f}'.format(k, m)

    if channel_e >= 0:
        log_string += ' Ch={}'.format(int(channel_e))

    return log_string


def forward_step(x, model):
    y = model["encoder"](x)
    y_q, p_y = model["fact_ent"](y)
    x_r = model["decoder"](y_q)
    t_pred, t_aux_pred = model["class_model"](y_q)

    return dict(x_r=x_r, y=y, y_q=y_q, p_y=p_y, t_pred=t_pred,
                t_aux_pred=t_aux_pred)


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

    for k in model.keys():
        model[k].eval()

    sum_loss = 0
    channel_e_history = []

    if args.progress_bar:
        q = tqdm(total=len(data), desc='Validating', position=1, leave=None)

    with torch.no_grad():
        for i, (x, t) in enumerate(data):
            output = forward_step(x, model)
            t = t.to(output['y_q'].device)

            loss_dict = criterion(input=x, output=output, target=t,
                                  net=model)
            loss = torch.mean(loss_dict['loss'])
            sum_loss += loss.item()


            channel_e_history.append(loss_dict['channel_e'])
            channel_e = int(torch.median(torch.LongTensor(channel_e_history)))

            if args.progress_bar:
                q.set_description(
                    log_info(None, i + 1, None, model, x, t, output,
                             sum_loss / (i + 1),
                             loss_dict,
                             channel_e,
                             step_type='Validation',
                             progress_bar=True))
                q.update()

            if i % max(1, int(0.1 * len(data))) == 0:
                logger.debug(log_info(None, i + 1, len(data), model, x, t,
                                      output,
                                      sum_loss / (i + 1),
                                      loss_dict,
                                      channel_e,
                                      step_type='Validation',
                                      progress_bar=False))

    if args.progress_bar:
        q.close()

    mean_loss = sum_loss / len(data)

    return mean_loss


def train(model, train_data, valid_data, criterion, stopping_criteria,
          optimizer,
          scheduler,
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
                optimizer.zero_grad()

                output = forward_step(x, model)
                t = t.to(output['y_q'].device)

                loss_dict = criterion(input=x, output=output, target=t,
                                      net=model)

                loss = torch.mean(loss_dict['loss'])
                loss.backward()

                if 'entropy_loss' in loss_dict:
                    aux_loss = torch.mean(loss_dict['entropy_loss'])
                    aux_loss.backward()

                # Clip the gradients to prevent from exploding gradients
                # problems
                for k in args.trainable_modules:
                    nn.utils.clip_grad_norm_(model[k].parameters(),
                                             max_norm=1.0)

                optimizer.step()

                step_loss = loss.item()
                sub_step_loss += step_loss
                channel_e_history.append(loss_dict.get('channel_e', -1))

                # When training with penalty on the energy of the compression
                # representation, update the respective stopping criterion
                if 'penalty' in stopping_criteria.keys():
                    if args.progress_bar:
                        channel_e = int(torch.median(torch.LongTensor(channel_e_history)))
                        q_penalty.set_description(
                            log_info(sub_step, None, model, x, t, output,
                                        sub_step_loss / sub_step,
                                        loss_dict,
                                        channel_e,
                                        step_type='Sub-iter',
                                        progress_bar=True))

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

            if (scheduler is not None
               and 'metrics' not in signature(scheduler.step).parameters):
                scheduler.step()

            # End of training step
            if args.progress_bar:
                channel_e = int(torch.median(torch.LongTensor((channel_e_history))))

                q.set_description(
                    log_info(None, i + 1, None, model, x, t, output,
                             sum_loss / (i + 1),
                             loss_dict,
                             channel_e,
                             step_type='Training',
                             progress_bar=True))
                q.update()

            # Log the training performance every 10% of the training set
            if i % max(1, int(0.01 * len(train_data))) == 0:
                channel_e = int(torch.median(torch.LongTensor(channel_e_history)))

                logger.debug(log_info(step, i + 1, train_data_size, model, x,
                                      t,
                                      output,
                                      sum_loss / (i + 1),
                                      loss_dict,
                                      channel_e,
                                      step_type='Training',
                                      progress_bar=False))

            # Checkpoint step
            keep_training = stopping_criteria['early_stopping'].check()

            if (not keep_training
              or (step >= args.early_warmup
                  and (step-args.early_warmup) % args.checkpoint_steps == 0)
                  and step > 1):
                train_loss = sum_loss / (i+1)

                # Evaluate the model with the validation set
                valid_loss = valid(model, valid_data, criterion, args)

                for k in model.keys():
                    if k in args.trainable_modules:
                        model[k].train()

                stopping_info = ';'.join(map(lambda k_sc:\
                                             k_sc[0] + ": " + k_sc[1].__repr__(),
                                             stopping_criteria.items()))

                # If there is a learning rate scheduler, perform a step
                # Log the overall network performance every checkpoint step
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
                channel_e = int(torch.median(torch.LongTensor(channel_e_history)))
                best_valid_loss = utils.checkpoint(step, model, optimizer,
                                                   scheduler,
                                                   best_valid_loss,
                                                   train_loss_history,
                                                   valid_loss_history,
                                                   args,
                                                   {'channel_e': channel_e})
                channel_e_history = []

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
    args.multiscale_analysis = 'Multiscale' in args.criterion
    model = models.setup_autoencoder_modules(**args.__dict__)
    model.update(models.setup_classifier_modules(**args.__dict__))

    # If there are more than one GPU, DataParallel handles automatically the
    # distribution of the work.
    for k in model.keys():
        model[k] = nn.DataParallel(model[k])

        if args.gpu:
            model[k].cuda()

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
    optim_algo = optimization_algorithms[args.optim_algo]

    # Parse the values of learning rate and weight decay for each module. These
    # must be passed in the form `--mod-lrate class_model=0.1`, e.g. to assign
    # an initial learning rate to the weights update of the `class` module of
    # the neural network.
    modules_learning_rate = dict([(mlr.split('=')[0], float(mlr.split('=')[1]))
                                  for mlr in args.mod_learning_rate])

    modules_weight_decay = dict([(mwd.split('=')[0], float(mwd.split('=')[1]))
                                  for mwd in args.mod_weight_decay])

    modules_aux_learning_rate = dict([(mlr.split('=')[0],
                                       float(mlr.split('=')[1]))
                                      for mlr in args.mod_aux_learning_rate])

    modules_aux_weight_decay = dict([(mwd.split('=')[0],
                                      float(mwd.split('=')[1]))
                                     for mwd in args.mod_aux_weight_decay])

    optim_groups = []
    for k in model.keys():
        optim_aux_pars = {}
        optim_pars = {}
        pars = []
        aux_pars = []

        if k in args.trainable_modules:
            for par_name, par in model[k].named_parameters():
                if 'quantiles' in par_name or 'aux' in par_name:
                    aux_pars.append(par)
                else:
                    pars.append(par)

            optim_pars['params'] = pars
            module_lr = modules_learning_rate.get(k, None)
            module_wd = modules_weight_decay.get(k, None)

            if module_lr is not None:
                optim_pars['lr'] = module_lr

            if module_wd is not None:
                optim_pars['weight_decay'] = module_wd

            optim_groups.append(optim_pars)

            if len(aux_pars) > 0:
                optim_aux_pars['params'] = aux_pars
                module_aux_lr = modules_aux_learning_rate.get(k, None)
                module_aux_wd = modules_aux_weight_decay.get(k, None)

                if module_aux_lr is not None:
                    optim_aux_pars['lr'] = module_aux_lr

                if module_wd is not None:
                    optim_aux_pars['weight_decay'] = module_aux_wd

                optim_groups.append(optim_aux_pars)

        else:
            model[k].eval()

    optimizer = optim_algo(optim_groups,
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    # Only the the reduce on plateau, or none at all scheduler are used
    if args.scheduler_type == 'None':
        scheduler = None

    elif args.scheduler_type in scheduler_options.keys():
        if args.scheduler_type == "LinearLR":
            scheduler_args = dict(start_factor=0.3333333333333333,
                                  end_factor=1.0,
                                  total_iters=5,
                                  last_epoch=-1,
                                  verbose=False)

        elif args.scheduler_type == "ReduceOnPlateau":
            scheduler_args = dict(mode='min', factor=0.1, patience=10,
                                  threshold=0.0001,
                                  threshold_mode='rel',
                                  cooldown=0,
                                  min_lr=0,
                                  eps=1e-08,
                                  verbose=False)

        elif args.scheduler_type == "StepLR":
            scheduler_args = dict(gamma=0.1, last_epoch=- 1, verbose=False)

        else:
            scheduler_args = {}

        scheduler = scheduler_options[args.scheduler_type](optimizer=optimizer,
                                                           **scheduler_args)
    else:
        raise ValueError('Scheduler \"%s\" ' % args.scheduler_type
                         + 'is not implemented')

    return optimizer, scheduler


def resume_checkpoint(model, optimizer, scheduler, checkpoint, gpu=True,
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
                                      map_location='cpu')
    else:
        checkpoint_state = torch.load(checkpoint)

    for k in model.keys():
        assert k in checkpoint_state
        model[k].module.load_state_dict(checkpoint_state[k],
                                                  strict=False)

    if checkpoint_state['args']['version'] == '0.5.5':
        for color_layer in model['decoder'].module.color_layers:
            color_layer[0].weight.data.copy_(
                checkpoint_state['decoder']['synthesis_track.4.weight'])

    model['fact_ent'].module.update(force=True)

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

    train_data, valid_data, num_classes = utils.get_data(args)
    args.num_classes = num_classes

    model = setup_network(args)
    criterion, stopping_criteria = setup_criteria(args, checkpoint=args.resume)
    optimizer, scheduler = setup_optim(model, args)

    if args.resume is not None:
        resume_checkpoint(model, optimizer, scheduler, args.resume,
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
    logger.info(optimizer)

    logger.info('\nScheduler parameters:')
    logger.info(scheduler)

    train(model, train_data, valid_data, criterion, stopping_criteria,
          optimizer,
          scheduler,
          args)


if __name__ == '__main__':
    args = utils.get_args(task='autoencoder', mode='training')

    utils.setup_logger(args)

    main(args)

    logging.shutdown()
