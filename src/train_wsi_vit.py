import copy
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


def valid(cae_model, model, data, criterion, args):
    """ Validation step.
    Evaluates the performance of the network in its current state using the
    full set of validation elements.

    Parameters
    ----------
    cae_model : dict
        The convolutional autoencoder mode to compress the inputs
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

    compress_forward_step = models.decorate_trainable_modules(
        trainable_modules=None,
        enabled_modules=["encoder", "fact_ent"])

    valid_forward_step = models.decorate_trainable_modules(
        trainable_modules=None,
        enabled_modules=args.enabled_modules)

    for k in model.keys():
        model[k].eval()

    sum_loss = 0
    num_examples = 0
    rec_metrics = None
    channel_e_history = []

    if args.progress_bar:
        q = tqdm(desc='Validating', position=1, leave=None)

    for i, (p, x, t) in enumerate(data):
        comp_x = compress_forward_step(x, cae_model)["y_q"]
        b, c, _, _ = comp_x.shape
        comp_x = comp_x.permute(1, 0, 2, 3).reshape(1, c, 1, -1)

        t = torch.max(t).view(1, -1)

        output = valid_forward_step(comp_x, model, position=p)
        output["t_pred"] = output["t_pred"].view(1, -1)
        if output["t_aux_pred"] is not None:
            output["t_aux_pred"] = output["t_aux_pred"].view(1, -1)

        loss_dict = criterion(inputs=x, outputs=output, targets=t, net=model)
        loss = torch.mean(loss_dict['loss'])
        sum_loss += loss.item()
        num_examples += b

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

        if i % 1000 == 0:
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

    mean_loss = sum_loss / num_examples

    return mean_loss, avg_rec_metrics


def train(cae_model, model, train_data,
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
    cae_model : dict
        The convolutional autoencoder mode to compress the inputs
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

    compress_forward_step = models.decorate_trainable_modules(
        trainable_modules=None,
        enabled_modules=["encoder", "fact_ent"])

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
        for i, (p, x, t) in enumerate(train_data):
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
                # Start of training step
                sub_step += 1

                comp_x = compress_forward_step(x, cae_model)["y_q"]
                b, c, _, _ = comp_x.shape
                comp_x = comp_x.permute(1, 0, 2, 3).reshape(1, c, 1, -1)

                t = torch.max(t).view(1, -1)

                output = train_forward_step(comp_x, model, position=p)
                output["t_pred"] = output["t_pred"].view(1, -1)
                if output["t_aux_pred"] is not None:
                    output["t_aux_pred"] = output["t_aux_pred"].view(1, -1)

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

            if i % max(1, 1000) == 0:
                current_lr = ''
                for k, sched in mod_schedulers.items():
                    if hasattr(sched, '_last_lr'):
                        current_lr += '{}={:.2e} '.format(k, sched._last_lr[0])
                    else:
                        current_lr += '{}=None '.format(k)

                channel_e = int(torch.median(torch.LongTensor(channel_e_history)))

                (log_str,
                 curr_rec_metrics) = utils.log_info(step, i + 1,
                                                    None,
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
                 val_avg_metrics) = valid(cae_model, model, valid_data,
                                          criterion,
                                          args)

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
                        trn_avg_metrics[k + '_last_lr'] = sched._last_lr[0]
                    else:
                        current_lr += '{}=None '.format(k)
                        trn_avg_metrics[k + '_last_lr'] = float('nan')

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
    """Setup a neural network for image compression/decompression.

    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. All the parameters are
        passed directly to the model constructor. This way, the constructor can
        take the parameters needed that have been passed by the user.

    Returns
    -------
    cae_model : dict
        The convolutional autoencoder model used to compress original inputs.
        This model is fixed and not trained.

    model : dict
        The models for subsequent extra tasks.
    """
    cae_args = copy.deepcopy(args)
    cae_args.checkpoint = args.cae_checkpoint
    cae_args.enabled_modules = ["encoder", "fact_ent"]
    cae_model = models.autoencoder_from_state_dict(cae_args, gpu=args.gpu,
                                                   train=False)

    args = utils.setup_network_args(args)
    model = {}
    if 'class_model' in args.enabled_modules:
        model['class_model'] = models.classifier_from_state_dict(args,
                                                                 gpu=args.gpu,
                                                                 train=train)
    if 'seg_model' in args.enabled_modules:
        model['seg_model'] = models.segmenter_from_state_dict(args,
                                                              gpu=args.gpu,
                                                              train=train)

    return cae_model, model


def main(args):
    """Set up the training environment

    Parameters
    ----------
    args : dict or Namespace
        The set of parameters passed to the different constructors to set up
        the convolutional autoencoder training.
    """
    logger = logging.getLogger(args.mode + '_log')

    args.return_batches = True
    args.return_positions = True
    train_data, valid_data, num_classes = utils.get_data(args)
    args.num_classes = num_classes

    comp_model, model = setup_network(args)

    (criterion,
     stopping_criteria) = utils.setup_criteria(args,
                                               checkpoint=args.checkpoint)

    (mod_optimizers,
     mod_schedulers,
     mod_grad_accumulate) = utils.setup_optim(model, args)

    if args.resume_optimizer:
        utils.resume_optimizer(mod_optimizers, mod_schedulers, args.checkpoint,
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

    train(comp_model, model, train_data, valid_data, criterion,
          stopping_criteria,
          mod_optimizers,
          mod_schedulers,
          mod_grad_accumulate,
          args)


if __name__ == '__main__':
    args = utils.get_args(task='autoencoder', mode='training')

    utils.setup_logger(args)

    main(args)

    logging.shutdown()
