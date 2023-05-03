import os
import logging
import torch
from inspect import signature

from .critutils._metrics import compute_metrics_per_image
from ._info import VER, SEG_VER


def setup_logger(args):
    """ Sets up a logger for the diferent purposes.
    When training a model on a HPC cluster, it is better to save the logs into a file, rather than printing to a console.

    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. Only the code version and random seed are used from this.
    """
    args.version = SEG_VER if args.task in ["segmentation"] else VER

    if torch.cuda.is_available() and args.use_gpu:
        args.gpu = True
    else:
        args.gpu = False

    # Create the logger
    logger = logging.getLogger(args.mode + '_log')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # By now, only if the model is training or testing, the logs are stored into a file
    if args.mode in ['training', 'test']:
        logger_fn = os.path.join(args.log_dir, '%s_ver%s_%s%s.log' % (args.mode, args.version, args.seed, args.log_identifier))
        fh = logging.FileHandler(logger_fn, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
        logger.setLevel(logging.DEBUG)

    if args.print_log:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    
    logger.info('Code version %s, with random number generator seed: %s\n' % (args.version, args.seed))
    logger.info(args)


def save_state(name, model_state, args):
    """ Save the current state of the model at this step of training.
    
    Parameters
    ----------
    name : str
        The checkpoint file name
    model_state : dict
        A dictionary containing the current training state
    args : Namespace
        The input arguments passed at running time. Only the code version and random seed are used from this.
    """
    if isinstance(args, dict):
        save_fn = os.path.join(args['log_dir'], name + '_ver%s_%s%s.pth' % (args['version'], args['seed'], args['log_identifier']))
    else:
        save_fn = os.path.join(args.log_dir, name + '_ver%s_%s%s.pth' % (args.version, args.seed, args.log_identifier))

    torch.save(model_state, save_fn)
    logger = logging.getLogger('training_log')
    logger.info('Saved model in %s' % save_fn)


def checkpoint(step, model, mod_optimizers, mod_schedulers, best_valid_loss,
               train_loss_history,
               valid_loss_history,
               args,
               extra_info={}):
    """Creates a checkpoint with the current trainig state.

    Parameters
    ----------
    step : int
        The current training step    
    model : torch.nn.Module
        The network model in the current state    
    optimizer : torch.optim.Optimizer
        The parameter's optimizer method
    scheduler : torch.optim.lr_scheduler or None
        If provided, a learning rate scheduler for the optimizer
    best_valid_loss : float
        The current best validation loss obtained through all training
    train_loss_history : list[float]
        A list of all training criterion evaluation during the training
    valid_loss_history : list[float]
        A list of all validation criterion evaluation during the training
    args : Namespace
        The input arguments passed at running time
    extra_info : dict

    Returns
    -------
    best_valid_loss : float
        The updated current best validation loss obtained through all training
    """
    # Create a dictionary with the current state as checkpoint
    training_state = dict(args.__dict__)

    training_state['best_val'] = best_valid_loss
    training_state['step'] = step
    training_state['train_loss'] = train_loss_history
    training_state['valid_loss'] = valid_loss_history
    training_state['code_version'] = args.version

    # Append any extra information passed by the training loop
    training_state.update(extra_info)

    if 'fact_ent' in model.keys() and 'fact_ent' in args.trainable_modules:
        model['fact_ent'].module.update(force=True)

    for k in model.keys():
        training_state[k] = model[k].module.state_dict()

    for k, optim in mod_optimizers.items():
        training_state['optimizer_' + k] = optim.state_dict()
    else:
        training_state['optimizer_' + k] = None

    for k, sched in mod_schedulers.items():
        training_state['scheduler_' + k] = sched.state_dict()
    else:
        training_state['scheduler_' + k] = None

    save_state('last', training_state, args)

    if valid_loss_history[-1] < best_valid_loss:
        best_valid_loss = valid_loss_history[-1]
        save_state('best', training_state, args)

    return best_valid_loss


def load_state(args):
    """ Loads an exisiting training state for its deployment, or resume its training.

    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time

    Returns
    ----------
    state : dict
        A dictionary containing all the fields saved as checkpoint
    """

    # If optimizer is not none, the state is being open as checkpoint to resume training
    if args.mode in ['training']:
        save_fn = os.path.join(args.log_dir, 'last_ver%s_%s.pth' % (args.version, args.seed))
    else:
        save_fn = args.checkpoint

    if not os.path.exists(save_fn):
        raise ValueError('The checkpoint %s does not exist' % save_fn)
        return None

    if not torch.cuda.is_available() or not args.gpu:
        state = torch.load(save_fn, map_location=torch.device('cpu'))
        state['gpu'] = False
    else:
        state = torch.load(save_fn)

    logger = logging.getLogger(args.mode + '_log')
    logger.info('Loaded model from %s' % save_fn)

    logger.info('Training arguments')
    logger.info(state)

    return state


def log_info(step, sub_step, len_data, model, inputs, targets, output,
             avg_loss=None,
             loss_dict=None,
             channel_e=-1,
             step_type='Training',
             lr=None,
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

    recorded_metrics = {}
    if avg_loss is not None:
        log_string += '{} Loss {:.4f}'.format(step_type, avg_loss)
        recorded_metrics['loss'] = avg_loss
    
    if loss_dict is None:
        loss_dict = {}

    if 'dist' in loss_dict:
        log_string += ' D=[{}]'.format(','.join(['%0.4f' % d.item()
                                                for d in loss_dict['dist']]))
        recorded_metrics['D'] = [d.item() for d in loss_dict['dist']]

        log_string += ' Xo={:.2f},{:.2f},std={:.2f}'.format(
            inputs.min(),
            inputs.max(),
            inputs.std())

        if isinstance(output['x_r'], list):
            x_r = output['x_r'][0].detach().cpu()
        else:
            x_r = output['x_r'].detach().cpu()

        x_r_min = x_r.min()
        x_r_max = x_r.max()
        x_r_std = x_r.std()
        log_string += ' Xr={:.2f},{:.2f},std={:.2f}'.format(x_r_min, x_r_max,
                                                            x_r_std)
        recorded_metrics['x_r_min'] = x_r_min
        recorded_metrics['x_r_max'] = x_r_max
        recorded_metrics['x_r_std'] = x_r_std

    if 'rate_loss' in loss_dict:
        log_string += ' R={:.2f}'.format(loss_dict['rate_loss'].item())
        recorded_metrics['R'] = loss_dict['rate_loss'].item()

        y = output['y'].detach().cpu()
        p_y = output['p_y'].detach().cpu()

        y_min = y.min()
        y_max = y.max()
        p_y_min = p_y.min()
        p_y_max = p_y.max()

        log_string += ' BN={:.2f},{:.2f} P={:.2f},{:.2f}'.format(y_min, y_max,
                                                                 p_y_min,
                                                                 p_y_max)
        recorded_metrics['y_min'] = y_min
        recorded_metrics['y_max'] = y_max
        recorded_metrics['p_y_min'] = p_y_min
        recorded_metrics['p_y_max'] = p_y_max

    if 'entropy_loss' in loss_dict:
        log_string += ' A={:.3f}'.format(loss_dict['entropy_loss'].item())
        recorded_metrics['A'] = loss_dict['entropy_loss'].item()

        quantiles = model['fact_ent'].module.quantiles.detach().cpu()
        q1 = quantiles[:, 0, 0].median()
        q2 = quantiles[:, 0, 1].median()
        q3 = quantiles[:, 0, 2].median()

        log_string += ' QP={:.2f},{:.2f},{:.2f}'.format(q1, q2, q3)
        recorded_metrics['q1'] = q1
        recorded_metrics['q2'] = q2
        recorded_metrics['q3'] = q3

    if 'energy' in loss_dict:
        log_string += ' E={:.3f}'.format(loss_dict['energy'].item())
        recorded_metrics['E'] = loss_dict['energy'].item()

    if 'class_error' in loss_dict:
        if output.get('t_pred', None) is not None:
            class_task_type='C'
            class_task_key = 't_pred'
        else:
            class_task_type='S'
            class_task_key = 's_pred'

        log_string += ' {}={:.3f}'.format(class_task_type, 
                                          loss_dict['class_error'].item())
        recorded_metrics[class_task_type] = loss_dict['class_error'].item()

        class_metrics = compute_metrics_per_image(output[class_task_key],
                                                  targets,
                                                  top_k=5,
                                                  num_classes=None)

        for k, m in class_metrics.items():
            log_string += ' {}:{:.3f}'.format(k, m)
            recorded_metrics[k] = m

    if channel_e >= 0:
        log_string += ' Ch={}'.format(int(channel_e))
        recorded_metrics['Ch'] = channel_e

    if lr is not None and len(lr) > 0:
        log_string += ' lr={}'.format(lr)

    return log_string, recorded_metrics