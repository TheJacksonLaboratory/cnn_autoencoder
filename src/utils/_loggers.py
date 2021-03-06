import os
import logging
import torch
from inspect import signature


from ._info import VER


def setup_logger(args):
    """ Sets up a logger for the diferent purposes.
    When training a model on a HPC cluster, it is better to save the logs into a file, rather than printing to a console.
    
    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. Only the code version and random seed are used from this.
    """
    args.version = VER

    if torch.cuda.is_available() and args.use_gpu:
        args.gpu = True
    else:
        args.gpu = False
    
    # Create the logger
    logger = logging.getLogger(args.mode + '_log')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # By now, only if the model is training or testing, the logs are stored into a file
    if args.mode in ['training', 'testing']:
        logger_fn = os.path.join(args.log_dir, '%s_ver%s_%s.log' % (args.mode, args.version, args.seed))
        fh = logging.FileHandler(logger_fn)
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
    save_fn = os.path.join(args.log_dir, name + '_ver%s_%s.pth' % (args.version, args.seed))

    torch.save(model_state, save_fn)
    logger = logging.getLogger('training_log')
    logger.info('Saved model in %s' % save_fn)


def checkpoint(step, model, optimizer, scheduler, best_valid_loss, train_loss_history, valid_loss_history, args):
    """ Creates a checkpoint with the current trainig state

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
    
    Returns
    -------
    best_valid_loss : float
        The updated current best validation loss obtained through all training
    """

    # Create a dictionary with the current state as checkpoint
    training_state = dict(
        optimizer=optimizer.state_dict(),
        args=args.__dict__,
        best_val=best_valid_loss,
        step=step,
        train_loss=train_loss_history,
        valid_loss=valid_loss_history,
        code_version=args.version
    )
    
    if args.task == 'autoencoder':
        training_state['embedding'] = model.module.embedding.state_dict()
        training_state['encoder'] = model.module.analysis.state_dict()
        training_state['decoder'] = model.module.synthesis.state_dict()
        training_state['fact_ent'] = model.module.fact_entropy.state_dict()

    elif args.task == 'segmentation':
        training_state['model'] = model.module.state_dict()

    if scheduler is not None:
        if 'metrics' in dict(signature(scheduler.step).parameters).keys():
            scheduler.step(metrics=valid_loss_history[-1])

        training_state['scheduler'] = scheduler.state_dict()
    else:
        training_state['scheduler'] = None
    
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
    if args.mode == 'training':
        save_fn = os.path.join(args.log_dir, 'last_ver%s_%s.pth' % (args.version, args.seed))
    else:
        save_fn = os.path.join(args.trained_model)

    if not torch.cuda.is_available() or not args.gpu:
        state = torch.load(save_fn, map_location=torch.device('cpu'))
        state['args']['gpu'] = False
    else:
        state = torch.load(save_fn)
    
    logger = logging.getLogger(args.mode + '_log')
    logger.info('Loaded model from %s' % save_fn)

    logger.info('Training arguments')
    logger.info(state['args'])

    return state
