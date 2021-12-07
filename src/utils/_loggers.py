import os
import logging
import torch

from ._info import VER


def setup_logger(args):
    args.version = VER

    if torch.cuda.is_available():
        args.gpu = True
    else:
        args.gpu = False
    
    # Create the logger
    logger = logging.getLogger(args.mode + '_log')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

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
    """
    save_fn = os.path.join(args.log_dir, name + '_ver%s_%s.pth' % (args.version, args.seed))

    torch.save(model_state, save_fn)
    logger = logging.getLogger('training_log')
    logger.info('Saved model in %s' % save_fn)


def load_state(args):
    """ Load an existent state into 'model'.
    """

    # If optimizer is not none, the state is being open as checkpoint to resume training
    if args.mode == 'training':
        save_fn = os.path.join(args.log_dir, 'last_ver%s_%s.pth' % (args.version, args.seed))
    else:
        save_fn = os.path.join(args.trained_model)

    if not torch.cuda.is_available():
        state = torch.load(save_fn, map_location=torch.device('cpu'))
    
    else:
        state = torch.load(save_fn)
    
    logger = logging.getLogger(args.mode + '_log')
    logger.info('Loaded model from %s' % save_fn)

    logger.info('Training arguments')
    logger.info(state['args'])

    return state
