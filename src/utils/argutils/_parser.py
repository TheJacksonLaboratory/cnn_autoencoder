import numpy as np
import torch
import torch.optim as optim

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

import json
import argparse

from ._confargs import config_args
from ._lcargs import lc_args
from ._logargs import logging_args
from ._critargs import criteria_args
from ._dataargs import data_args
from ._taskargs import task_args

def override_config_file(parser):
    args = parser.parse_args()

    config_parser = argparse.ArgumentParser(parents=[parser], add_help=False)

    # Parse the arguments from a json configure file, when given
    if args.config_file is not None:
        if '.json' in args.config_file:
            config = json.load(open(args.config_file, 'r'))
            config_parser.set_defaults(**config)

        else:
            raise ValueError('The configure file must be a .json file')

    # The parameters passed through a json file are overridable from console instructions
    args = config_parser.parse_args()

    # Set the random number generator seed for reproducibility
    if args.seed < 0:
        args.seed = np.random.randint(1, 100000)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)

    args.use_gpu = args.use_gpu if torch.cuda.is_available() else False

    return args


def get_args(task, mode, parser_only=False):
    parser = argparse.ArgumentParser('Arguments for running ' + task
                                     + ' in mode '
                                     +  mode,
                                     conflict_handler='resolve')

    parser.add_argument('-c', '--config', dest='config_file', type=str,
                        help='A configuration .json file')
    parser.add_argument('-g', '--gpu', action='store_true', dest='use_gpu',
                        help='Use GPU when available')

    all_args = (config_args + lc_args + logging_args + criteria_args
                + data_args
                + task_args)

    for par in all_args:
        if ((task in par['tasks'] or 'all' in par['tasks'])
          and (mode in par['modes'] or 'all' in par['modes'])):
            parser.add_argument(*par['flags'], **par['details'])

    if parser_only:
        return parser

    args = override_config_file(parser)

    args.mode = mode
    args.task = task

    return args


def parse_typed_arguments(args):
    if args is None:
        return {}

    parsed_args = {}

    for arg in args:
        arg_name, arg_type_val = arg.split("=")
        arg_type, arg_val = arg_type_val.split(":")

        if arg_type == 'int':
            arg_val = int(arg_val)
        elif arg_type == 'float':
            arg_val = float(arg_val)
        elif arg_type.lower() == 'none':
            arg_val = None
        else:
            arg_val = arg_val

        parsed_args[arg_name] = arg_val

    return parsed_args


def setup_network_args(args):
    """Setup the parameters that define the neural network architecture.

    If the arguments point to an existing checkpoint, those parameters are used
    instead.

    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. All the parameters are
        passed directly to the model constructor. This way, the constructor can
        take the parameters needed that have been passed by the user.

    Returns
    -------
    args : Namespace
        The updated arguments needed to load models from the state dictionary.
    """
    if hasattr(args, "checkpoint") and args.checkpoint is not None:
        checkpoint_state = torch.load(args.checkpoint, map_location='cpu')
        checkpoint_state.update(args.__dict__)
        args.__dict__ = checkpoint_state

    args.multiscale_analysis = 'Multiscale' in args.criterion

    return args



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
    optim_algos = parse_typed_arguments(args.mod_optim_algo)

    scheduler_algos = {}
    for mod_pars in args.mod_scheduler_algo:
        mod = mod_pars[:mod_pars.find('=')]
        sched_type_args = mod_pars[mod_pars.find('=') + 1:]
        sched_type = sched_type_args.split(',')[0]
        if sched_type.lower() == 'none':
            sched_type = None
        sched_args = sched_type_args.split(',')[1:]
        scheduler_algos[mod] = (sched_type,
                                parse_typed_arguments(sched_args))

    # Parse the values of learning rate and weight decay for each module. These
    # must be passed in the form `--mod-lrate class_model=0.1`, e.g. to assign
    # an initial learning rate to the weights update of the `class` module of
    # the neural network.
    mod_grad_accumulate = parse_typed_arguments(
        args.mod_grad_accumulate)

    mod_learning_rate = parse_typed_arguments(
        args.mod_learning_rate)

    mod_weight_decay = parse_typed_arguments(args.mod_weight_decay)

    mod_aux_learning_rate = parse_typed_arguments(
        args.mod_aux_learning_rate)

    mod_aux_weight_decay = parse_typed_arguments(
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
