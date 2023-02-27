import numpy as np
import torch

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
    parsed_args = {}

    for arg in args:
        arg_name, arg_type_val = arg.split("=")
        arg_type, arg_val = arg_type_val.split(":")

        if arg_type == 'int':
            arg_val = int(arg_val)
        elif arg_type == 'float':
            arg_val = float(arg_val)
        else:
            arg_val = arg_val
    
        parsed_args[arg_name] = arg_val

    return parsed_args
