import numpy as np
import torch

import json
import argparse

from ._info import DATASETS, CRITERIONS


def get_training_args():
    parser = argparse.ArgumentParser('Training of an image compression model based on a convolutional autoencoer')
    parser.add_argument('-c', '--config', dest='config_file', type=str, help='A configuration .json file')

    parser.add_argument('-rs', '--seed', type=int, dest='seed', help='Seed for random number generators', default=-1)
    parser.add_argument('-s', '--steps', type=int, dest='steps', help='Number of training steps', default=1e5)
    parser.add_argument('-cs', '--checksteps', type=int, dest='checkpoint_steps', help='Create a checkpoint every this number of steps', default=1e3)
    
    parser.add_argument('-rm', '--resume', type=str, dest='resume', help='Resume training from an existing checkpoint')
    
    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)
    parser.add_argument('-ld', '--logdir', type=str, dest='log_dir', help='Directory where all logging and model checkpoints are stored', default='.')
    parser.add_argument('-dd', '--datadir', type=str, dest='data_dir', help='Directory where the data is stored', default='.')

    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
    parser.add_argument('-ds', '--dataset', type=str, dest='dataset', help='Dataset used for training the model', default=DATASETS[0], choices=DATASETS)
    parser.add_argument('-dwn', '--download', dest='download_data', action='store_true', help='Download the dataset if it is not in the data directory', default=False)

    parser.add_argument('-cr', '--criterion', type=str, dest='criterion', help='Training criterion for the compression evaluation', default=CRITERIONS[0], choices=CRITERIONS)

    parser.add_argument('-ich', '--inputch', type=int, dest='channels_org', help='Number of channels in the input data', default=3)
    parser.add_argument('-nch', '--netch', type=int, dest='channels_net', help='Number of channels in the analysis and synthesis tracks', default=8)
    parser.add_argument('-bch', '--bnch', type=int, dest='channels_bn', help='Number of channels of the compressed representation', default=16)
    parser.add_argument('-ech', '--expch', type=int, dest='channels_expansion', help='Rate of expansion of the number of channels in the analysis and synthesis tracks', default=1)
    parser.add_argument('-cl', '--compl', type=int, dest='compression_level', help='Level of compression', default=3)
    parser.add_argument('-dl', '--distl', type=float, dest='distorsion_lambda', help='Distorsion penalty parameter (lambda)', default=0.01)
    parser.add_argument('-eK', '--entK', type=float, dest='K', help='Number of layers in the latent space of the factorized entropy model', default=4)
    parser.add_argument('-er', '--entr', type=float, dest='r', help='Number of channels in the latent space of the factorized entropy model', default=3)

    parser.add_argument('-bs', '--batch', type=int, dest='batch_size', help='Batch size for the training step', default=16)
    parser.add_argument('-vbs', '--valbatch', type=int, dest='val_batch_size', help='Batch size for the validation step', default=32)
    parser.add_argument('-lr', '--lrate', type=float, dest='learning_rate', help='Optimizer initial learning rate', default=1e-4)

    config_parser = argparse.ArgumentParser(parents=[parser], add_help=False)

    args = parser.parse_args()

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

    args.mode = 'training'

    return args


def get_testing_args():
    parser = argparse.ArgumentParser('Testing of an image compression-decompression model')
    parser.add_argument('-c', '--config', type=str, dest='config_file', help='A configuration .json file')

    parser.add_argument('-rs', '--seed', type=int, dest='seed', help='Seed for random number generators', default=-1)
    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be tested')
    
    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)
    parser.add_argument('-ld', '--logdir', type=str, dest='log_dir', help='Directory where all logging and model checkpoints are stored', default='.')
    parser.add_argument('-dd', '--datadir', type=str, dest='data_dir', help='Directory where the data is stored', default='.')

    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
    parser.add_argument('-ds', '--dataset', type=str, dest='dataset', help='Dataset used for training the model', default='MNIST', choices=DATASETS)
    parser.add_argument('-dwn', '--download', dest='download_data', action='store_true', help='Download the dataset if it is not in the data directory', default=False)

    parser.add_argument('-bs', '--batch', type=int, dest='batch_size', help='Batch size for the training step', default=16)

    config_parser = argparse.ArgumentParser(parents=[parser], add_help=False)

    args = parser.parse_args()

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
    
    args.mode = 'testing'

    return args


def get_compress_args():
    parser = argparse.ArgumentParser('Testing of an image compression model')
    parser.add_argument('-c', '--config', type=str, dest='config_file', help='A configuration .json file')

    parser.add_argument('-rs', '--seed', type=int, dest='seed', help='Seed for random number generators', default=-1)
    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be tested')
    
    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)

    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
    parser.add_argument('-i', '--input', type=str, nargs='+', dest='input', help='Input images to compress (list of images).')
    parser.add_argument('-o', '--output', type=str, dest='output_dir', help='Output directory to store the compressed image')

    config_parser = argparse.ArgumentParser(parents=[parser], add_help=False)

    args = parser.parse_args()

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
    
    args.mode = 'compress'

    return args


def get_decompress_args():
    parser = argparse.ArgumentParser('Testing of an image decompression model')
    parser.add_argument('-c', '--config', type=str, dest='config_file', help='A configuration .json file')

    parser.add_argument('-rs', '--seed', type=int, dest='seed', help='Seed for random number generators', default=-1)
    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be tested')
    
    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)

    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
    parser.add_argument('-i', '--input', type=str, nargs='+', dest='input', help='Input compressed images (list of .pth files)')
    parser.add_argument('-o', '--output', type=str, dest='output_dir', help='Output directory to store the decompressed image')

    config_parser = argparse.ArgumentParser(parents=[parser], add_help=False)

    args = parser.parse_args()

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
    
    args.mode = 'decompress'

    return args
