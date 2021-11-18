import argparse
import logging
import torch
import os

import numpy as np
from torch.serialization import save


VER = '0.1'
DATASETS = ['MNIST']


def save_state(name, model_state, args):
    save_fn = os.path.join(args.log_dir, name + '_ver%s_%s.pth' % (args.version, args.seed))

    torch.save(model_state, save_fn)
    logger = logging.getLogger('training_log')
    logger.info('Saved model in %s' % save_fn)


def get_training_args():
    parser = argparse.ArgumentParser('Training of an image compression model based on a convolutional autoencoer')
    parser.add_argument('-rs', '--seed', dest='seed', type=int, help='Seed for random number generators', default=-1)
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, help='Number of training epochs', default=10)
    parser.add_argument('-ce', '--checkepochs', dest='checkpoint_epochs', type=int, help='Create a checkpoint every this number of epochs', default=10)
    
    parser.add_argument('-ld', '--logdir', dest='log_dir', help='Directory where all logging and model checkpoints are stored', default='.')
    parser.add_argument('-dd', '--datadir', dest='data_dir', help='Directory where the data is stored', default='.')
    parser.add_argument('-ds', '--dataset', dest='dataset', help='Dataset used for training the model', default='MNIST', choices=DATASETS)

    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
    parser.add_argument('-ich', '--inputch', type=int, dest='input_channels', help='Number of channels in the input data', default=3)
    parser.add_argument('-nch', '--netch', type=int, dest='net_channels', help='Number of channels in the analysis and synthesis tracks', default=8)
    parser.add_argument('-bch', '--bnch', type=int, dest='bn_channels', help='Number of channels of the compressed representation', default=16)
    parser.add_argument('-ech', '--expch', type=int, dest='channels_expansion', help='Rate of expansion of the number of channels in the analysis and synthesis tracks', default=1)
    parser.add_argument('-cl', '--compl', type=int, dest='compression_level', help='Level of compression', default=3)
    parser.add_argument('-dl', '--distl', type=float, dest='distorsion_lambda', help='Distorsion penalty parameter (lambda)', default=0.01)
    parser.add_argument('-eK', '--entK', type=float, dest='factorized_entropy_K', help='Number of layers in the latent space of the factorized entropy model', default=4)
    parser.add_argument('-er', '--entr', type=float, dest='factorized_entropy_r', help='Number of channels in the latent space of the factorized entropy model', default=3)

    parser.add_argument('-lr', '--lrate', dest='learning_rate', help='Optimizer initial learning rate', default=0.01)

    args = parser.parse_args()

    args.version = VER
    
    # Set the random number generator seed for reproducibility
    if args.seed < 0:
        args.seed = np.random.randint(1, 100000)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)

    return args


def setup_logger(args):
    # Create the training logger
    logger = logging.getLogger('training_log')
    logger.setLevel(logging.INFO)

    logger_fn = os.path.join(args.log_dir, 'training_ver%s_%s.log' % (args.version, args.seed))
    fh = logging.FileHandler(logger_fn)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    console = logging.StreamHandler()

    logger.addHandler(fh)    
    logger.addHandler(console)
    logger.info('Code version %s, with random number generator seed: %s\n' % (args.version, args.seed))
