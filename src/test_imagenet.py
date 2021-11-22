import logging
import argparse
import json

from datasets import get_data
from utils import setup_logger

DATASETS = ['MNIST', 'ImageNet']


def main(args):
    logger = logging.getLogger(args.mode + '_log')
    data_queue = get_data(args, normalize=False)

    for i, (x, _) in enumerate(data_queue):
        if i > 10:
            break
        logger.info('Input image {}, of size: {}'.format(i, x.size()))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Testing of an image compression model')
    parser.add_argument('-c', '--config', type=str, dest='config_file', help='A configuration .json file')

    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)
    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
    
    parser.add_argument('-dd', '--datadir', type=str, dest='data_dir', help='Directory where the data is stored', default='.')
    parser.add_argument('-ds', '--dataset', type=str, dest='dataset', help='Dataset used for training the model', default='MNIST', choices=DATASETS)
    parser.add_argument('-dwn', '--download', dest='download_data', action='store_true', help='Download the dataset if it is not in the data directory', default=False)
    parser.add_argument('-bs', '--batch', type=int, dest='batch_size', help='Batch size for the training step', default=16)

    parser.add_argument('-n', '--nimgs', type=int, dest='n_imgs', help='Number of images to extract from the dataset and pass through the compression-decompression process', default=10)

    config_parser = argparse.ArgumentParser(parents=[parser], add_help=False)

    args = parser.parse_args()
    
    # Parse the arguments from a json configure file, when given
    if args.config_file is not None:
        if '.json' in args.config_file:
            config = json.load(open(args.config_file, 'r'))
            config_parser.set_defaults(**config)

        else:
            raise ValueError('The configure file must be a .json file')

    args = config_parser.parse_args()

    args.seed = -1
    args.mode = 'imagenet_testing'

    setup_logger(args)

    main(args)