import logging
import argparse
import os
import json

import utils
import models


DATASETS = ['Histology']


def main(args):
    logger = logging.getLogger(args.mode + '_log')

    data_queue = utils.get_data(args, normalize=False)

    if args.dataset == 'Histology':
        img_ext = 'png'
    else:
        raise ValueError('The dataset \'%s\' is not supported.' % args.dataset)

    # Export the images from the dataset and store them in a single directory
    fn_list = []
    comp_list = []
    for i, (x, _) in enumerate(data_queue):
        if i > args.n_imgs:
            break

        logger.info('Input image {}, of size: {}'.format(i, x.size()))

        save_fn = os.path.join(args.output_dir, '{:03d}.{}'.format(i, img_ext))
        comp_fn = os.path.join(args.output_dir, '{:03d}.zarr'.format(i))
        utils.save_image(save_fn, x)
        fn_list.append(save_fn)
        comp_list.append(comp_fn)

    # Compress the list of images
    args.input = fn_list
    compress.compress(args)
    logger.info('All %s images compressed successfully' % args.n_imgs)

    args.input = comp_list
    decompress.decompress(args)
    logger.info('All %s images decompressed successfully' % args.n_imgs)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Testing of an image compression model')
    parser.add_argument('-c', '--config', type=str, dest='config_file', help='A configuration .json file')

    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be tested')
    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)
    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
    
    parser.add_argument('-ps', '--patchsize', type=int, dest='patch_size', help='Size of the patch taken from the orignal image (set to -1 for default sizes given the dataset)', default=-1)
    parser.add_argument('-dd', '--datadir', type=str, dest='data_dir', help='Directory where the data is stored', default='.')
    parser.add_argument('-ds', '--dataset', type=str, dest='dataset', help='Dataset used for training the model', default=DATASETS[0], choices=DATASETS)
    parser.add_argument('-dwn', '--download', dest='download_data', action='store_true', help='Download the dataset if it is not in the data directory', default=False)

    parser.add_argument('-n', '--nimgs', type=int, dest='n_imgs', help='Number of images to extract from the dataset and pass through the compression-decompression process', default=10)
    parser.add_argument('-o', '--output', type=str, dest='output_dir', help='Output directory to store the compressed and decompressed files', default='.')
    parser.add_argument('-pth', '--store-pth', action='store_true', dest='store_pth', help='Store the compressed representation of the image before the arithmetic encoding inside a .pth file?')

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
    args.batch_size = 1
    args.mode = 'models_testing'
    args.task = 'segmentation'

    utils.setup_logger(args)

    main(args)