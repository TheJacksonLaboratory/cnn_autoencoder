import logging
import argparse
import os
import json

import utils
import compress
import decompress


DATASETS = ['MNIST', 'ImageNet', 'Histology']


def main(args):
    logger = logging.getLogger(args.mode + '_log')

    if args.use_zarr:        
        fn_list = [os.path.join(args.data_dir, fn) for fn in sorted(os.listdir(args.data_dir)) if fn.endswith('.zarr')]
        comp_list = [os.path.join(args.output_dir, 'rec_comp.zarr') for fn in sorted(os.listdir(args.data_dir)) if fn.endswith('.zarr')]

        fn_list = fn_list[:args.n_imgs]
        comp_list = comp_list[:args.n_imgs]
        
        # Compress the list of images
        args.input = fn_list
        compress.compress_zarr(args)
        logger.info('All %s images compressed successfully' % args.n_imgs)

        args.input = comp_list
        decompress.decompress_zarr(args)
        logger.info('All %s images decompressed successfully' % args.n_imgs)
        
    else:
        data_queue = utils.get_data(args, normalize=False)

        if args.dataset == 'MNIST':
            img_ext = 'pgm'
        elif args.dataset == 'ImageNet':
            img_ext = 'jpg'
        elif args.dataset == 'Histology':
            img_ext = 'png'
        else:
            raise ValueError('The dataset \'%s\' is not supported.' % args.dataset)

        # Export the images from the dataset and store them in a single directory
        fn_list = []
        comp_list = []
        for i, (x, _) in enumerate(data_queue):
            if i >= args.n_imgs:
                break

            logger.info('Input image {}, of size: {}'.format(i, x.size()))
            save_fn = os.path.join(args.output_dir, '{:03d}.{}'.format(i, img_ext))
            comp_fn = os.path.join(args.output_dir, '{:03d}_comp.zarr'.format(i))
            # comp_fn = os.path.join(args.output_dir, '{:03d}_comp.pth'.format(i))
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
    args = utils.get_testing_args()

    args.seed = -1
    args.batch_size = 1
    args.mode = 'models_testing'
    args.task = 'autoencoder'

    utils.setup_logger(args)

    main(args)