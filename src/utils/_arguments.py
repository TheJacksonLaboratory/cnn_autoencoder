import numpy as np
import torch

import json
import argparse

from ._info import DATASETS, SEG_MODELS, CAE_MODELS, PROJ_MODELS, FE_MODELS, CAE_CRITERIONS, SEG_CRITERIONS, SCHEDULERS


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

    return args


def get_training_args(task='autoencoder', parser_only=False):
    parser = argparse.ArgumentParser('Training of an image compression model based on a convolutional autoencoer')
    parser.add_argument('-c', '--config', dest='config_file', type=str, help='A configuration .json file')

    parser.add_argument('-rs', '--seed', type=int, dest='seed', help='Seed for random number generators', default=-1)
    parser.add_argument('-s', '--steps', type=int, dest='steps', help='Number of training steps', default=1e5)
    parser.add_argument('-cs', '--checksteps', type=int, dest='checkpoint_steps', help='Create a checkpoint every this number of steps', default=1e3)
    parser.add_argument('-esp', '--earlypatience', type=int, dest='patience', help='Early stopping patience, i.e. number of consecutive bad evaluations before stop training', default=5)
    parser.add_argument('-esw', '--earlywarmup', type=int, dest='warmup', help='Early stopping warmup steps, i.e. number of steps to perform before starting to count bad evaluations', default=1e4)
    
    parser.add_argument('-rm', '--resume', type=str, dest='resume', help='Resume training from an existing checkpoint')
    
    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)
    parser.add_argument('-ps', '--patchsize', type=int, dest='patch_size', help='Size of the patch taken from the orignal image', default=128)
    
    parser.add_argument('-ld', '--logdir', type=str, dest='log_dir', help='Directory where all logging and model checkpoints are stored', default='.')
    parser.add_argument('-li', '--logid', type=str, dest='log_identifier', help='Identifier added to the log file', default='')
    parser.add_argument('-dd', '--datadir', type=str, nargs='+', dest='data_dir', help='Directory where the data is stored. If giving lists of filenames, provide first the traing set, and then the validation set.', default='.')

    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
    parser.add_argument('-ds', '--dataset', type=str, dest='dataset', help='Dataset used for training the model', default=DATASETS[0], choices=DATASETS)

    parser.add_argument('-ich', '--inputch', type=int, dest='channels_org', help='Number of channels in the input data', default=3)
    parser.add_argument('-nch', '--netch', type=int, dest='channels_net', help='Number of channels in the analysis and synthesis tracks', default=8)
    parser.add_argument('-bch', '--bnch', type=int, dest='channels_bn', help='Number of channels of the compressed representation', default=16)
    parser.add_argument('-ech', '--expch', type=int, dest='channels_expansion', help='Rate of expansion of the number of channels in the analysis and synthesis tracks', default=1)
    
    parser.add_argument('-cl', '--compl', type=int, dest='compression_level', help='Level of compression', default=3)
    if task == 'autoencoder':
        parser.add_argument('-cr', '--criterion', type=str, dest='criterion', help='Training criterion for the compression evaluation', default=CAE_CRITERIONS[0], choices=CAE_CRITERIONS)
        parser.add_argument('-el', '--energylimit', type=float, dest='energy_limit', help='When using a penalty criterion, the maximum energy on the channel that consentrates the most of it is limited to this value', default=0.7)

        parser.add_argument('-dl', '--distl', type=float, nargs='+', dest='distorsion_lambda', help='Distorsion penalty parameter (lambda)', default=0.01)
        parser.add_argument('-eK', '--entK', type=int, dest='K', help='Number of layers in the latent space of the factorized entropy model', default=4)
        parser.add_argument('-er', '--entr', type=int, dest='r', help='Number of channels in the latent space of the factorized entropy model', default=3)

        parser.add_argument('-mt', '--model-type', type=str, dest='model_type', help='Type of autoencoder model', choices=CAE_MODELS)

        parser.add_argument('-nm', '--n-masks', type=int, dest='n_masks', help='Number of mask patches for the masked autoencoder training', default=5)
        parser.add_argument('-ms', '--masks-size', type=int, dest='masks_size', help='Standard size of the patched masks for the masked autoencoder training', default=4)
        
    elif task == 'segmentation':
        parser.add_argument('-cr', '--criterion', type=str, dest='criterion', help='Training criterion for the segmentation evaluation', default=SEG_CRITERIONS[0], choices=SEG_CRITERIONS)
        parser.add_argument('-tc', '--target-classes', type=int, dest='classes', help='Number of target classes', default=1)
        parser.add_argument('-mt', '--model-type', type=str, dest='model_type', help='Type of segmetnation model', choices=SEG_MODELS)
        parser.add_argument('-off', '--offset', action='store_true', dest='add_offset', help='Add offset to prevent stitching artifacts', default=False)
        parser.add_argument('-do', '--dropout', type=float, dest='dropout', help='Use drop out in the training stage', default=0.0)
        parser.add_argument('-dm', '--decoder-model', type=str, dest='autoencoder_model', help='A pretrained autoencoder model')

        parser.add_argument('-aed', '--elastic-def', action='store_true', dest='elastic_deformation', help='Use elastic deformation to augment the original data')
        parser.add_argument('-ar', '--rotation', action='store_true', dest='rotation', help='Augment the original data by rotating the inputs and their respectice targets')
    
    parser.add_argument('-bs', '--batch', type=int, dest='batch_size', help='Batch size for the training step', default=16)
    parser.add_argument('-vbs', '--valbatch', type=int, dest='val_batch_size', help='Batch size for the validation step', default=32)
    parser.add_argument('-lr', '--lrate', type=float, dest='learning_rate', help='Optimizer initial learning rate', default=1e-4)
    parser.add_argument('-sch', '--scheduler', type=str, dest='scheduler_type', help='Learning rate scheduler for the optimizer method', default='None', choices=SCHEDULERS)
    parser.add_argument('-wd', '--wdecay', type=float, dest='weight_decay', help='Optimizer weight decay (L2 regularizer)', default=0)

    parser.add_argument('-g', '--gpu', action='store_true', dest='use_gpu', help='Use GPU when available')
    
    if parser_only:
        return parser

    args = override_config_file(parser)

    args.mode = 'training'
    args.task = task

    return args


def get_testing_args(parser_only=False):
    parser = argparse.ArgumentParser('Testing of an image compression-decompression model')
    parser.add_argument('-c', '--config', type=str, dest='config_file', help='A configuration .json file')
    
    parser.add_argument('-rs', '--seed', type=int, dest='seed', help='Seed for random number generators', default=-1)
    parser.add_argument('-mt', '--model-type', type=str, dest='model_type', help='Type of autoencoder model', choices=CAE_MODELS)
    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be tested')
    parser.add_argument('-ld', '--logdir', type=str, dest='log_dir', help='Directory where all logging and model checkpoints are stored', default='.')
    parser.add_argument('-li', '--logid', type=str, dest='log_identifier', help='Identifier added to the log file', default='')
    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)
    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
    
    parser.add_argument('-bs', '--batch', type=int, dest='batch_size', help='Batch size for the testing step', default=16)
    parser.add_argument('-ps', '--patchsize', type=int, dest='patch_size', help='Size of the patch taken from the orignal image', default=128)
    parser.add_argument('-dd', '--datadir', type=str, dest='data_dir', help='Directory where the data is stored', default='.')
    parser.add_argument('-ds', '--dataset', type=str, dest='dataset', help='Dataset used for training the model', default=DATASETS[0], choices=DATASETS)
    parser.add_argument('-if', '--src-format', type=str, dest='source_format', help='Format of the source files to compress', default='zarr')

    parser.add_argument('-md', '--mode-data', type=str, dest='mode_data', help='Mode of the dataset used to compute the metrics', choices=['train', 'va', 'test', 'all'], default='all')
    parser.add_argument('-sht', '--shuffle-test', action='store_true', dest='shuffle_test', help='Shuffle the test set? Works for large images where only small regions will be used to test the performance instead of whole images.')
    parser.add_argument('-nt', '--num-test', type=int, dest='test_size', help='Size of set of test images used to evaluate the model.', default=-1)

    parser.add_argument('-g', '--gpu', action='store_true', dest='use_gpu', help='Use GPU when available')
    
    parser.add_argument('-l', '--labeled', action='store_true', dest='is_labeled', help='Store the labels along woth the compressed representation (when provided)', default=False)
    
    if parser_only:
        return parser

    args = override_config_file(parser)
    
    args.mode = 'testing'

    return args


def get_fact_ent_args(parser_only=False):
    parser = argparse.ArgumentParser('Compute the factorized entropy of the compressed representation of an image')
    parser.add_argument('-c', '--config', type=str, dest='config_file', help='A configuration .json file')

    parser.add_argument('-rs', '--seed', type=int, dest='seed', help='Seed for random number generators', default=-1)
    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be tested')
    parser.add_argument('-mt', '--model-type', type=str, dest='model_type', help='Type of segmetnation model', choices=FE_MODELS)

    parser.add_argument('-bs', '--batch', type=int, dest='batch_size', help='Patched compression in batches if this size', default=1)
    parser.add_argument('-nsb', '--no-stitching', action='store_false', dest='stitch_batches', help='Stitch the batches into a single image?', default=True)
    
    parser.add_argument('-li', '--logid', type=str, dest='log_identifier', help='Identifier added to the log file', default='')
    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)
    parser.add_argument('-ps', '--patchsize', type=int, dest='patch_size', help='Size of the patch taken from the orignal image', default=512)
    
    parser.add_argument('-off', '--offset', action='store_true', dest='add_offset', help='Add offset to prevent stitching artifacts', default=False)

    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
    parser.add_argument('-i', '--input', type=str, nargs='+', dest='input', help='Input images to compress (list of images).')
    parser.add_argument('-o', '--output', type=str, dest='output_dir', help='Output directory to store the compressed image')

    parser.add_argument('-if', '--src-format', type=str, dest='source_format', help='Format of the source files to compress', default='zarr')
    parser.add_argument('-of', '--dst-format', type=str, dest='destination_format', help=argparse.SUPPRESS, default='zarr')

    parser.add_argument('-g', '--gpu', action='store_true', dest='use_gpu', help='Use GPU when available')

    if parser_only:
        return parser

    args = override_config_file(parser)
    
    args.mode = 'fact_ent'

    return args


def get_compress_args(parser_only=False):
    parser = argparse.ArgumentParser('Testing of an image compression model')
    parser.add_argument('-c', '--config', type=str, dest='config_file', help='A configuration .json file')

    parser.add_argument('-rs', '--seed', type=int, dest='seed', help='Seed for random number generators', default=-1)
    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be tested')
    parser.add_argument('-bs', '--batch', type=int, dest='batch_size', help='Patched compression in batches if this size', default=1)
    parser.add_argument('-nsb', '--no-stitching', action='store_false', dest='stitch_batches', help='Stitch the batches into a single image?', default=True)
    
    parser.add_argument('-li', '--logid', type=str, dest='log_identifier', help='Identifier added to the log file', default='')
    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)
    parser.add_argument('-ps', '--patchsize', type=int, dest='patch_size', help='Size of the patch taken from the orignal image', default=512)
    parser.add_argument('-l', '--labeled', action='store_true', dest='is_labeled', help='Store the labels along woth the compressed representation (when provided)', default=False)
    
    parser.add_argument('-off', '--offset', action='store_true', dest='add_offset', help='Add offset to prevent stitching artifacts', default=False)

    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
    parser.add_argument('-i', '--input', type=str, nargs='+', dest='input', help='Input images to compress (list of images).')
    parser.add_argument('-o', '--output', type=str, dest='output_dir', help='Output directory to store the compressed image')

    parser.add_argument('-if', '--src-format', type=str, dest='source_format', help='Format of the source files to compress', default='zarr')
    parser.add_argument('-of', '--dst-format', type=str, dest='destination_format', help=argparse.SUPPRESS, default='zarr')

    parser.add_argument('-g', '--gpu', action='store_true', dest='use_gpu', help='Use GPU when available')

    if parser_only:
        return parser

    args = override_config_file(parser)
    
    args.mode = 'compress'

    return args


def get_decompress_args(parser_only=False):
    parser = argparse.ArgumentParser('Testing of an image decompression model')
    parser.add_argument('-c', '--config', type=str, dest='config_file', help='A configuration .json file')

    parser.add_argument('-rs', '--seed', type=int, dest='seed', help='Seed for random number generators', default=-1)
    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be tested')
    parser.add_argument('-bs', '--batch', type=int, dest='batch_size', help='Patched decompression in batches if this size', default=1)
    parser.add_argument('-nsb', '--no-stitching', action='store_false', dest='stitch_batches', help='Stitch the batches into a single image?', default=True)
    
    parser.add_argument('-li', '--logid', type=str, dest='log_identifier', help='Identifier added to the log file', default='') 
    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)
    parser.add_argument('-ps', '--patchsize', type=int, dest='patch_size', help='Size of the patch taken from the orignal image', default=512)

    parser.add_argument('-off', '--offset', action='store_true', dest='add_offset', help='Add offset to prevent stitching artifacts', default=False)
    
    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
    parser.add_argument('-i', '--input', type=str, nargs='+', dest='input', help='Input compressed images (list of .pth files)')
    parser.add_argument('-o', '--output', type=str, dest='output_dir', help='Output directory to store the decompressed image')
    
    parser.add_argument('-if', '--src-format', type=str, dest='source_format', help=argparse.SUPPRESS, default='zarr')
    parser.add_argument('-of', '--dst-format', type=str, dest='destination_format', help='Format of the output image', default='zarr')

    parser.add_argument('-rl', '--rec-level', type=int, dest='reconstruction_level', help='Level of reconstruction obtained from the compressed representation (<=compression level)', default=-1)
    parser.add_argument('-pyr', '--pyramids', action='store_true', dest='compute_pyramids', help='Compute a pyramid representation of the image and store it in the same file', default=False)

    parser.add_argument('-g', '--gpu', action='store_true', dest='use_gpu', help='Use GPU when available')
    
    if parser_only:
        return parser
        
    args = override_config_file(parser)

    args.mode = 'decompress'

    return args


def get_segment_args(parser_only=False):
    parser = argparse.ArgumentParser('Testing of an image segmentation model')
    
    parser.add_argument('-c', '--config', type=str, dest='config_file', help='A configuration .json file')

    parser.add_argument('-rs', '--seed', type=int, dest='seed', help='Seed for random number generators', default=-1)
    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be tested')
    parser.add_argument('-dm', '--decoder-model', type=str, dest='autoencoder_model', help='A pretrained autoencoder model')
    parser.add_argument('-bs', '--batch', type=int, dest='batch_size', help='Patched segmentation in batches if this size', default=1)
    parser.add_argument('-nsb', '--no-stitching', action='store_false', dest='stitch_batches', help='Stitch the batches into a single image?', default=True)
    
    parser.add_argument('-li', '--logid', type=str, dest='log_identifier', help='Identifier added to the log file', default='')
    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)
    parser.add_argument('-ps', '--patchsize', type=int, dest='patch_size', help='Size of the patch taken from the orignal image', default=512)

    parser.add_argument('-off', '--offset', action='store_true', dest='add_offset', help='Add offset to prevent stitching artifacts', default=False)
    parser.add_argument('-l', '--labeled', action='store_true', dest='is_labeled', help='Store the labels along woth the compressed representation (when provided)', default=False)
    
    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
    parser.add_argument('-i', '--input', type=str, nargs='+', dest='input', help='Input compressed images (list of .pth files)')
    parser.add_argument('-o', '--output', type=str, dest='output_dir', help='Output directory to store the decompressed image')

    parser.add_argument('-if', '--src-format', type=str, dest='source_format', help='Format of the source files to compress', default='zarr')
    parser.add_argument('-of', '--dst-format', type=str, dest='destination_format', help='Format of the output image', default='zarr')

    parser.add_argument('-g', '--gpu', action='store_true', dest='use_gpu', help='Use GPU when available')
    
    if parser_only:
        return parser
    
    args = override_config_file(parser)

    args.mode = 'segment'

    return args


def get_project_args(parser_only=False):
    parser = argparse.ArgumentParser('Testing of an image kernel projection model')
    
    parser.add_argument('-c', '--config', type=str, dest='config_file', help='A configuration .json file')

    parser.add_argument('-rs', '--seed', type=int, dest='seed', help='Seed for random number generators', default=-1)
    
    parser.add_argument('-ld', '--logdir', type=str, dest='log_dir', help='Directory where all logging and model checkpoints are stored', default='.')
    parser.add_argument('-li', '--logid', type=str, dest='log_identifier', help='Identifier added to the log file', default='')
    
    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be tested')
    parser.add_argument('-mt', '--model-type', type=str, dest='model_type', help='Type of autoencoder model', choices=PROJ_MODELS, default=PROJ_MODELS[0])
    parser.add_argument('-np', '--num-projs', type=int, dest='num_projections', help='Number of projections', default=1000)
    parser.add_argument('-ich', '--inputch', type=int, dest='channels_org', help='Number of channels in the input data', default=3)

    parser.add_argument('-bs', '--batch', type=int, dest='batch_size', help='Patched segmentation in batches if this size', default=1)
    
    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)
    parser.add_argument('-ps', '--patchsize', type=int, dest='patch_size', help='Size of the patch taken from the orignal image', default=512)

    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)

    parser.add_argument('-i', '--input', type=str, nargs='+', dest='input', help='Input compressed images (list of .pth files)')
    parser.add_argument('-o', '--output', type=str, dest='output_dir', help='Output directory to store the decompressed image')

    parser.add_argument('-g', '--gpu', action='store_true', dest='use_gpu', help='Use GPU when available')
    
    if parser_only:
        return parser
    
    args = override_config_file(parser)

    args.task = 'projection'
    args.mode = 'project'

    return args
