from ast import arg
import numpy as np
import torch

import json
import argparse

from ._info import DATASETS, SEG_MODELS, CAE_MODELS, PROJ_MODELS, FE_MODELS, CLASS_MODELS, CAE_CRITERIONS, SEG_CRITERIONS, SCHEDULERS, MERGE_TYPES


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


def add_logging_args(parser):
    parser.add_argument('-rs', '--seed', type=int, dest='seed', help='Seed for random number generators', default=-1)

    parser.add_argument('-pl', '--printlog', dest='print_log', action='store_true', help='Print log into console (Not recommended when running on clusters).', default=False)
    parser.add_argument('-ld', '--logdir', type=str, dest='log_dir', help='Directory where all logging and model checkpoints are stored', default='.')
    parser.add_argument('-li', '--logid', type=str, dest='log_identifier', help='Identifier added to the log file', default='')


def add_data_args(parser, task, mode='training'):
    if task in ['classifier', 'segmentation']:
        parser.add_argument('-lg', '--labels-group', type=str, dest='labels_group', help='For Zarr datasets, the group from where the lables are retrieved', default='labels/0/0')
        parser.add_argument('-lda', '--labels-data-axes', type=str, dest='labels_data_axes', help='Order of the axes in which the labels are stored. For 5 channels: XYZCT')


    if task in ['decoder']:
        parser.add_argument('-dg', '--data-group', type=str, dest='data_group', help='For Zarr datasets, the group from where the data is retrieved', default='compressed/0/0')
    else:
        parser.add_argument('-dg', '--data-group', type=str, dest='data_group', help='For Zarr datasets, the group from where the data is retrieved', default='0/0')

    parser.add_argument('-dd', '--datadir', type=str, nargs='+', dest='data_dir', help='Directory, list of files, or text file with a list of files to be used as inputs.')

    if task in ['encoder', 'decoder', 'classifier', 'segmentation', 'autoencoder']:
        parser.add_argument('-ps', '--patchsize', type=int, dest='patch_size', help='Size of the patch taken from the orignal image', default=128)
        parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of worker threads', default=0)
        parser.add_argument('-da', '--data-axes', type=str, dest='data_axes', help='Order of the axes in which the data is stored. For 5 channels: XYZCT', default='XYZCT')

    if mode == 'inference':
        parser.add_argument('-off', '--offset', action='store_true', dest='add_offset', help='Add offset to prevent stitching artifacts', default=False)
        if task in ['decoder', 'segmentation']:
            parser.add_argument('-of', '--dst-format', type=str, dest='destination_format', help='Format of the destination files', default='zarr')
        if task in ['encoder', 'segmentation' 'segmentation']:
            parser.add_argument('-if', '--src-format', type=str, dest='source_format', help='Format of the source files to compress', default='zarr')

    if mode in ['trainig', 'test']:
        parser.add_argument('-md', '--mode-data', type=str, dest='data_mode', help='Mode of the dataset used to compute the metrics', choices=['train', 'va', 'test', 'all'], default='all')

    if mode == 'training':
        parser.add_argument('-aed', '--elastic-def', action='store_true', dest='elastic_deformation', help='Use elastic deformation to augment the original data')
        parser.add_argument('-ar', '--rotation', action='store_true', dest='rotation', help='Augment the original data by rotating the inputs and their respectice targets')

        parser.add_argument('-shtr', '--shuffle-train', action='store_true', dest='shuffle_train', help='Shuffle the training set?')
        parser.add_argument('-shva', '--shuffle-val', action='store_true', dest='shuffle_val', help='Shuffle the validation set?')
        parser.add_argument('-ntr', '--num-train', type=int, dest='train_dataset_size', help='Size of set of test images used to evaluate the model.', default=-1)
        parser.add_argument('-nva', '--num-val', type=int, dest='val_dataset_size', help='Size of set of test images used to evaluate the model.', default=-1)

    elif mode == 'test':
        parser.add_argument('-shte', '--shuffle-test', action='store_true', dest='shuffle_test', help='Shuffle the test set? Works for large images where only small regions will be used to test the performance instead of whole images.')
        parser.add_argument('-nte', '--num-test', type=int, dest='test_dataset_size', help='Size of set of test images used to evaluate the model.', default=-1)

    if mode in ['training', 'test']:
        parser.add_argument('-ds', '--dataset', type=str, dest='dataset', help='Dataset used for training the model', default=DATASETS[0], choices=DATASETS)

    if task in ['encoder', 'decoder', 'arithmetic_encoder', 'segmentation']:
        parser.add_argument('-o', '--output', type=str, nargs='+', dest='output_dir', help='Output directory, or list of filenames where to store the compressed image')
        parser.add_argument('-ci', '--identifier', type=str, dest='comp_identifier', help='Identifier added as suffix to the output filename of a compression/decompression process', default='')

    # TODO: Probably a general label identifier for any task that stores
    # annotations into a zarr file. The output could be inserted in the same
    # zarr file this way.
    if task in ['segmentation']:        
        parser.add_argument('-tli', '--task-label-identifier', type=str, dest='task_label_identifier', help='Name of the sub group inside the labels gorup where to store the segmentation', default='segmentation')

    if task in ['encoder']:
        parser.add_argument('-tli', '--task-label-identifier', type=str, dest='task_label_identifier', help='Name of the sub group where to store the compressed representation', default='compressed')

    if task in ['decoder']:
        parser.add_argument('-tli', '--task-label-identifier', type=str, dest='task_label_identifier', help='Name of the sub group where to store the reconstructed image', default='reconstruction')
        parser.add_argument('-rl', '--rec-level', type=int, dest='reconstruction_level', help='Level of reconstruction obtained from the compressed representation (<=compression level)', default=-1)
        parser.add_argument('-pyr', '--pyramids', action='store_true', dest='compute_pyramids', help='Compute a pyramid representation of the image and store it in the same file', default=False)


def add_config_args(parser, mode=True):
    parser.add_argument('-bs', '--batch', type=int, dest='batch_size', help='Batch size for the training step', default=16)

    if mode not in 'training': return
    parser.add_argument('-vbs', '--valbatch', type=int, dest='val_batch_size', help='Batch size for the validation step', default=32)
    parser.add_argument('-lr', '--lrate', type=float, dest='learning_rate', help='Optimizer initial learning rate', default=1e-4)
    parser.add_argument('-sch', '--scheduler', type=str, dest='scheduler_type', help='Learning rate scheduler for the optimizer method', default='None', choices=SCHEDULERS)
    parser.add_argument('-wd', '--wdecay', type=float, dest='weight_decay', help='Optimizer weight decay (L2 regularizer)', default=0)

    parser.add_argument('-s', '--steps', type=int, dest='steps', help='Number of training steps', default=1e5)
    parser.add_argument('-cs', '--checksteps', type=int, dest='checkpoint_steps', help='Create a checkpoint every this number of steps', default=1e3)
    parser.add_argument('-esp', '--earlypatience', type=int, dest='patience', help='Early stopping patience, i.e. number of consecutive bad evaluations before stop training', default=5)
    parser.add_argument('-esw', '--earlywarmup', type=int, dest='warmup', help='Early stopping warmup steps, i.e. number of steps to perform before starting to count bad evaluations', default=1e4)

    parser.add_argument('-rm', '--resume', type=str, dest='resume', help='Resume training from an existing checkpoint')

    parser.add_argument('-ich', '--inputch', type=int, dest='channels_org', help='Number of channels in the input data', default=3)
    parser.add_argument('-nch', '--netch', type=int, dest='channels_net', help='Number of channels in the analysis and synthesis tracks', default=8)
    parser.add_argument('-bch', '--bnch', type=int, dest='channels_bn', help='Number of channels of the compressed representation', default=16)
    parser.add_argument('-ech', '--expch', type=int, dest='channels_expansion', help='Rate of expansion of the number of channels in the analysis and synthesis tracks', default=1)

    parser.add_argument('-cl', '--compl', type=int, dest='compression_level', help='Level of compression', default=3)


def add_model_args(parser, task, mode=True):
    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be used')

    if task in ['segmentation']:
        parser.add_argument('-dm', '--decoder-model', type=str, dest='autoencoder_model', help='A pretrained autoencoder model')
        parser.add_argument('-st', '--segmentation-threshold', type=float, dest='seg_threshold', help='Objects will be assigned to their corresponding class if those have a predicted confidence higher than this threshold value', default=0.5)

    if task == 'autoencoder':
        model_choices = CAE_MODELS
        if mode not in ['training']:
            parser.add_argument('-eK', '--entK', type=int, dest='K', help='Number of layers in the latent space of the factorized entropy model', default=4)
            parser.add_argument('-er', '--entr', type=int, dest='r', help='Number of channels in the latent space of the factorized entropy model', default=3)
            parser.add_argument('-nm', '--n-masks', type=int, dest='n_masks', help='Number of mask patches for the masked autoencoder training', default=5)
            parser.add_argument('-ms', '--masks-size', type=int, dest='masks_size', help='Standard size of the patched masks for the masked autoencoder training', default=4)

    elif task == 'classifier':
        model_choices = CLASS_MODELS
        if mode not in ['training']:
            parser.add_argument('-pt', '--pre-trained', action='store_true', dest='pretrained', help='Use a pretrained model')
            parser.add_argument('-cns', '--consensus', action='store_true', dest='consensus', help='Use consensus to define the models\'s predicted class')
            parser.add_argument('-mrg', '--merge-labels', type=str, dest='merge_labels', help='Merge the labels spatialy to define the class of a patch', choices=MERGE_TYPES)

    elif task == 'segmentation':
        model_choices = SEG_MODELS
        if mode in ['training']:
            parser.add_argument('-tc', '--target-classes', type=int, dest='classes', help='Number of target classes', default=1)
            parser.add_argument('-do', '--dropout', type=float, dest='dropout', help='Use drop out in the training stage', default=0.0)
        parser.add_argument('-thr', '--prediction-threshold', type=float, dest='prediction_threshold', help=argparse.SUPPRESS, default=0.5)

    elif task == 'projection':
        model_choices = PROJ_MODELS

    elif task == 'fact_ent':
        model_choices = FE_MODELS

    if mode in ['training']:
        parser.add_argument('-mt', '--model-type', type=str, dest='model_type', help='Type of %s model' % task, choices=model_choices)


def add_criteria_args(parser, task, mode=True):
    if mode not in ['training']:
        return

    if task == 'autoencoder':
        criteria_choices = CAE_CRITERIONS
        parser.add_argument('-el', '--energylimit', type=float, dest='energy_limit', help='When using a penalty criterion, the maximum energy on the channel that consentrates the most of it is limited to this value', default=0.7)
        parser.add_argument('-dl', '--distl', type=float, nargs='+', dest='distorsion_lambda', help='Distorsion penalty parameter (lambda)', default=0.01)
    elif task in ['classifier', 'segmentation']:
        criteria_choices = SEG_CRITERIONS
    else:
        raise ValueError('Task %s not supported' % task)

    parser.add_argument('-cr', '--criterion', type=str, dest='criterion', help='Training criterion for the compression evaluation', choices=criteria_choices)


def get_args(task, mode, add_model=True, add_criteria=True, add_config=True, add_data=True, add_logging=True, parser_only=False):
    parser = argparse.ArgumentParser('Arguments for running ' + task + ' in mode ' +  mode, conflict_handler='resolve')
    parser.add_argument('-c', '--config', dest='config_file', type=str, help='A configuration .json file')
    parser.add_argument('-g', '--gpu', action='store_true', dest='use_gpu', help='Use GPU when available')

    if add_model:
        add_model_args(parser, task, mode)

    if add_criteria:
        add_criteria_args(parser, task, mode)

    if add_config:
        add_config_args(parser, mode)

    if add_data:
        add_data_args(parser, task, mode)

    if add_logging:
        add_logging_args(parser)

    if parser_only:
        return parser

    args = override_config_file(parser)

    args.mode = mode
    args.task = task

    return args
