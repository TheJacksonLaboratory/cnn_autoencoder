import argparse

from .._info import DATASETS


data_args = [
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-ldg', '--labels-data-group'),
        'details': {
        'dest': 'label_data_group',
        'type': str,
        'help': 'For Zarr datasets, the group from where the lables are '
                'retrieved',
        'default': None
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-lda', '--labels-data-axes'),
        'details': {
        'dest': 'labels_data_axes',
        'type': str,
        'help': 'Order of the axes in which the labels are stored',
        'default': 'XYZCT'
        }
    },
    {'tasks': ['all'],
        'modes': ['training', 'test'],
        'flags': ('-ncl', '--num-classes'),
        'details': {
        'dest': 'num_classes',
        'type': int,
        'help': 'Number of classes n the classification task',
        'default': None
        }
    },
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-dg', '--data-group'),
        'details': {
        'dest': 'data_group',
        'type': str,
        'help': 'For Zarr datasets, the group from where the data is retrieved',
        'default': ''
        }
    },
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-dd', '--data-dir'),
        'details': {
        'dest': 'data_dir',
        'type': str,
        'help': 'Directory, list of files, or text file with a list of files '
                'to be used as inputs',
        'default': '0/0',
        'nargs': '+'
        }
    },
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-ps', '--patch-size'),
        'details': {
        'dest': 'patch_size',
        'type': int,
        'help': 'Size of the patch taken from the orignal image',
        'default': 128
        }
    },
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-nw', '--workers'),
        'details': {
        'dest': 'workers',
        'type': int,
        'help': 'Number of worker threads for distributed data loading',
        'default': 0
        }
    },
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-da', '--data-axes'),
        'details': {
        'dest': 'data_axes',
        'type': str,
        'help': 'Order of the axes in which the data is stored',
        'default': 'XYZCT'
        }
    },
    {'tasks': ['all'],
        'modes': ['test', 'inference'],
        'flags': ('-off', '--add-offset'),
        'details': {
        'dest': 'add_offset',
        'action': 'store_true',
        'help': 'Add offset to prevent stitching artifacts',
        'default': False
        }
    },
    {'tasks': ['decoder'],
        'modes': ['test', 'inference'],
        'flags': ('-of', '--dst-format'),
        'details': {
        'dest': 'destination_format',
        'type': str,
        'help': 'Format of the output file',
        'default': 'zarr'
        }
    },
    {'tasks': ['encoder'],
        'modes': ['test', 'inference'],
        'flags': ('-if', '--src-format'),
        'details': {
        'dest': 'source_format',
        'type': str,
        'help': 'Format of the source files',
        'default': 'zarr'
        }
    },
    {'tasks': ['encoder', 'decoder'],
        'modes': ['test', 'inference'],
        'flags': ('-md', '--mode-data'),
        'details': {
        'dest': 'data_mode',
        'type': str,
        'help': 'Mode of the dataset used to compute the metrics',
        'choices': ['train', 'val', 'test', 'all'],
        'default': 'all'
        }
    },
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-nor', '--normalize'),
        'details': {
        'dest': 'normalize',
        'action': 'store_true',
        'help': 'Normalize input to range [-1, 1]',
        'default': False
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-aed', '--elastic-def'),
        'details': {
        'dest': 'elastic_deformation',
        'action': 'store_true',
        'help': 'Use elastic deformation to augment the original data',
        'default': False
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-ar', '--rotation'),
        'details': {
        'dest': 'rotation',
        'action': 'store_true',
        'help': 'Augment the original data by rotating the inputs and their '
                'respectice targets',
        'default': False
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-wms', '--weight-map-sigma'),
        'details': {
        'dest': 'weight_map_sigma',
        'type': float,
        'help': 'Sigma value used to compute the weights map during training',
        'default': None
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-wmw', '--weight-map-w'),
        'details': {
        'dest': 'weight_map_w_0',
        'type': float,
        'help': 'Omega value used to compute the weights map during training',
        'default': None
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-an', '--noise'),
        'details': {
        'dest': 'add_noise',
        'action': 'store_true',
        'help': 'Augment the original data by adding Gaussian noise',
        'default': False
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-nshtr', '--no-shuffle-train'),
        'details': {
        'dest': 'shuffle_train',
        'action': 'store_true',
        'help': argparse.SUPPRESS,
        'default': True
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-shva', '--shuffle-val'),
        'details': {
        'dest': 'shuffle_val',
        'action': 'store_true',
        'help': 'Shuffle the validation set',
        'default': False
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-shva', '--shuffle-val'),
        'details': {
        'dest': 'shuffle_val',
        'action': 'store_true',
        'help': 'Shuffle the validation set',
        'default': False
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-ntr', '--num-train'),
        'details': {
        'dest': 'train_dataset_size',
        'type': int,
        'help': 'Size of set of images used to train the model',
        'default': -1
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-nva', '--num-val'),
        'details': {
        'dest': 'val_dataset_size',
        'type': int,
        'help': 'Size of set of images used to evaluate the model',
        'default': -1
        }
    },
    {'tasks': ['all'],
        'modes': ['test'],
        'flags': ('-shte', '--shuffle-test'),
        'details': {
        'dest': 'shuffle_test',
        'action': 'store_true',
        'help': 'Wether to shuffle the test set or not. Works for large images'
                ' where only small regions will be used to test the '
                'performance instead of whole images',
        'default': False
        }
    },
    {'tasks': ['all'],
        'modes': ['test'],
        'flags': ('-nte', '--num-test'),
        'details': {
        'dest': 'test_dataset_size',
        'type': int,
        'help': 'Size of set of test images used to evaluate the model',
        'default': -1
        }
    },
    {'tasks': ['all'],
        'modes': ['training', 'test'],
        'flags': ('-ds', '--dataset'),
        'details': {
        'dest': 'dataset',
        'type': str,
        'help': 'Dataset used for training the model',
        'choices': DATASETS,
        'default': DATASETS[0]
        }
    },
    {'tasks': ['all'],
        'modes': ['inference', 'test'],
        'flags': ('-o', '--output'),
        'details': {
        'dest': 'output_dir',
        'type': str,
        'help': 'Output directory, or list of filenames where to store the '
                'outputs',
        'nargs': '+',
        'default': '.'
        }
    },
    {'tasks': ['encoder', 'decoder'],
        'modes': ['test', 'inference'],
        'flags': ('-tli', '--task-label-identifier'),
        'details': {
        'dest': 'task_label_identifier',
        'type': str,
        'help': 'Name of the sub group used to store compression and '
                'reconstruction outputs',
        'default': ''
        }
    },
    {'tasks': ['decoder'],
        'modes': ['inference'],
        'flags': ('-rl', '--rec-level'),
        'details': {
        'dest': 'reconstruction_level',
        'type': int,
        'help': 'Level of reconstruction obtained from the compressed '
                'representation (must be <= compression level)',
        'default': -1
        }
    },
    {'tasks': ['decoder'],
        'modes': ['inference'],
        'flags': ('-pyr', '--store-pyramids'),
        'details': {
        'dest': 'compute_pyramids',
        'action': 'store_true',
        'help': 'Compute a pyramid representation of the image and store it in'
                ' the same file',
        'default': False
        }
    },
]