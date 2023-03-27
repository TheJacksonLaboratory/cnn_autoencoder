task_args = [
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-cm', '--compressed-model'),
        'details': {
        'dest': 'compressed_trained_model',
        'type': str,
        'help': 'The checkpoint of the model to be used',
        'default': None
        }
    },
    {'tasks': ['autoencoder', 'lc-compress'],
        'modes': ['training'],
        'flags': ('-eK', '--entropy-K'),
        'details': {
        'dest': 'K',
        'type': int,
        'help': 'Number of layers in the latent space of the factorized '
                'entropy model',
        'default': 4
        }
    },
    {'tasks': ['autoencoder', 'lc-compress'],
        'modes': ['training'],
        'flags': ('-er', '--entropy-r'),
        'details': {
        'dest': 'r',
        'type': int,
        'help': 'Number of channels in the latent space of the factorized '
                'entropy model',
        'default': 3
        }
    },
    {'tasks': ['autoencoder', 'lc-compress'],
        'modes': ['training'],
        'flags': ('-res', '--use-residual'),
        'details': {
        'dest': 'use_residual',
        'action': 'store_true',
        'help': 'Use residual blocks',
        'default': False
        }
    },
    {'tasks': ['autoencoder', 'lc-compress'],
        'modes': ['training'],
        'flags': ('-act', '--activation-type'),
        'details': {
        'dest': 'act_layer_type',
        'type': str,
        'help': 'Type of activation layer used for the Convolutional '
                'Autoencoder architecture',
        'default': 'LeakyRelU'
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-ich', '--input-channels'),
        'details': {
        'dest': 'channels_org',
        'type': int,
        'help': 'Number of channels in the input data',
        'default': 3
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-nch', '--net-channels'),
        'details': {
        'dest': 'channels_net',
        'type': int,
        'help': 'Number of channels in the analysis and synthesis tracks',
        'default': 128
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-bch', '--bottleneck-channels'),
        'details': {
        'dest': 'channels_bn',
        'type': int,
        'help': 'Number of channels of the compressed representation',
        'default': 48
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-ech', '--expansion-channels'),
        'details': {
        'dest': 'channels_expansion',
        'type': int,
        'help': 'Rate of expansion of the number of channels in the '
                'analysis and synthesis tracks',
        'default': 1
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-cl', '--compression-level'),
        'details': {
        'dest': 'compression_level',
        'type': int,
        'help': 'Level of compression (number of compression layers)',
        'default': 3
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-mt', '--model-type'),
        'details': {
        'dest': 'model_type',
        'type': str,
        'help': 'Convoutional Autoencoder model type',
        'default': 'AutoEncoder',
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-cmt', '--class-model-type'),
        'details': {
        'dest': 'class_model_type',
        'type': str,
        'help': 'Classifier model type',
        'default': None,
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-ccp', '--class-cut-poisition'),
        'details': {
        'dest': 'cut_position',
        'type': int,
        'help': 'Position on the architecture where to insert the bottleneck',
        'default': None
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-smt', '--seg-model-type'),
        'details': {
        'dest': 'seg_model_type',
        'type': str,
        'help': 'Segmentation model type',
        'default': None,
        }
    },
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-bn', '--batch-norm'),
        'details': {
        'dest': 'batch_norm',
        'action': 'store_true',
        'help': 'Add batch normalization layers',
        'default': False,
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-do', '--dropout'),
        'details': {
        'dest': 'dropout',
        'type': float,
        'help': 'Drop out for the training stages',
        'default': 0.0,
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-trm', '--trainable-modules'),
        'details': {
        'dest': 'trainable_modules',
        'type': str,
        'nargs': '+',
        'help': 'Trainable modules of the network. All modules not listed here'
                ' will set as evaluation mode and their parameters will not be'
                ' tracked by the optimizer',
        'default': ['encoder', 'decoder', 'fact_ent', 'class_model'],
        }
    },
    {'tasks': ['all'],
        'modes': ['test'],
        'flags': ('-thr', '--threshold'),
        'details': {
        'dest': 'seg_threshold',
        'type': float,
        'help': 'Confidence threshold use to determine if a pixel has been '
                'segmented into the corresponding class',
        'default': 0.5,
        }
    },
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-clsw', '--class-weights'),
        'details': {
        'dest': 'class_weights',
        'nargs': '+',
        'type': float,
        'help': 'Weight associated to each class to balance its contribution '
                'to the loss function',
        'default': None,
        }
    },
    {'tasks': ['all'],
        'modes': ['test'],
        'flags': ('-spr', '--save-prediction'),
        'details': {
        'dest': 'save_prediction',
        'action': 'store_true',
        'help': 'Store the thresholded prediction from the segmentation head',
        'default': False,
        }
    },
]