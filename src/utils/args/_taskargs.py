from .._info import CAE_MODELS, CLASS_MODELS, CAE_ACT_LAYERS


task_args = [
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-m', '--model'),
        'details': {
        'dest': 'trained_model',
        'type': str,
        'help': 'The checkpoint of the model to be used',
        'default': None
        }
    },
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-cm', '--compressed-model'),
        'details': {
        'dest': 'trained_model',
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
        'choices': CAE_ACT_LAYERS,
        'default': CAE_ACT_LAYERS[0]
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
        'choices': CAE_MODELS,
        'default': CAE_MODELS[0],
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-cmt', '--class-model-type'),
        'details': {
        'dest': 'class_model_type',
        'type': str,
        'help': 'Classifier model type',
        'choices': CLASS_MODELS,
        'default': CLASS_MODELS[0],
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-ccp', '--class-cut-poisition'),
        'details': {
        'dest': 'cut_poisition',
        'type': int,
        'help': 'Position on the architecture where to insert the bottleneck',
        'default': None
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
]