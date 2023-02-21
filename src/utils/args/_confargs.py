from .._info import OPTIMIZERS, SCHEDULERS


config_args = [
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-bs', '--batch'),
        'details': {
        'dest': 'batch_size',
        'type': int,
        'help': 'Batch size for the training step',
        'default': 16
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-vbs', '--valbatch'),
        'details': {
        'dest': 'val_batch_size',
        'type': int,
        'help': 'Batch size for the validation step',
        'default': 32
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-lr', '--lrate'),
        'details': {
        'dest': 'learning_rate',
        'type': float,
        'help': 'Optimizer initial learning rate',
        'default': 1e-4
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-alr', '--auxlrate'),
        'details': {
        'dest': 'aux_learning_rate',
        'type': float,
        'help': 'Auxiliar optimizer initial learning rate',
        'default': 1e-3
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-opt', '--optimizer'),
        'details': {
        'dest': 'optim_algo',
        'type': str,
        'help': 'Optimization algorithm',
        'default': OPTIMIZERS[0],
        'choices': OPTIMIZERS
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-sch', '--scheduler'),
        'details': {
        'dest': 'scheduler_type',
        'type': str,
        'help': 'Learning rate scheduler for the optimizer method',
        'default': SCHEDULERS[0],
        'choices': SCHEDULERS
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-wd', '--wdecay'),
        'details': {
        'dest': 'weight_decay',
        'type': float,
        'help': 'Optimizer weight decay (L2 regularizer)',
        'default': 0
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-awd', '--auxwdecay'),
        'details': {
        'dest': 'aux_weight_decay',
        'type': float,
        'help': 'Auxiliar optimizer weight decay (L2 regularizer)',
        'default': 0
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-s', '--steps'),
        'details': {
        'dest': 'steps',
        'type': int,
        'help': 'Number of training steps',
        'default': 10000
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-ss', '--sub-steps'),
        'details': {
        'dest': 'sub_iter_steps',
        'type': int,
        'help': 'Number of steps for sub iteration (on penalty based '
                'training)',
        'default': 100
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-cs', '--checksteps'),
        'details': {
        'dest': 'checkpoint_steps',
        'type': int,
        'help': 'Create a checkpoint every this number of steps',
        'default': 1000
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-esp', '--early-patience'),
        'details': {
        'dest': 'early_patience',
        'type': int,
        'help': 'Early stopping patience, i.e. number of consecutive bad '
                'evaluations before stop training',
        'default': 100
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-esw', '--early-warmup'),
        'details': {
        'dest': 'early_patience',
        'type': int,
        'help': 'Early stopping warmup steps, i.e. number of steps to '
                'perform before starting to count bad evaluations',
        'default': -1
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-chk', '--resume-checkpoint'),
        'details': {
        'dest': 'resume',
        'type': str,
        'help': 'Resume training from an existing checkpoint',
        'default': None
        }
    },
]
