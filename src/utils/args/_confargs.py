from .._info import OPTIMIZERS


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
        'flags': ('-mopt', '--mod-optimizer'),
        'details': {
        'dest': 'mod_optim_algo',
        'nargs': '+',
        'type': str,
        'help': 'Optimization algorithm for each module of the neural network'
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-mga', '--mod-grad-accum'),
        'details': {
        'dest': 'mod_grad_accumulate',
        'nargs': '+',
        'type': str,
        'help': 'Steps to accumulate the gradient for each module of the '
                'neural network'
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-sch', '--scheduler'),
        'details': {
        'dest': 'scheduler_type',
        'type': str,
        'help': 'Learning rate scheduler for the optimizer method',
        'default': None
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-sch', '--scheduler'),
        'details': {
        'dest': 'mod_scheduler_algo',
        'nargs': '+',
        'type': str,
        'help': 'Learning rate scheduler for the optimizer method of each '
                'module of the neural network'
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
        'flags': ('-alr', '--aux-lrate'),
        'details': {
        'dest': 'aux_learning_rate',
        'type': float,
        'help': 'Auxiliar optimizer initial learning rate',
        'default': 1e-3
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-mlr', '--mod-lrate'),
        'details': {
        'dest': 'mod_learning_rate',
        'type': str,
        'nargs': '+',
        'help': 'Optimizer initial learning rate for each specific module (if '
                'not given, the general learning rate will be used instead)'
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-malr', '--mod-aux-lrate'),
        'details': {
        'dest': 'mod_aux_learning_rate',
        'type': float,
        'help': 'Optimizer initial learning rate for each specific auxiliar '
                'module (if not given, the general learning rate will be used '
                'instead)'
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
        'flags': ('-awd', '--aux-wdecay'),
        'details': {
        'dest': 'aux_weight_decay',
        'type': float,
        'help': 'Auxiliar optimizer weight decay (L2 regularizer)',
        'default': 0
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-mwd', '--mod-wdecay'),
        'details': {
        'dest': 'mod_weight_decay',
        'type': str,
        'nargs': '+',
        'help': 'Optimizer weight decay (L2 regularizer) for each specific '
                'module (if not given, the general learning rate will be used '
                'instead)'
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-mawd', '--mod-aux-wdecay'),
        'details': {
        'dest': 'mod_aux_weight_decay',
        'type': str,
        'nargs': '+',
        'help': 'Optimizer weight decay (L2 regularizer) for each specific '
                'auxiliar module (if not given, the general learning rate will'
                ' be used instead)'
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
        'dest': 'early_warmup',
        'type': int,
        'help': 'Early stopping warmup steps, i.e. number of steps to '
                'perform before starting to count bad evaluations',
        'default': -1
        }
    },
    {'tasks': ['all'],
        'modes': ['training'],
        'flags': ('-chk', '--checkpoint'),
        'details': {
        'dest': 'checkpoint',
        'type': str,
        'help': 'Resume training from an existing checkpoint',
        'default': None
        }
    },
]
