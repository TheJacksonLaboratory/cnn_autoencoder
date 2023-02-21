logging_args = [
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-rs', '--seed'),
        'details': {
        'dest': 'seed',
        'type': int,
        'help': 'Seed for random number generators',
        'default': -1
        }
    },
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-pl', '--printlog'),
        'details': {
        'dest': 'print_log',
        'action': 'store_true',
        'help': 'Print log into console (Not recommended when running on '
                'clusters)',
        'default': False
        }
    },
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-pb', '--progress-bar'),
        'details': {
        'dest': 'progress_bar',
        'action': 'store_true',
        'help': 'Show progress bar (Not recommended when running on '
                'clusters)',
        'default': False
        }
    },
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-ld', '--logdir'),
        'details': {
        'dest': 'log_dir',
        'type': str,
        'help': 'Directory where all logging and model checkpoints are '
                'stored',
        'default': '.'
        }
    },
    {'tasks': ['all'],
        'modes': ['all'],
        'flags': ('-li', '--logid'),
        'details': {
        'dest': 'log_identifier',
        'type': str,
        'help': 'Identifier added to the log file',
        'default': ''
        }
    },
]