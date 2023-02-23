criteria_args = [
    {'tasks': ['autoencoder', 'lc-compress'],
        'modes': ['training'],
        'flags': ('-el', '--energylimit'),
        'details': {
        'dest': 'energy_limit',
        'type': float,
        'help': 'When using a penalty criterion, the maximum energy on the '
                'channel that consentrates the most of it is limited to this '
                'value',
        'default': None
        }
    },
    {'tasks': ['autoencoder', 'lc-compress'],
        'modes': ['training'],
        'flags': ('-dl', '--distortion-lambda'),
        'details': {
        'dest': 'distortion_lambda',
        'type': float,
        'help': 'Distortion penalty parameter (lambda)',
        'nargs': '+',
        'default': 0.01
        }
    },
    {'tasks': ['autoencoder', 'lc-compress'],
        'modes': ['training'],
        'flags': ('-b', '--penalty-beta'),
        'details': {
        'dest': 'penalty_beta',
        'type': float,
        'help': 'Energy compaction penalty parameter (beta)',
        'default': 0.001
        }
    },
    {'tasks': ['autoencoder', 'lc-compress'],
        'modes': ['training'],
        'flags': ('-cr', '--criterion'),
        'details': {
        'dest': 'criterion',
        'type': str,
        'help': 'Training criterion to evaluate the convolutional autoencoder',
        'default': 'RateMSE'
        }
    },
    {'tasks': ['autoencoder', 'lc-compress'],
        'modes': ['training'],
        'flags': ('-cem', '--class-error-mu'),
        'details': {
        'dest': 'class_error_mu',
        'type': float,
        'help': 'Classification error penalty parameter (mu)',
        'default': 1.0
        }
    },
    {'tasks': ['autoencoder', 'lc-compress'],
        'modes': ['training'],
        'flags': ('-ceam', '--class-error-aux-mu'),
        'details': {
        'dest': 'class_error_aux_mu',
        'type': float,
        'help': 'Auxiliary classification error penalty parameter (aux-mu)',
        'default': 0.0
        }
    },
]
