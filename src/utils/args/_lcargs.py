lc_args = [
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lctp', '--lc-type'),
        'details': {
        'dest': 'lc_type',
        'type': str,
        'choices': ['lc', 'ft'],
        'default': 'lc'
        }
    },
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lctg', '--lc-tag'),
        'details': {
        'dest': 'lc_tag',
        'type': str,
        'default': 'tag'
        }
    },
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lcr', '--lc-resume'),
        'details': {
        'dest': 'lc_resume',
        'action': 'store_true'
        }
    },
    {'tasks': ['lc-compress', 'encoder', 'decoder'],
        'modes': ['training', 'inference'],
        'flags': ('-lcpm', '--lc-pretrained-model'),
        'details': {
        'dest': 'lc_pretrained_model',
        'type': str,
        'help': 'Model pretrained with the LC algorthm'
        }
    },
    {'tasks': ['lc-compress', 'encoder', 'decoder'],
        'modes': ['training', 'inference'],
        'flags': ('-lcft', '--lc-pretrained-model-ft'),
        'details': {
        'dest': 'ft_pretrained_model',
        'type': str,
        'help': 'Model fine tuned with the LC algorthm'
        }
    },
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lcs', '--lc-steps'),
        'details': {
        'dest': 'lc_steps',
        'type': int,
        'default': 20
        }
    },
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lcmui', '--lc-mu_init'),
        'details': {
        'dest': 'lc_mu_init',
        'type': float,
        'default': 9e-5}
    },
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lcmuc', '--lc-mu_inc'),
        'details': {
        'dest': 'lc_mu_inc',
        'type': float,
        'default': 1.09
        }
    },
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lcmur', '--lc-mu_rep'),
        'details': {
        'dest': 'lc_mu_rep',
        'type': int,
        'default': 1}},
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lccvs', '--lc-conv_scheme'),
        'details': {
        'dest': 'lc_conv_scheme',
        'type': str,
        'choices': ['scheme_1', 'scheme_2'],
        'default': 'scheme_1'}},
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lcal', '--lc-alpha'),
        'details': {
        'dest': 'lc_alpha',
        'type': float}},
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lccr', '--lc-criterion'),
        'details': {
        'dest': 'lc_criterion',
        'type': str,
        'choices': ['storage', 'flops'],
        'default': 'storage'}},
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lcld', '--lc-lr_decay'),
        'details': {
        'dest': 'lc_lr_decay',
        'default': 0.98,
        'type': float,
        'metavar': 'LRD',
        'help': 'learning rate decay'}},
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lcldm', '--lc-lr_decay_mode'),
        'details': {
        'dest': 'lc_lr_decay_mode',
        'type': str,
        'choices': ['after_l', 'restart_on_l'],
        'default': 'after_l'}},
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lcha', '--lc-half'),
        'details': {
        'dest': 'lc_half',
        'action': 'store_true'}},
    {'tasks': ['lc-compress'],
        'modes': ['training'],
        'flags': ('-lcdd', '--lc-data-dir'),
        'details': {
        'type': str,
        'nargs': '+',
        'dest': 'lc_data_dir',
        'help': 'Directory, list of files, or text file with a list of '
                'files to be used as inputs for LC steps evaluations.'}},
]
