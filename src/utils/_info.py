VER = '0.5.7'
SEG_VER = '0.5.6'

DATASETS = ['MNIST',
            'CIFAR10',
            'CIFAR100',
            'ImageNet',
            'ImageNet.S3',
            'Zarr']

CLASS_CRITERIONS = ['CELoss',
                    'BCELoss',
                    'CELossWithAux',
                    'BCELossWithAux']

CAE_ACT_LAYERS = ['LeakyReLU',
                  'ReLU',
                  'GDN',
                  'Identiy']

OPTIMIZERS = ['Adam',
              'SGD']

SCHEDULERS = ['None',
              'ReduceOnPlateau',
              'StepLR',
              'ExponentialLR']

CAE_MODELS = ['AutoEncoder']

CLASS_MODELS = ['Empty',
                'ViT',
                'ResNet']
