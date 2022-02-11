VER = '0.5.4'

DATASETS = ['MNIST', 'ImageNet', 'Histology']

CAE_CRITERIONS = ['RD', 'RD_PA', 'RD_PB']
SEG_CRITERIONS = ['CE']

SCHEDULERS = ['None', 'ReduceOnPlateau']

SEG_MODELS = ['DecoderUNet', 'UNet']
CAE_MODELS = ['MaskedAutoEncoder', 'AutoEncoder']
