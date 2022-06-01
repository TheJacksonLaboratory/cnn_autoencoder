VER = '0.5.5'

DATASETS = ['MNIST', 'ImageNet', 'Zarr', 'Histology']

CAE_CRITERIONS = ['RD', 'RD_MS', 'RD_PA', 'RD_PB']
SEG_CRITERIONS = ['CE']

SCHEDULERS = ['None', 'ReduceOnPlateau']

SEG_MODELS = ['DecoderUNet', 'UNet', 'UNetNoBridge']
CAE_MODELS = ['MaskedAutoEncoder', 'AutoEncoder']
PROJ_MODELS = ['KCCA']
FE_MODELS = ['FactorizedEntropy', 'FactorizedEntropyLaplace']
