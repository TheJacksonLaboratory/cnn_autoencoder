VER = '0.5.7'
SEG_VER = '0.5.6'

DATASETS = ['MNIST', 'ImageNet', 'ImageNet.S3', 'Zarr', 'Histology']

CAE_CRITERIONS = ['RD',
                  'MultiscaleRD',
                  'RD_PA',
                  'RD_PB',
                  'MultiscaleRD_PA',
                  'MultiscaleRD_PB',
                  'RMS-SSIM',
                  'MultiscaleRMS-SSIM',
                  'RMS-SSIM_PA',
                  'RMS-SSIM_PB',
                  'MultiscaleRMS-SSIM_PA',
                  'MultiscaleRMS-SSIM_PB']

CAE_ACT_LAYERS = ['LeakyReLU',
                  'ReLU',
                  'GDN',
                  'Identiy']

SEG_CRITERIONS = ['CE', 'CEW']

OPTIMIZERS = ['Adam', 'SGD']
SCHEDULERS = ['None', 'ReduceOnPlateau', 'StepLR']

SEG_MODELS = ['DecoderUNet', 'UNet', 'UNetNoBridge']
CAE_MODELS = ['MaskedAutoEncoder', 'AutoEncoder']

PROJ_MODELS = ['KCCA']
FE_MODELS = ['FactorizedEntropy', 'FactorizedEntropyLaplace']
CLASS_MODELS = ['InceptionV3', 'MobileNet', 'ResNet', 'ViT']

MERGE_TYPES = ['mean', 'max', 'median']
