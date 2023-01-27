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

OPTIMIZERS = ['Adam', 'SGD']
SCHEDULERS = ['None', 'ReduceOnPlateau', 'StepLR']
CAE_MODELS = ['AutoEncoder']
