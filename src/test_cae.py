import logging
import os

from time import perf_counter
from skimage.color import deltaE_ciede2000, rgb2lab

import numpy as np
import zarr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models
import utils
from numcodecs import Blosc

model_options = {"AutoEncoder": models.AutoEncoder, "MaskedAutoEncoder": models.MaskedAutoEncoder}

def compute_deltaCIELAB(x=None, x_r=None, y_q=None):
    return np.mean(deltaE_ciede2000(rgb2lab(np.moveaxis(x, 1, -1)), rgb2lab(np.moveaxis(x_r, 1, -1))))


def compute_psnr(x=None, x_r=None, y_q=None):
    rmse = compute_rmse(x=x, x_r=x_r)
    return 20 * np.log10(1.0 / rmse)


def compute_rmse(x=None, x_r=None, y_q=None):
    return np.sqrt(np.mean((x_r - x)**2))


def compute_rate(x=None, x_r=None, y_q=None):
    # Check compression directly from the information of the zarr file
    z_y_q = zarr.array(data=y_q, chunks=(1, y_q.shape[1], y_q.shape[2], y_q.shape[3]), dtype='u1', compressor=Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE))
    return float(z_y_q.nbytes_stored) / np.prod(x.shape)

"""
Available metrics (can add more later):
    dist=Distortion
    rate=Compression rate (bpp)
    psnr=Peak Dignal-to-Noise Ratio (dB)
    delta_cielab=Distance between images in the CIELAB color space
"""
metric_fun = {'dist': compute_rmse, 'rate':compute_rate, 'psnr': compute_psnr, 'delta_cielab': compute_deltaCIELAB}


def test(cae_model, data, args):
    """ Test step.
    Evaluates the performance of the network in its current state using the full set of test elements.

    Parameters
    ----------
    cae_model : torch.nn.Module
        The network model in the current state
    data : torch.utils.data.DataLoader or list[tuple]
        The test dataset. Because the target is recosntruct the input, the label associated is ignored
    args: Namespace
        The input arguments passed at running time
    
    Returns
    -------
    metrics_dict : dict
        Dictionary with the computed metrics         
    """
    logger = logging.getLogger(args.mode + '_log')

    cae_model.eval()

    all_metrics = dict([(m_k, []) for m_k in metric_fun])
    all_metrics['time'] = []

    with torch.no_grad():
        for i, (x, _) in enumerate(data):
            e_time = perf_counter()
            x_r, y, _ = cae_model(x)
            y_q = y + 127.5
            y_q = y_q.round().to(torch.uint8)
            x_r = 0.5 * x_r.clip(-1.0, 1.0) + 0.5
            e_time = perf_counter() - e_time

            x = 0.5 * x + 0.5

            x_r = x_r.detach().cpu().numpy()
            x = x.detach().cpu().numpy()
            y_q = y_q.detach().cpu().numpy()

            for m_k in metric_fun.keys():
                score = metric_fun[m_k](x=x, x_r=x_r, y_q=y_q)
                if score > 0.0:
                    all_metrics[m_k].append(score)
                else:
                    all_metrics[m_k].append(np.nan)
            
            all_metrics['time'].append(e_time)

            if i % max(1, int(0.1 * len(data))) == 0:
                avg_metrics = ''
                for m_k in all_metrics.keys():
                    avg_metric = np.nanmean(all_metrics[m_k])
                    avg_metrics += '%s=%0.4f ' % (m_k, avg_metric)
                logger.debug('\t[{:04d}/{:04d}] Test metrics {}'.format(i, len(data), avg_metrics))
    
    all_metrics_stats = {}
    for m_k in all_metrics.keys():
        avg_metric = np.nanmean(all_metrics[m_k])
        std_metric = np.nanstd(all_metrics[m_k])
        med_metric = np.nanmedian(all_metrics[m_k])
        min_metric = np.nanmin(all_metrics[m_k])
        max_metric = np.nanmax(all_metrics[m_k])

        all_metrics_stats[m_k + '_stats'] = dict(avg=avg_metric, std=std_metric, med=med_metric, min=min_metric, max=max_metric)

    return all_metrics.update(all_metrics_stats)


def setup_network(args):
    """ Setup a nerual network for image compression/decompression.

    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. All the parameters are passed directly to the model constructor.
        This way, the constructor can take the parameters needed that have been passed by the user.
    
    Returns
    -------
    cae_model : nn.Module
        The convolutional neural network autoencoder model.
    """

    # The autoencoder model contains all the modules
    if not args.gpu:
        checkpoint_state = torch.load(args.trained_model, map_location=torch.device('cpu'))
    else:
        checkpoint_state = torch.load(args.trained_model)

    cae_model = model_options[checkpoint_state['args']['model_type']](**checkpoint_state['args'])

    # If there are more than one GPU, DataParallel handles automatically the distribution of the work
    cae_model = nn.DataParallel(cae_model)
    if args.gpu:
        cae_model.cuda()

    cae_model.module.embedding.load_state_dict(checkpoint_state['embedding'])
    cae_model.module.analysis.load_state_dict(checkpoint_state['encoder'])
    cae_model.module.synthesis.load_state_dict(checkpoint_state['decoder'])
    cae_model.module.fact_entropy.load_state_dict(checkpoint_state['fact_ent'])

    return cae_model


def main(args):
    """ Set up the training environment

    Parameters
    ----------
    args : dict or Namespace
        The set of parameters passed to the different constructors to set up the convolutional autoencoder training
    """
    logger = logging.getLogger(args.mode + '_log')

    cae_model = setup_network(args)

    # Log the training setup
    logger.info('Network architecture:')
    logger.info(cae_model)
    
    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    transform, _ = utils.get_zarr_transform(normalize=True)
    
    if not args.source_format.startswith('.'):
        args.source_format = '.' + args.source_format

    zarr_ds = utils.ZarrDataset(root=args.data_dir, mode=args.mode_data, dataset_size=args.test_size, patch_size=args.patch_size, offset=0, transform=transform, source_format=args.source_format, workers=args.workers)
    test_data = DataLoader(zarr_ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=args.shuffle_test, pin_memory=True, worker_init_fn=utils.zarrdataset_worker_init)
    
    all_metrics_stats = test(cae_model, test_data, args)
    torch.save(all_metrics_stats, os.path.join(args.log_dir, 'metrics_stats_%s%s.pth' % (args.seed, args.log_identifier)))


if __name__ == '__main__':
    args = utils.get_testing_args()

    utils.setup_logger(args)

    main(args)
    
    logging.shutdown()