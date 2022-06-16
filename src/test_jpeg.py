from ctypes import util
import logging
import os

from time import perf_counter
from skimage.color import deltaE_cie76, rgb2lab

import numpy as np
import zarr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from numcodecs import Blosc, register_codec
from imagecodecs.numcodecs import Jpeg2k, JpegXl, JpegXr, Jpeg

import utils

COMPRESSORS = {'Jpeg': Jpeg, 'Jpeg2k': Jpeg2k, 'JpegXl': JpegXl, 'JpegXr': JpegXr}

register_codec(Jpeg)
register_codec(Jpeg2k)
register_codec(JpegXl)
register_codec(JpegXr)


def compute_deltaCIELAB(x=None, x_r=None):
    convert_x_time = perf_counter()
    x_lab = rgb2lab(x)
    convert_x_time = perf_counter() - convert_x_time
    
    convert_r_time = perf_counter()
    x_r_lab = rgb2lab(x_r[:])
    convert_r_time = perf_counter() - convert_r_time
    
    delta_cielab_time = perf_counter()
    delta_cielab = deltaE_cie76(x_lab, x_r_lab)
    delta_cielab_time = perf_counter() - delta_cielab_time
    
    mean_delta_cielab_time = perf_counter()
    mean_delta_cielab = np.mean(delta_cielab)
    mean_delta_cielab_time = perf_counter() - mean_delta_cielab_time
    
    return mean_delta_cielab, dict(convert_x=convert_x_time, convert_r=convert_r_time, delta_cielab=delta_cielab_time, mean_delta_cielab=mean_delta_cielab_time, delta_shape_ndim=delta_cielab.ndim, delta_shape_x=delta_cielab.shape[0], delta_shape_y=delta_cielab.shape[1])


def compute_psnr(x=None, x_r=None):
    rmse, _ = compute_rmse(x=x, x_r=x_r)
    if rmse < 1e-12:
        return -1.0, None
    return 20 * np.log10(1.0 / rmse), None


def compute_rmse(x=None, x_r=None):    
    rmse = np.sqrt(np.mean((x_r[:]/255. - x/255.)**2))
    return rmse, None


def compute_rate(x=None, x_r=None):
    # Check compression directly from the information of the zarr file
    return float(x_r.nbytes_stored) / np.prod(x.shape), None

"""
Available metrics (can add more later):
    dist=Distortion
    rate=Compression rate (bpp)
    psnr=Peak Dignal-to-Noise Ratio (dB)
    delta_cielab=Distance between images in the CIELAB color space
"""
metric_fun = {'dist': compute_rmse, 'rate':compute_rate, 'psnr': compute_psnr, 'delta_cielab': compute_deltaCIELAB}


def test(data, args):
    """ Test step.
    Evaluates the performance of the JPEG compressor.

    Parameters
    ----------
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
    compressor = COMPRESSORS[args.compressor_type](level=100 - args.compression_level)

    all_metrics = dict([(m_k, []) for m_k in metric_fun])
    all_metrics['time'] = []

    load_times = []
    eval_times = []
    metrics_eval_times = dict([(m_k, []) for m_k in metric_fun])
    
    all_extra_info = {}
    
    n_examples = 0

    with torch.no_grad():
        load_time = perf_counter()
        for i, (x, _) in enumerate(data):            
            load_time = perf_counter() - load_time            
            load_times.append(load_time)

            n_examples += x.size(0)
            
            x = 127.5 * x[0] + 127.5
            x = x.detach().cpu().to(torch.uint8).numpy().transpose(1, 2, 0)

            e_time = perf_counter()
            x_r = zarr.array(data=x, chunks=tuple(x.shape), dtype='u1', compressor=compressor)
            e_time = perf_counter() - e_time

            eval_time = perf_counter()
            for m_k in metric_fun.keys():
                metrics_eval_time = perf_counter()
                score, extra_info = metric_fun[m_k](x=x, x_r=x_r)
                metrics_eval_time = perf_counter() - metrics_eval_time
                metrics_eval_times[m_k].append(metrics_eval_time)
                
                if extra_info is not None:
                    for e_k in extra_info.keys():
                        if all_extra_info.get(e_k, None) is None:
                            all_extra_info[e_k] = []
                        
                        all_extra_info[e_k].append(extra_info[e_k])
                
                if score >= 0.0:
                    all_metrics[m_k].append(score)
                else:
                    all_metrics[m_k].append(np.nan)
            eval_time = perf_counter() - eval_time
            eval_times.append(eval_time)
            
            all_metrics['time'].append(e_time)

            if n_examples % max(1, int(0.1 * args.test_size)) == 0:
                avg_metrics = ''
                for m_k in all_metrics.keys():
                    avg_metric = np.nanmean(all_metrics[m_k])
                    avg_metrics += '%s=%0.5f ' % (m_k, avg_metric)
                logger.debug('\t[{:05d}/{:05d}][{:05d}/{:05d}] Test metrics {}'.format(i, len(data), n_examples, args.test_size, avg_metrics))
            
            load_time = perf_counter()
            if n_examples >= args.test_size:
                break
    
    logger.debug('Loading avg. time: {:0.5f} (+-{:0.5f}), evaluation avg. time: {:0.5f}(+-{:0.5f})'.format(np.mean(load_times), np.std(load_times), np.mean(eval_times), np.std(eval_times)))
    
    for m_k in metrics_eval_times.keys():
        avg_eval_time = np.mean(metrics_eval_times[m_k])
        std_eval_time = np.std(metrics_eval_times[m_k])
        logger.debug('\tEvaluation of {} avg. time: {:0.5f} (+- {:0.5f})'.format(m_k, avg_eval_time, std_eval_time))    
    
    for e_k in all_extra_info.keys():
        avg_ext_time = np.mean(all_extra_info[e_k])
        std_ext_time = np.std(all_extra_info[e_k])
        logger.debug('\tExtra info of {} avg. time: {:0.5f} (+- {:0.5f})'.format(e_k, avg_ext_time, std_ext_time))
        
    all_metrics_stats = {}
    for m_k in all_metrics.keys():
        avg_metric = np.nanmean(all_metrics[m_k])
        std_metric = np.nanstd(all_metrics[m_k])
        med_metric = np.nanmedian(all_metrics[m_k])
        min_metric = np.nanmin(all_metrics[m_k])
        max_metric = np.nanmax(all_metrics[m_k])

        all_metrics_stats[m_k + '_stats'] = dict(avg=avg_metric, std=std_metric, med=med_metric, min=min_metric, max=max_metric)

    all_metrics.update(all_metrics_stats)
    
    return all_metrics


def main(args):
    """ Set up the training environment

    Parameters
    ----------
    args : dict or Namespace
        The set of parameters passed to the different constructors to set up the convolutional autoencoder training
    """
    logger = logging.getLogger(args.mode + '_log')
   
    if not args.source_format.startswith('.'):
        args.source_format = '.' + args.source_format

    args.batch_size = 1

    if args.dataset.lower() == 'imagenet.s3':
        if utils.ImageS3 is not None:
            transform = utils.get_imagenet_transform(args.mode_data, normalize=True, patch_size=args.patch_size)
            test_data = utils.ImageS3(root=args.data_dir, transform=transform)
            test_queue = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=args.shuffle_test, pin_memory=True)
        else:
            raise ValueError('Boto3 is not installed, cannot use ImageNet from a S3 bucket')
    elif args.dataset.lower() == 'imagenet':
        transform = utils.get_imagenet_transform(args.mode_data, normalize=True, patch_size=args.patch_size)
        test_data = utils.ImageFolder(root=args.data_dir, transform=transform)
        test_queue = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=args.shuffle_test, pin_memory=True)
    else:
        # Generate a dataset from a single image to divide in patches and iterate using a dataloader
        transform, _ = utils.get_zarr_transform(normalize=True)
        test_data = utils.ZarrDataset(root=args.data_dir, dataset_size=1000000, mode=args.mode_data, patch_size=args.patch_size, offset=0, transform=transform, source_format=args.source_format, workers=args.workers)
        test_queue = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=args.shuffle_test, pin_memory=True, worker_init_fn=utils.zarrdataset_worker_init)
    
    if args.test_size < 0:
        args.test_size = len(test_data)

    all_metrics_stats = test(test_queue, args)
    torch.save(all_metrics_stats, os.path.join(args.log_dir, 'metrics_stats_%s%s.pth' % (args.seed, args.log_identifier)))
    
    for m_k in list(metric_fun.keys()) + ['time']:
        avg_metrics = '%s=%0.4f (+-%0.4f)' % (m_k, all_metrics_stats[m_k + '_stats']['avg'], all_metrics_stats[m_k + '_stats']['std'])
        logger.debug('==== Test metrics {}'.format(avg_metrics))


if __name__ == '__main__':
    parser = utils.get_testing_args(parser_only=True)

    parser.add_argument('-ct', '--comp-type', type=str, dest='compressor_type', help='Compressor type', choices=COMPRESSORS.keys(), default=list(COMPRESSORS.keys())[0])
    parser.add_argument('-cl', '--comp-level', type=int, dest='compression_level', help='Compression level (0-100)', default=50)

    args = utils.override_config_file(parser)
    args.mode = 'testing'

    utils.setup_logger(args)

    main(args)
    
    logging.shutdown()