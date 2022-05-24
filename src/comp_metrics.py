from functools import reduce
import logging
import argparse
import os

from time import perf_counter

import numpy as np
import random

import compress
import decompress
import factorized_entropy
import utils


def compute_psnr(rmse):
    return 20 * np.log10(1.0 / rmse)


def compute_rmse(img, rec):
    return np.sqrt(np.mean((rec[0, :]/255 - img[:]/255)**2))


def compute_rate(img, p_comp):    
    return np.sum(-np.log2(p_comp[:])) / img.size


def metrics_image(img, rec, p_comp):
    """ Compute distortion and compression ratio from the compressed representation and reconstruction from a single image.
    
    Parameters:
    ----------
    img: torch.Tensor
        Original image, used to compare the distortion between this and its reconstruction
    comp : torch.Tensor
        The compressed representation of the input image
    rec: torch.Tensor
        Reconstructed image obtained with the model

    Returns
    -------
    metrics_dict : Dictionary
        Dictionary with the computed metrics (dist=Distortion, rate=Compression rate (bpp), psnr=Peak Dignal-to-Noise Ratio (dB))
    """
    dist = compute_rmse(img, rec)
    rate = compute_rate(img, p_comp)
    psnr = compute_psnr(dist)
    
    metrics_dict = dict(dist=dist, rate=rate, psnr=psnr)

    return metrics_dict


def metrics(args):
    """ Compute the average perfomance in terms of different metrics for a set of images.
    Metrics computed: 
        Distortion
        Compression ratio
        Peak Dignal-to-Noise Ratio
    """
    # Override the destination format to 'zarr_memory' in case that it was given
    args.destination_format = 'zarr_memory'

    # Override 'is_labeled' to True, in order to get the segmentation response along with its respective ground-truth
    args.is_labeled = True
    args.mode = 'testing'
    
    utils.setup_logger(args)    
    logger = logging.getLogger(args.mode + '_log')

    all_metrics = dict(dist=[], rate=[], psnr=[], time=[])

    zarr_ds = utils.ZarrDataset(root=args.input, patch_size=args.patch_size, dataset_size=args.test_size, offset=False, transform=None, source_format=args.source_format)

    if args.test_size < 0:
        args.test_size = len(zarr_ds)

    args.source_format = 'zarr'

    for i in range(len(zarr_ds)):
        if args.shuffle_test:
            index = random.randrange(0, args.test_size)
        else:
            index = i
        
        e_time = perf_counter()

        img_arr, _ = zarr_ds[index]
        args.input = img_arr
        args.is_labeled = False
        comp_group = next(compress.compress(args))
        args.input = comp_group
        p_comp_group = next(factorized_entropy.fact_ent(args))
        rec_group = next(decompress.decompress(args))

        e_time = perf_counter() - e_time

        scores = metrics_image(img_arr, rec_group['0/0'], p_comp_group['0/0'])

        for m_k in scores.keys():
            if scores[m_k] > 0.0:
                all_metrics[m_k].append(scores[m_k])
            else:
                all_metrics[m_k].append(np.nan)

            logger.info('[Image %i] Metric %s: %0.4f' % (i+1, m_k, scores[m_k]))
        
        all_metrics['time'].append(e_time)
        logger.info('[Image %i] Execution time: %0.4f' % (i+1, e_time))
                
    logger.info('Metrics summary: min, mean, median, max, std. dev.')
    for m_k in all_metrics.keys():
        avg_metric = np.nanmean(all_metrics[m_k])
        std_metric = np.nanstd(all_metrics[m_k])
        med_metric = np.nanmedian(all_metrics[m_k])
        min_metric = np.nanmin(all_metrics[m_k])
        max_metric = np.nanmax(all_metrics[m_k])

        logger.info('[%s] Summary %s: %0.4f, %0.4f, %0.4f, %0.4f, %0.4f' % (args.trained_model, m_k, min_metric, avg_metric, med_metric, max_metric, std_metric))

    logging.shutdown()
    return all_metrics


if __name__ == '__main__':
    seg_parser = utils.get_segment_args(parser_only=True)
    
    parser = argparse.ArgumentParser(prog='Evaluate a model on a testing set (Segmentation models only)', parents=[seg_parser], add_help=False)
    
    parser.add_argument('-ld', '--logdir', type=str, dest='log_dir', help='Directory where all logging and model checkpoints are stored', default='.')
    parser.add_argument('-sht', '--shuffle-test', action='store_true', dest='shuffle_test', help='Shuffle the test set? Works for large images where only small regions will be used to test the performance instead of whole images.')
    parser.add_argument('-nt', '--num-test', type=int, dest='test_size', help='Size of set of test images used to evaluate the model.', default=-1)

    args = utils.override_config_file(parser)

    all_metrics = metrics(args)
