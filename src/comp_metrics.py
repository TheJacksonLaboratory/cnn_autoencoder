from functools import reduce
import logging
import argparse
import os

from time import perf_counter
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score

import numpy as np
import zarr
import dask.array as da
from numcodecs import Blosc

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

    if not args.input[0].lower().endswith(args.source_format.lower()):
        # If a directory has been passed, get all image files inside to compress
        input_fn_list = list(map(lambda fn: os.path.join(args.input[0], fn), filter(lambda fn: fn.endswith(args.source_format.lower()), os.listdir(args.input[0]))))
    elif args.input[0].lower().endswith('txt'):
        with open(args.input[0], mode='r') as f:
            input_fn_list = [l.strip('\n\r') for l in f.readlines()]
    else:
        input_fn_list = args.input

    if args.rois is not None:
        args.rois = [
                [(tuple(start_coords), tuple(axis_lengths))
                    for start_coords, axis_lengths in rois
                ]
                for rois in args.rois
            ]
    else:
        args.rois = [None for _ in range(len(input_fn_list))]
    args.source_format = 'zarr'
    compressor = Blosc(cname='zlib', clevel=0, shuffle=Blosc.BITSHUFFLE)

    for i, (img_fn, img_rois) in enumerate(zip(input_fn_list, args.rois)):
        if img_fn.endswith('zarr'):
            # img_group = zarr.open(img_fn, mode='r')
            img_arr = da.from_zarr(img_fn, component='0/0')
        else:
            # Open the image (supported by PIL) as a zarr group in memory
            # img_group = zarr.group()
            # tmp_group = img_group.create_group('0')
            # img_arr = tmp_group.array(name='0', data=utils.load_image(img_fn, args.patch_size), chunks=(3, args.patch_size, args.patch_size), compressor=compressor)
            img_arr = da.from_array(utils.load_image(img_fn, args.patch_size), chunks=(3, args.patch_size, args.patch_size))
        
        if img_rois is not None:
            arr_rois = []
            for (x_o, y_o, _, c_o, _), (x_l, y_l, _, c_l, _) in img_rois:
                arr_rois.append(img_arr[c_o:c_o+c_l, x_o:x_o+x_l, y_o:y_o+y_l])
            img_arr = reduce(lambda l1, l2: l1+l2, arr_rois)

        e_time = perf_counter()

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
    parser.add_argument('-roi', '--input-rois', type=int, nargs='+', dest='rois', help='Regions of interest extacted from the input images to compute the compression performance')

    args = utils.override_config_file(parser)

    all_metrics = metrics(args)
