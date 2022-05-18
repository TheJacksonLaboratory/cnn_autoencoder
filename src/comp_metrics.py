from encodings import normalize_encoding
import logging
import argparse
import os

from time import perf_counter
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score

import numpy as np
import zarr
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
    else:
        input_fn_list = args.input

    args.source_format = 'zarr'
    compressor = Blosc(cname='zlib', clevel=0, shuffle=Blosc.BITSHUFFLE)

    for i, img_fn in enumerate(input_fn_list):
        if img_fn.endswith('zarr'):
            img_group = zarr.open(img_fn, mode='r')
        else:
            # Open the image (supported by PIL) as a zarr group in memory
            img_group = zarr.group()
            tmp_group = img_group.create_group('0')
            tmp_arr = tmp_group.array(name='0', data=utils.load_image(img_fn, args.patch_size), chunks=(3, args.patch_size, args.patch_size), compressor=compressor)

        e_time = perf_counter()

        args.input = img_group
        args.is_labeled = False
        comp_group = next(compress.compress(args))
        args.input = comp_group
        p_comp_group = next(factorized_entropy.fact_ent(args))
        rec_group = next(decompress.decompress(args))

        e_time = perf_counter() - e_time

        scores = metrics_image(img_group['0/0'], rec_group['0/0'], p_comp_group['0/0'])

        for m_k in scores.keys():
            if scores[m_k] > 0.0:
                all_metrics[m_k].append(scores[m_k])
            else:
                all_metrics[m_k].append(np.nan)

            logger.info('[Image %i] Metric %s: %0.4f' % (i+1, m_k, scores[m_k]))
        
        all_metrics['time'].append(e_time)
        logger.info('[Image %i] Execution time: %0.4f' % (i+1, e_time))
                
    for m_k in all_metrics.keys():
        avg_metric = np.nanmean(avg_metrics[m_k])
        std_metric = np.nanstd(avg_metrics[m_k])
        med_metric = np.nanmedian(avg_metrics[m_k])
        min_metric = np.nanmin(avg_metrics[m_k])
        max_metric = np.nanmax(avg_metrics[m_k])

        logger.info('[%s] Summary %s: %0.4f, %0.4f, %0.4f, %0.4f, %0.4f' % (args.trained_model, m_k, min_metric, avg_metric, med_metric, max_metric, std_metric))

    logging.shutdown()
    return avg_metrics


if __name__ == '__main__':
    seg_parser = utils.get_segment_args(parser_only=True)
    
    parser = argparse.ArgumentParser(prog='Evaluate a model on a testing set (Segmentation models only)', parents=[seg_parser], add_help=False)
    
    parser.add_argument('-ld', '--logdir', type=str, dest='log_dir', help='Directory where all logging and model checkpoints are stored', default='.')

    args = utils.override_config_file(parser)
    
    avg_metrics = metrics(args)
