import logging
import argparse
import os

from time import perf_counter

from skimage.color import deltaE_ciede2000, rgb2lab
import numpy as np
import zarr
from torch.utils.data import DataLoader

import compress
import decompress
import factorized_entropy
import utils


def compute_deltaCIELAB(img, rec):
    return np.mean(deltaE_ciede2000(rgb2lab(np.moveaxis(img, 1, -1)), rgb2lab(np.moveaxis(rec, 1, -1))))


def compute_psnr(rmse):
    return 20 * np.log10(1.0 / rmse)


def compute_rmse(img, rec):
    return np.sqrt(np.mean((rec[0, :]/255 - img[:]/255)**2))


def compute_rate(img, comp):
    # return np.sum(-np.log2(p_comp[:]+1e-10)) / (img.shape[0] * img.shape[-2] * img.shape[-1])
    # Check compression directly from the information of the zarr file
    return float(comp.nbytes_stored) / float(img.nbytes_stored)


def metrics_image(img, comp, rec):
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
        Dictionary with the computed metrics (dist=Distortion, rate=Compression rate (bpp), psnr=Peak Dignal-to-Noise Ratio (dB)), delta_cielab=Distance between images in the CIELAB color space
    """    
    dist = compute_rmse(img, rec)
    rate = compute_rate(img, comp)
    psnr = compute_psnr(dist)
    delta_cielab = compute_deltaCIELAB(img, rec)
    
    metrics_dict = dict(dist=dist, rate=rate, psnr=psnr, delta_cielab=delta_cielab)

    return metrics_dict


def compute_rois(img_arr):
    """ Compute a set of ROIs from the current batch of images.
    A batch can then be passed through the autoencoder model and be processeced as independent images, instead of stitching those together
    
    Parameters:
    ----------
    img_arr: torch.Tensor
        A batch of input images

    Returns
    -------
    rois : list of tuples
        Each tuple contains the starting coordinates and the axes lengths
    """
    b, c, h, w = img_arr.shape[0], img_arr.shape[1], img_arr.shape[-2], img_arr.shape[-1]
    rois = [((0, 0, 0, 0, b_i), (w, h, 1, c, 1)) for b_i in range(b)]
    return rois


def metrics(args):
    """ Compute the average perfomance in terms of different metrics for a set of images.
    Metrics computed: 
        Distortion
        Compression ratio
        Peak Dignal-to-Noise Ratio
    """
    # Override the destination format to 'zarr_memory' in case that it was given
    args.destination_format = 'zarr_memory'
    args.mode = 'testing'
    
    utils.setup_logger(args)    
    logger = logging.getLogger(args.mode + '_log')

    all_metrics = dict(dist=[], rate=[], psnr=[], delta_cielab=[], time=[])

    if len(args.input) == 1 and not args.input[0].lower().endswith(args.source_format):
        args.input = args.input[0]

    zarr_ds = utils.ZarrDataset(root=args.input, patch_size=args.patch_size, dataset_size=args.test_size, mode=args.mode_data, offset=False, transform=None, source_format=args.source_format)
    zarr_dl = DataLoader(zarr_ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=args.shuffle_test, worker_init_fn=utils.zarrdataset_worker_init)

    if args.test_size < 0:
        args.test_size = len(zarr_ds)

    # Override some arguments to be passed to the compression, decompression modules
    args.source_format = 'zarr'
    args.workers = 0
    args.stitch_batches = False
    args.reconstruction_level = -1
    args.compute_pyramids = False

    # Create a zarr group on memory to store the batch of input images
    img_group = zarr.group()
    img_subgroup = img_group.create_group('0')

    for i, (img_arr, _) in enumerate(zarr_dl):
        e_time = perf_counter()

        # Include the batch indices as ROIs (0, 0, 0, 0, b), where b is the batch index
        # i.e. if there are 8 images in the batch, there must be also 8 ROIs in that group
        img_group.attrs['rois'] = compute_rois(img_arr)
        img_subgroup.create_dataset('0', data=img_arr.numpy(), chunks=(1, img_arr.shape[1], args.patch_size, args.patch_size), overwrite=True)

        args.input = img_group
        # Compress the input images
        comp_group = next(compress.compress(args))
        # Compute dummy ROIs for the processed batch. This allows to pass a single batch of images like it were a set of independent images.
        comp_group.attrs['rois'] = compute_rois(comp_group['0/0'])
        
        args.input = comp_group
        # Reconstruct the images from their compressed representations
        rec_group = next(decompress.decompress(args))

        e_time = perf_counter() - e_time

        # Compute compression metrics
        scores = metrics_image(img_group['0/0'], comp_group['0/0'], rec_group['0/0'])

        for m_k in scores.keys():
            if scores[m_k] > 0.0:
                all_metrics[m_k].append(scores[m_k])
            else:
                all_metrics[m_k].append(np.nan)

            logger.info('[Image/batch %i] Metric %s: %0.4f' % (i+1, m_k, scores[m_k]))
        
        all_metrics['time'].append(e_time)
        logger.info('[Image/batch %i] Execution time: %0.4f' % (i+1, e_time))
                
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
    
    parser.add_argument('-ld', '--log-dir', type=str, dest='log_dir', help='Directory where all logging and model checkpoints are stored', default='.')
    parser.add_argument('-md', '--mode-data', type=str, dest='mode_data', help='Mode of the dataset used to compute the metrics', choices=['train', 'va', 'test', 'all'], default='all')
    parser.add_argument('-sht', '--shuffle-test', action='store_true', dest='shuffle_test', help='Shuffle the test set? Works for large images where only small regions will be used to test the performance instead of whole images.')
    parser.add_argument('-nt', '--num-test', type=int, dest='test_size', help='Size of set of test images used to evaluate the model.', default=-1)

    args = utils.override_config_file(parser)

    all_metrics = metrics(args)
