import argparse
import os
import torch
import numpy as np

from tqdm import tqdm
from time import perf_counter

from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import (mean_squared_error,
                             peak_signal_noise_ratio,
                             structural_similarity)

format_dict = {'JPEG2000': 'jp2', 'JPEG': 'jpeg', 'PNG': 'png'}


def compute_deltaCIELAB(img, rec):
    return np.mean(deltaE_ciede2000(rgb2lab(img), rgb2lab(rec)))


def compute_ssim(x=None, x_r=None, **kwargs):
    ssim = structural_similarity(x, x_r, channel_axis=2)
    return ssim, None


def compute_psnr(x=None, x_r=None, **kwargs):
    psnr = peak_signal_noise_ratio(x, x_r)
    return psnr, None


def compute_rmse(x=None, x_r=None, **kwargs):
    rmse = np.sqrt(mean_squared_error(x / 255.0, x_r / 255.0))
    return rmse, None


def compute_rate(img, comp_size):
    # Compute the compression rate as bits per pixel (bpp)
    return 8 * float(comp_size) / np.prod(img.shape[:-1])


def metrics_image(src_fn, comp_fn):
    """Compute distortion and compression ratio from the compressed
    representation and reconstruction from a single image.

    Parameters:
    ----------
    src_fn: str
        Filename of the original image
    comp_fn : str
        Filename of the compressed image

    Returns
    -------
    metrics_dict : Dictionary
        Dictionary with the computed metrics
        (dist=Distortion, rate=Compression rate (bpp),
         psnr=Peak Dignal-to-Noise Ratio (dB)),
         delta_cielab=Distance between images in the CIELAB color space
    """
    img = Image.open(src_fn, mode='r')
    img_arr = np.array(img)
    img.close()

    comp_size = os.path.getsize(comp_fn)
    comp = Image.open(comp_fn, mode='r')
    comp_arr = np.array(comp)
    comp.close()

    dist = compute_rmse(img_arr, comp_arr)
    rate = compute_rate(img_arr, comp_size)
    psnr = compute_psnr(dist)
    delta_cielab = compute_deltaCIELAB(img_arr, comp_arr)

    metrics_dict = dict(dist=dist, rate=rate, psnr=psnr, delta_cielab=delta_cielab)

    return metrics_dict


def convert(src_filename, dst_filename, file_format, **kwargs):
    im = Image.open(src_filename, mode='r')
    im.save(dst_filename, format=file_format, **kwargs)
    im.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compute compression performance metrics between a compressed image and the original one')

    parser.add_argument('-sd', '--src-dir', type=str, dest='src_dir', help='Source directory', default='.')
    parser.add_argument('-dd', '--dst-dir', type=str, dest='dst_dir', help='Destination directory', default='.')
    parser.add_argument('-sf', '--src-format', type=str, dest='src_format', help='Source image format')
    parser.add_argument('-df', '--dst-format', type=str, dest='dst_format', help='Destination image format', choices=list(format_dict.keys()))
    parser.add_argument('-ld', '--log-dir', type=str, dest='log_dir', help='Path where to store the performance logging', default='.')
    parser.add_argument('-cq', '--comp-quality', type=int, dest='comp_quality', help='Compression quality (from 0 to 100)', default=100)
    parser.add_argument('-li', '--log-id', type=str, dest='log_identifier', help='An identifier added to the log files', default='')

    args = parser.parse_args()

    in_filenames = ['.'.join(fn.split('.')[:-1]) for fn in os.listdir(args.src_dir) if fn.lower().endswith(format_dict[args.src_format])]

    all_metrics = dict(dist=[], rate=[], psnr=[], delta_cielab=[], time=[])

    if 'JPEG' in args.dst_format:
        quality_opts = {'quality': args.comp_quality}
    elif 'PNG' in args.dst_format:
        quality_opts = {'compress_level': 9 - args.comp_quality, 'optimize': False}

    q = tqdm(total=len(in_filenames))
    for in_fn in in_filenames:
        src_fn = os.path.join(args.src_dir, '%s.%s' % (in_fn, format_dict[args.src_format]))
        comp_fn = os.path.join(args.dst_dir, '%s_%03d.%s' % (in_fn, args.comp_quality, format_dict[args.dst_format]))

        e_time = perf_counter()
        convert(src_fn, comp_fn, args.dst_format, **quality_opts)
        e_time = perf_counter() - e_time

        scores = metrics_image(src_fn, comp_fn)

        for m_k in scores.keys():
            if scores[m_k] > 0.0:
                all_metrics[m_k].append(scores[m_k])
            else:
                all_metrics[m_k].append(np.nan)

        all_metrics['time'].append(e_time)
        os.remove(comp_fn)
        q.update()

    q.close()

    all_metrics_stats = {}
    for m_k in all_metrics.keys():
        avg_metric = np.nanmean(all_metrics[m_k])
        std_metric = np.nanstd(all_metrics[m_k])
        med_metric = np.nanmedian(all_metrics[m_k])
        min_metric = np.nanmin(all_metrics[m_k])
        max_metric = np.nanmax(all_metrics[m_k])

        all_metrics_stats[m_k + '_stats'] = dict(avg=avg_metric, std=std_metric, med=med_metric, min=min_metric, max=max_metric)

    all_metrics.update(all_metrics_stats)

    torch.save(all_metrics, os.path.join(args.log_dir, 'metrics_stats_%s_%03d%s.pth' % (args.dst_format, args.comp_quality, args.log_identifier)))
