import logging
import os

from time import perf_counter
from skimage.color import deltaE_cie76, rgb2lab
from skimage.metrics import (mean_squared_error,
                             peak_signal_noise_ratio,
                             structural_similarity)
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure

import numpy as np
import zarr
import torch

import models
import utils
import compress
import decompress


def compute_deltaCIELAB(x=None, x_r=None, **kwargs):
    convert_x_time = perf_counter()
    x_lab = rgb2lab(x)
    convert_x_time = perf_counter() - convert_x_time

    convert_r_time = perf_counter()
    x_r_lab = rgb2lab(x_r)
    convert_r_time = perf_counter() - convert_r_time

    delta_cielab_time = perf_counter()
    delta_cielab = deltaE_cie76(x_lab, x_r_lab)
    delta_cielab_time = perf_counter() - delta_cielab_time

    mean_delta_cielab_time = perf_counter()
    mean_delta_cielab = np.mean(delta_cielab)
    mean_delta_cielab_time = perf_counter() - mean_delta_cielab_time

    return mean_delta_cielab, dict(convert_x=convert_x_time,
                                   convert_r=convert_r_time,
                                   delta_cielab_time=delta_cielab_time,
                                   mean_delta_cielab=mean_delta_cielab_time,
                                   delta_shape_ndim=delta_cielab.ndim,
                                   delta_shape_x=delta_cielab.shape[0],
                                   delta_shape_y=delta_cielab.shape[1])


def compute_ms_ssim(x=None, x_r=None, **kwargs):
    ms_ssim_fn = MultiScaleStructuralSimilarityIndexMeasure(
        kernel_size=(11, 11),
        sigma=(1.5, 1.5),
        reduction='elementwise_mean',
        k1=0.01,
        k2=0.03,
        data_range=1,
        betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        normalize='relu'
    )
    ms_ssim = ms_ssim_fn(
        torch.from_numpy(np.moveaxis(x_r, -1, 0)[np.newaxis]).float() / 255.0,
        torch.from_numpy(np.moveaxis(x, -1, 0)[np.newaxis]).float() / 255.0)
    return ms_ssim, None


def compute_ssim(x=None, x_r=None, **kwargs):
    ssim = structural_similarity(x, x_r, channel_axis=2)
    return ssim, None


def compute_psnr(x=None, x_r=None, **kwargs):
    psnr = peak_signal_noise_ratio(x, x_r)
    return psnr, None


def compute_rmse(x=None, x_r=None, **kwargs):
    rmse = np.sqrt(mean_squared_error(x / 255.0, x_r / 255.0))
    return rmse, None


def compute_rate(x=None, x_r=None, y_q_ptr=None, **kwargs):
    # Check compression directly from the information of the zarr file
    return 8 * float(y_q_ptr.nbytes_stored) / np.prod(x.shape[:-1]), None


"""Available metrics (can add more later):
    dist=Distortion (RMSE)
    rate=Compression rate (bits-per-pixel bpp)
    ssim=Structural similarity
    psnr=Peak Dignal-to-Noise Ratio (dB)
    delta_cielab=Distance between images in the CIELAB color space (RMSE in
    CIELAB space)
"""
metric_fun = {'dist': compute_rmse,
              'rate': compute_rate,
              'ms-ssim': compute_ms_ssim,
              'ssim': compute_ssim,
              'psnr': compute_psnr,
              'delta_cielab': compute_deltaCIELAB}


def test_image(comp_model, decomp_model, input_filename,
               channels_bn,
               compression_level,
               patch_size=512,
               add_offset=False,
               transform_comp=None,
               transform_decomp=None,
               source_format='zarr',
               data_group='0/0',
               data_axes='TCZYX',
               seed=None,
               temp_output_filename="./temp.zarr",
               min_range=0.0,
               max_range=1.0,
               range_offset=0.0,
               range_scale=1.0):

    e_time = perf_counter()
    compress.compress_image(comp_model, input_filename, temp_output_filename,
                            channels_bn,
                            compression_level,
                            patch_size=patch_size,
                            add_offset=add_offset,
                            transform=transform_comp,
                            source_format=source_format,
                            data_group=data_group,
                            data_axes=data_axes,
                            seed=seed,
                            comp_label='compressed')
    decompress.decompress_image(decomp_model, temp_output_filename,
                                temp_output_filename,
                                patch_size=patch_size,
                                add_offset=add_offset,
                                transform=transform_decomp,
                                compute_pyramids=False,
                                reconstruction_level=-1,
                                destination_format='zarr',
                                data_group='compressed',
                                data_axes='TCZYX',
                                seed=seed,
                                decomp_label='decompressed',
                                min_range=min_range,
                                max_range=max_range,
                                range_offset=range_offset,
                                range_scale=range_scale)
    e_time = perf_counter() - e_time

    arr, arr_shape, _ = utils.image_to_zarr(input_filename.split(';')[0],
                                            patch_size,
                                            source_format,
                                            data_group)

    H, W, in_channels = [arr_shape[data_axes.index(a)] for a in 'YXC']
    slices = [slice(0, W, 1), slice(0, H, 1), slice(0, 1, 1),
              slice(0, in_channels, 1),
              slice(0, 1, 1)]
    slices = tuple([slices['XYZCT'.index(a)] for a in data_axes])

    unused_axis = list(set(data_axes) - set('YXC'))
    transpose_order = [data_axes.index(a) for a in unused_axis]
    transpose_order += [data_axes.index(a) for a in 'YXC']

    x = arr[slices]
    y_q = zarr.open(temp_output_filename, mode="r")['compressed']
    x_r = zarr.open(temp_output_filename, mode="r")['decompressed/0']

    x = np.transpose(x, transpose_order).squeeze()
    x_r = np.transpose(x_r, (3, 4, 1, 0, 2)).squeeze()

    all_metrics = {}
    all_extra_info = {}

    eval_time = perf_counter()
    for m_k in metric_fun.keys():
        metrics_eval_time = perf_counter()
        score, extra_info = metric_fun[m_k](x=x, x_r=x_r, y_q_ptr=y_q)
        metrics_eval_time = perf_counter() - metrics_eval_time
        all_metrics[m_k + '_time'] = metrics_eval_time

        if extra_info is not None:
            for e_k in extra_info.keys():
                all_extra_info[e_k] = extra_info[e_k]

        if score >= 0.0:
            all_metrics[m_k] = score
        else:
            all_metrics[m_k] = np.nan

    eval_time = perf_counter() - eval_time
    all_metrics['execution_time'] = e_time
    all_metrics['evaluation_time'] = eval_time
    all_metrics.update(all_extra_info)

    return all_metrics


def test_cae(args):
    """ Set up the training environment

    Parameters
    ----------
    args : dict or Namespace
        The set of parameters passed to the different constructors to set up the convolutional autoencoder training
    """
    logger = logging.getLogger(args.mode + '_log')

    state = utils.load_state(args)

    if state['args']['normalize']:
        min_range = -1.0
        max_range = 1.0
        range_scale = 0.5
        range_offset = 0.5
    else:
        min_range = 0.0
        max_range = 1.0
        range_scale = 1.0
        range_offset = 0.0

    comp_model = compress.setup_network(state, args.use_gpu)
    decomp_model = decompress.setup_network(state, rec_level=-1,
                                            compute_pyramids=False,
                                            use_gpu=args.use_gpu)

    # Log the training setup
    logger.info('Network architecture:')
    logger.info(comp_model)
    logger.info(decomp_model)

    transform_comp, _, _ = utils.get_zarr_transform(normalize=True)
    transform_decomp, _, _ = utils.get_zarr_transform(normalize=True,
                                                      compressed_input=True)

    # Get the compression level from the model checkpoint
    compression_level = state['args']['compression_level']

    input_fn_list = utils.get_filenames(args.data_dir, args.source_format,
                                        data_mode='all')

    all_metrics_stats = dict([(m_k, []) for m_k in metric_fun])
    all_metrics_stats['execution_time'] = []
    all_metrics_stats['evaluation_time'] = []

    if not args.output_dir.lower().endswith(".zarr"):
        args.output_dir += ".zarr"

    for i, in_fn in enumerate(input_fn_list):
        all_metrics = test_image(comp_model=comp_model,
                                 decomp_model=decomp_model,
                                 input_filename=in_fn,
                                 channels_bn=state['args']['channels_bn'],
                                 compression_level=compression_level,
                                 patch_size=args.patch_size,
                                 add_offset=args.add_offset,
                                 transform_comp=transform_comp,
                                 transform_decomp=transform_decomp,
                                 source_format=args.source_format,
                                 data_axes=args.data_axes,
                                 data_group=args.data_group,
                                 seed=state['args']['seed'],
                                 temp_output_filename=args.output_dir,
                                 min_range=min_range,
                                 max_range=max_range,
                                 range_offset=range_offset,
                                 range_scale=range_scale)

        avg_metrics = ''
        for m_k in all_metrics_stats.keys():
            all_metrics_stats[m_k].append(all_metrics[m_k])
            avg_metric = np.nanmean(all_metrics[m_k])
            avg_metrics += '%s=%0.5f ' % (m_k, avg_metric)

        logger.debug(
            '\t[{:05d}/{:05d}] Test metrics {}'.format(
                i+1,
                len(input_fn_list),
                avg_metrics))

    for m_k in list(all_metrics_stats.keys()):
        avg_metric = np.nanmean(all_metrics_stats[m_k])
        std_metric = np.nanstd(all_metrics_stats[m_k])
        med_metric = np.nanmedian(all_metrics_stats[m_k])
        min_metric = np.nanmin(all_metrics_stats[m_k])
        max_metric = np.nanmax(all_metrics_stats[m_k])

        all_metrics_stats[m_k + '_stats'] = dict(avg=avg_metric,
                                                 std=std_metric,
                                                 med=med_metric,
                                                 min=min_metric,
                                                 max=max_metric)
        avg_metrics = '%s=%0.4f (+-%0.4f)' % (m_k, avg_metric, std_metric)
        logger.debug('==== Test metrics {}'.format(avg_metrics))

    all_metrics_stats['codec'] = 'CAE'
    all_metrics_stats['seed'] = state['args']['seed']

    torch.save(all_metrics_stats,
               os.path.join(args.log_dir,
                            'metrics_stats_%s_CAE_%s.pth' % (args.seed,
                                                        args.log_identifier)))


if __name__ == '__main__':
    args = utils.get_args(task='autoencoder', mode='test')

    utils.setup_logger(args)

    test_cae(args)

    logging.shutdown()
