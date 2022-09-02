import logging
import os

from time import perf_counter
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from functools import partial

import models
import utils

seg_model_types = {"UNetNoBridge": models.UNetNoBridge, "UNet": models.UNet, "DecoderUNet": models.DecoderUNet}


def compute_roc(pred, target, pred_threshold=None):
    try:
        roc = roc_auc_score(target, pred)
    except ValueError:
        roc = -1
    return roc


def compute_f1(pred, target, pred_threshold=None):
    try:
        f1 = f1_score(target, pred > pred_threshold)
    except ValueError:
        f1 = -1
    return f1


def compute_acc(pred, target, pred_threshold=None):
    try:
        acc = accuracy_score(target, pred > pred_threshold)
    except ValueError:
        acc = -1
    return acc


def compute_recall(pred, target, pred_threshold=None):
    try:
        recall = recall_score(target, pred > pred_threshold)
    except ValueError:
        recall = -1
    return recall


def compute_prec(pred, target, pred_threshold=None):
    try:
        prec = precision_score(target, pred > pred_threshold)
    except ValueError:
        prec = -1
    return prec


"""
Available metrics (can add more later):
    roc=Area under the ROC curve
    f1=Dice coefficient also known as F1 score
    acc=Prediction accuracy
    recall=Sensitivity of the model
    prec=Precision of the model
"""
metric_fun = {'roc': compute_roc,
              'f1': compute_f1,
              'acc': compute_acc,
              'recall': compute_recall,
              'prec': compute_prec}


def test(forward_function, data, args):
    """ Test step.

    Evaluates the performance of the network in its current state using the
    full set of test elements.

    Parameters
    ----------
    forward_function : torch.nn.Module
        The network model in the current state
    data : torch.utils.data.DataLoader or list[tuple]
        The test dataset. Because the target is recosntruct the input, the
        label associated is ignored.
    args: Namespace
        The input arguments passed at running time.

    Returns
    -------
    metrics_dict : dict
        Dictionary with the computed metrics         
    """
    logger = logging.getLogger(args.mode + '_log')

    all_metrics = dict([(m_k, []) for m_k in metric_fun])
    all_metrics['time'] = []

    load_times = []
    eval_times = []
    metrics_eval_times = dict([(m_k, []) for m_k in metric_fun])

    n_examples = 0

    load_time = perf_counter()
    for i, (x, t) in enumerate(data):
        load_time = perf_counter() - load_time
        load_times.append(load_time)

        n_examples += x.size(0)

        e_time = perf_counter()
        y = forward_function(x)
        e_time = perf_counter() - e_time

        y = y.detach().cpu().numpy().flatten()
        t = t.numpy().flatten()

        eval_time = perf_counter()
        for m_k in metric_fun.keys():
            metrics_eval_time = perf_counter()
            score = metric_fun[m_k](pred=y, target=t, pred_threshold=args.prediction_threshold)
            metrics_eval_time = perf_counter() - metrics_eval_time
            metrics_eval_times[m_k].append(metrics_eval_time)

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


def forward_undecoded_step(x, seg_model=None, dec_model=None):
    with torch.no_grad():
        y = seg_model(x)
        y = torch.sigmoid(y)
    return y


def forward_decoded_step(x, seg_model=None, dec_model=None):
    # The compressed representation is stored as an unsigned integer between [0, 255].
    # The transformation used in the dataloader transforms it into the range [-127.5, 127.5].
    # However, the synthesis track of the segmentation task works better if the compressed representation is in the range [-1, 1].
    # For this reason the tensor x is divided by 127.5.
    with torch.no_grad():
        x_brg = dec_model.module.inflate(x, color=False)
        y = seg_model(x / 127.5, x_brg[:0:-1])
        y = torch.sigmoid(y)
    return y


def forward_parallel_decoded_step(x, seg_model=None, dec_model=None):
    with torch.no_grad():
        x_brg = dec_model.module.inflate(x, color=False)
        y = seg_model(x / 127.5, x_brg[:0:-1])
        y = torch.sigmoid(y)
    return y


def setup_network(state, autoencoder_state=None, use_gpu=False):
    """ Setup a nerual network for object segmentation.

    Parameters
    ----------
    state : dict
        Checkpoint state saved during the network training.

    Returns
    -------
    forward_function : function
        The function to be used as feed-forward step.
    """
    # When the model works on compressed representation, tell the dataloader to obtain the compressed input and normal size target
    if ('Decoder' in state['args']['model_type'] and autoencoder_state is None) or 'NoBridge' in state['args']['model_type']:
        state['args']['use_bridge'] = False
    else:
        state['args']['use_bridge'] = True

    if autoencoder_state is not None:
        dec_model = models.Synthesizer(**autoencoder_state['args'])
        dec_model.load_state_dict(autoencoder_state['decoder'])

        dec_model = nn.DataParallel(dec_model)
        if use_gpu:
            dec_model.cuda()

        dec_model.eval()
        state['args']['use_bridge'] = True
    else:
        dec_model = None

    seg_model_class = seg_model_types.get(state['args']['model_type'], None)
    if seg_model_class is None:
        raise ValueError('Model type %s not supported' % state['args']['model_type'])

    seg_model = seg_model_class(**state['args'])
    seg_model.load_state_dict(state['model'])

    if use_gpu:
        seg_model = nn.DataParallel(seg_model)
        seg_model.cuda()

    if 'Decoder' in state['args']['model_type']:
        state['args']['compressed_input'] = True

        if dec_model is None:
            dec_model = seg_model
    else:
        state['args']['compressed_input'] = False

    # Define what funtion use in the feed-forward step
    if seg_model is not None and dec_model is None:
        # Segmentation w/o decoder
        forward_function = partial(forward_undecoded_step, seg_model=seg_model, dec_model=dec_model)

    elif seg_model is not None and dec_model is not None:
        # Segmentation w/ decoder
        if use_gpu:
            forward_function = partial(forward_parallel_decoded_step, seg_model=seg_model, dec_model=dec_model)
        else:
            forward_function = partial(forward_decoded_step, seg_model=seg_model, dec_model=dec_model)

    return forward_function


def main(args):
    """ Set up the training environment

    Parameters
    ----------
    args : dict or Namespace
        The set of parameters passed to the different constructors to set up the convolutional autoencoder training
    """
    logger = logging.getLogger(args.mode + '_log')

    state = torch.load(args.trained_model, map_location=None if args.use_gpu else 'cpu')

    # If a decoder model is passed as argument, use the decoded step version of the feed-forward step
    autoencoder_state = None
    if args.autoencoder_model is not None:
        autoencoder_state = torch.load(args.autoencoder_model, map_location=None if args.use_gpu else 'cpu')

    forward_function = setup_network(state, autoencoder_state=autoencoder_state, use_gpu=args.use_gpu)

    # Log the training setup
    logger.info('Network architecture:')
    logger.info(forward_function)

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    if not args.source_format.startswith('.'):
        args.source_format = '.' + args.source_format

    if not hasattr(args, "test_size"):
        args.test_size = -1

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    if 'compressed' in args.data_group:
        compressed_input = True
    transform, _, _ = utils.get_zarr_transform(normalize=True, compressed_input=compressed_input)

    test_data = utils.LabeledZarrDataset(root=args.data_dir,
                                         dataset_size=args.test_dataset_size,
                                         data_mode=args.data_mode,
                                         patch_size=args.patch_size,
                                         offset=0,
                                         transform=transform,
                                         source_format=args.source_format,
                                         compressed_input=compressed_input,
                                         data_group=args.data_group,
                                         label_group=args.labels_group,
                                         labels_data_axes=args.labels_data_axes,
                                         workers=args.workers)
    test_queue = DataLoader(test_data, batch_size=args.batch_size,
                            num_workers=args.workers,
                            shuffle=args.shuffle_test,
                            pin_memory=True,
                            worker_init_fn=utils.zarrdataset_worker_init)

    if args.test_size < 0:
        args.test_size = len(test_data)

    all_metrics_stats = test(forward_function, test_queue, args)
    torch.save(all_metrics_stats, os.path.join(args.log_dir, 'metrics_stats_%s%s.pth' % (args.seed, args.log_identifier)))

    for m_k in list(metric_fun.keys()) + ['time']:
        avg_metrics = '%s=%0.4f (+-%0.4f)' % (m_k, all_metrics_stats[m_k + '_stats']['avg'], all_metrics_stats[m_k + '_stats']['std'])
        logger.debug('==== Test metrics {}'.format(avg_metrics))


if __name__ == '__main__':
    args = utils.get_args(task='segmentation', mode='test')

    utils.setup_logger(args)

    main(args)

    logging.shutdown()
