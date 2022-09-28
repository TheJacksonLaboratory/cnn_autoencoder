import logging
import os

from time import perf_counter
from sklearn.metrics import (accuracy_score,
                             roc_auc_score,
                             roc_curve,
                             recall_score,
                             precision_score,
                             f1_score)

import numpy as np
import torch

import models
import utils
import segment

seg_model_types = {"UNetNoBridge": models.UNetNoBridge,
                   "UNet": models.UNet,
                   "DecoderUNet": models.DecoderUNet}


def compute_roc(pred, target, seg_threshold=None):
    try:
        roc = roc_auc_score(target, pred)

    except ValueError:
        roc = -1

    return roc


def compute_f1(pred, target, seg_threshold=None):
    try:
        if target.sum() < 1.0:
            f1 = -1
        else:
            f1 = f1_score(target, pred > seg_threshold)
    except ValueError:
        f1 = -1
    return f1


def compute_acc(pred, target, seg_threshold=None):
    try:
        acc = accuracy_score(target, pred > seg_threshold)
    except ValueError:
        acc = -1
    return acc


def compute_recall(pred, target, seg_threshold=None):
    try:
        if target.sum() < 1.0:
            recall = -1
        else:
            recall = recall_score(target, pred > seg_threshold)
    except ValueError:
        recall = -1
    return recall


def compute_prec(pred, target, seg_threshold=None):
    try:
        if target.sum() < 1.0:
            prec = -1
        else:
            prec = precision_score(target, pred > seg_threshold)
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
    all_predictions = []
    all_gt_labels = []

    load_time = perf_counter()
    for i, (x, t) in enumerate(data):
        load_time = perf_counter() - load_time
        load_times.append(load_time)

        n_examples += x.size(0)

        e_time = perf_counter()
        with torch.no_grad():
            y = segment.segment_block(x, forward_function)
        e_time = perf_counter() - e_time

        y = y.cpu().numpy().flatten()
        t = t.numpy().flatten()

        all_predictions.append(y)
        all_gt_labels.append(t)

        eval_time = perf_counter()
        for m_k in metric_fun.keys():
            metrics_eval_time = perf_counter()
            score = metric_fun[m_k](pred=y, target=t,
                                    seg_threshold=args.seg_threshold)

            metrics_eval_time = perf_counter() - metrics_eval_time

            metrics_eval_times[m_k].append(metrics_eval_time)

            if score >= 0.0:
                all_metrics[m_k].append(score)
            else:
                all_metrics[m_k].append(np.nan)

        eval_time = perf_counter() - eval_time
        eval_times.append(eval_time)

        all_metrics['time'].append(e_time)

        if n_examples % max(1, int(0.1 * args.test_dataset_size)) == 0:
            avg_metrics = ''
            for m_k in all_metrics.keys():
                avg_metric = np.nanmean(all_metrics[m_k])
                avg_metrics += '%s=%0.5f ' % (m_k, avg_metric)
            logger.debug(
                '\t[{:05d}/{:05d}][{:05d}/{:05d}] '
                'Test metrics {}'.format(
                    i+1, len(data), n_examples,
                    args.test_dataset_size if args.test_dataset_size > 0 else len(data), avg_metrics))

        load_time = perf_counter()

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

        all_metrics_stats[m_k + '_stats'] = dict(avg=avg_metric,
                                                 std=std_metric,
                                                 med=med_metric,
                                                 min=min_metric,
                                                 max=max_metric)

    # Compute the ROC curve for all images to get a single curve
    all_gt_labels = np.concatenate(all_gt_labels)
    all_predictions = np.concatenate(all_predictions)

    all_metrics['roc_all'] = compute_roc(all_predictions, all_gt_labels)
    fpr, tpr, thresh = roc_curve(all_gt_labels, all_predictions)
    all_metrics['fpr'] = fpr
    all_metrics['tpr'] = tpr
    all_metrics['thresh'] = thresh

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

    # Open checkpoint from trained model state
    state = utils.load_state(args)

    (_,
     forward_fun,
     compressed_input) = segment.setup_network(state['args'],
                                               pretrained_model=state['model'],
                                               autoencoder_model=args.autoencoder_model,
                                               use_gpu=args.use_gpu)
    args.compressed_input = compressed_input

    # Log the training setup
    logger.info('Network architecture:')
    logger.info(str(forward_fun))

    # Generate a dataset from a single image to divide in patches and iterate using a dataloader
    if not args.source_format.startswith('.'):
        args.source_format = '.' + args.source_format

    test_data = utils.get_data(args)

    all_metrics_stats = test(forward_fun, test_data, args)
    torch.save(all_metrics_stats, os.path.join(args.log_dir, 'metrics_stats_%s%s.pth' % (args.seed, args.log_identifier)))

    for m_k in list(metric_fun.keys()) + ['time']:
        avg_metrics = '%s=%0.4f (+-%0.4f)' % (m_k, all_metrics_stats[m_k + '_stats']['avg'], all_metrics_stats[m_k + '_stats']['std'])
        logger.debug('==== Test metrics {}'.format(avg_metrics))


if __name__ == '__main__':
    args = utils.get_args(task='segmentation', mode='test')

    utils.setup_logger(args)

    main(args)

    logging.shutdown()
