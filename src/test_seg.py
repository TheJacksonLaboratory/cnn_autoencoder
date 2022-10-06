import logging
import os

from time import perf_counter
from skimage.filters.thresholding import threshold_otsu
from skimage.measure import label, regionprops
from sklearn.metrics import (accuracy_score,
                             roc_auc_score,
                             roc_curve,
                             recall_score,
                             precision_score,
                             average_precision_score,
                             precision_recall_curve,
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
        elif sum(pred > seg_threshold) < 1.0:
            prec = 0
        else:
            prec = precision_score(target, pred > seg_threshold)
    except ValueError:
        prec = -1
    return prec


def compute_avg_prec(pred, target, seg_threshold=None):
    try:
        if target.sum() < 1.0:
            avg_prec_score = -1
        else:
            avg_prec_score = average_precision_score(target, pred)
    except ValueError:
        avg_prec_score = -1
    return avg_prec_score

"""
Available metrics (can add more later):
    roc=Area under the ROC curve
    f1=Dice coefficient also known as F1 score
    acc=Prediction accuracy
    recall=Sensitivity of the model
    prec=Precision of the model
    avg_prec_score=Average precision score
"""
metric_fun = {'roc': compute_roc,
              'f1': compute_f1,
              'acc': compute_acc,
              'recall': compute_recall,
              'prec': compute_prec,
              'avg_prec_score': compute_avg_prec}


def compute_metrics(y, t, seg_threshold):
    y_flat = y.flatten()
    t_flat = t.flatten()
    all_metrics = {}
    eval_time = perf_counter()
    for m_k in metric_fun.keys():
        score = metric_fun[m_k](pred=y_flat, target=t_flat,
                                seg_threshold=seg_threshold)

        if score >= 0.0:
            all_metrics[m_k] = score
        else:
            all_metrics[m_k] = np.nan

    eval_time = perf_counter() - eval_time
    all_metrics['evaluation_time'] = eval_time
    return all_metrics


def compute_metrics_per_object(y, t, seg_threshold):
    t_ccs = label(t)
    t_ccs_props = regionprops(t_ccs)

    all_cc_metrics = {}

    all_cc_labels = []
    all_cc_predictions = []

    for cc_i, cc in enumerate(t_ccs_props):
        cc_slice = (slice(cc.bbox[0], cc.bbox[2], 1),
                    slice(cc.bbox[1], cc.bbox[3], 1))
        cc_pred = y[cc_slice]
        t_cc = np.copy(t[cc_slice])
        t_cc[t_cc != cc_i] = 0
        t_cc = t_cc > 0

        cc_metrics = compute_metrics(cc_pred, t_cc,
                                     seg_threshold=seg_threshold)

        for k in cc_metrics.keys():
            if k not in all_cc_metrics:
                all_cc_metrics[k] = []

            all_cc_metrics[k].append(cc_metrics[k])

        all_cc_labels.append(t_cc.flatten())
        all_cc_predictions.append(cc_pred.flatten())

    if len(t_ccs_props) < 1:
        for m_k in metric_fun.keys():
            all_cc_metrics[m_k] = [np.nan]
        all_cc_metrics['evaluation_time'] = [np.nan]

    return all_cc_metrics, all_cc_labels, all_cc_predictions


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
        Dictionary with the computed metrics.
    """
    logger = logging.getLogger(args.mode + '_log')

    all_metrics = {'execution_time': [], 'seg_threshold': []}

    load_times = []

    n_examples = 0
    all_predictions = []
    all_gt_labels = []
    all_cc_predictions = []
    all_cc_gt_labels = []

    load_time = perf_counter()
    for i, (x, t) in enumerate(data):
        load_time = perf_counter() - load_time
        load_times.append(load_time)

        n_examples += x.size(0)

        e_time = perf_counter()
        with torch.no_grad():
            y = segment.segment_block(x, forward_function)
        e_time = perf_counter() - e_time

        y = y.cpu().numpy().squeeze()
        t = t.cpu().numpy().squeeze()

        # Compute a threshold according to the prediction values
        seg_threshold = threshold_otsu(image=y)

        all_metrics['execution_time'].append(e_time)
        all_metrics['seg_threshold'].append(seg_threshold)

        img_metrics = compute_metrics(y, t, seg_threshold)
        (cc_metrics,
         cc_labels,
         cc_preds) = compute_metrics_per_object(y, t, seg_threshold)

        all_predictions.append(y.flatten())
        all_gt_labels.append(t.flatten())

        all_cc_gt_labels += cc_labels
        all_cc_predictions += cc_preds

        for k in img_metrics.keys():
            if k not in all_metrics.keys():
                all_metrics[k] = []
                all_metrics[k + '_cc'] = []

            all_metrics[k].append(img_metrics[k])
            all_metrics[k + '_cc'] += cc_metrics[k]

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
    all_cc_gt_labels = np.concatenate(all_cc_gt_labels)
    all_cc_predictions = np.concatenate(all_cc_predictions)

    all_metrics['roc_all'] = compute_roc(all_predictions, all_gt_labels)
    all_metrics['avg_prec_score_all'] = compute_avg_prec(all_predictions, all_gt_labels)

    fpr, tpr, thresh = roc_curve(all_gt_labels, all_predictions)
    all_metrics['fpr_all'] = fpr
    all_metrics['tpr_all'] = tpr
    all_metrics['roc_thresh_all'] = thresh

    prec, rec, thresh = precision_recall_curve(all_gt_labels, all_predictions)
    all_metrics['prec_all'] = prec
    all_metrics['rec_all'] = rec
    all_metrics['prec_rec_thresh_all'] = thresh

    # Compute ROC and precision-recall curves from the connected-level analysis
    all_metrics['roc_all_cc'] = compute_roc(all_cc_predictions, all_cc_gt_labels)
    all_metrics['avg_prec_score_all_cc'] = compute_avg_prec(all_cc_predictions, all_cc_gt_labels)

    fpr, tpr, thresh = roc_curve(all_cc_gt_labels, all_cc_predictions)
    all_metrics['fpr_all_cc'] = fpr
    all_metrics['tpr_all_cc'] = tpr
    all_metrics['roc_thresh_all_cc'] = thresh

    prec, rec, thresh = precision_recall_curve(all_cc_gt_labels, all_cc_predictions)
    all_metrics['prec_all_cc'] = prec
    all_metrics['rec_all_cc'] = rec
    all_metrics['prec_rec_thresh_all_cc'] = thresh

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

    avg_metrics = '%s=%0.4f (+-%0.4f)' % (
        'execution_time_stats',
        all_metrics_stats['execution_time_stats']['avg'],
        all_metrics_stats['execution_time_stats']['std'])
    logger.debug('==== Test metrics {}'.format(avg_metrics))

    for m_k in metric_fun.keys():
        avg_metrics = '%s (image level)=%0.4f (+-%0.4f)' % (
            m_k,
            all_metrics_stats[m_k + '_stats']['avg'],
            all_metrics_stats[m_k + '_stats']['std'])
        logger.debug('==== Test metrics {}'.format(avg_metrics))
        avg_metrics = '%s (component level)=%0.4f (+-%0.4f)' % (
            m_k,
            all_metrics_stats[m_k + '_cc_stats']['avg'],
            all_metrics_stats[m_k + '_cc_stats']['std'])
        logger.debug('==== Test metrics {}'.format(avg_metrics))


if __name__ == '__main__':
    args = utils.get_args(task='segmentation', mode='test')

    utils.setup_logger(args)

    main(args)

    logging.shutdown()
