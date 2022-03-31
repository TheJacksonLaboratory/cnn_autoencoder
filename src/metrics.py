import logging
import argparse
import os

from time import perf_counter
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score

import numpy as np
import zarr

import segment
import utils


def metrics_image(prediction, target, pred_threshold=0.5):
    """ Compute metrics from the predicted probability of each class for a single image.
    The predicted class is applied after applying a threshold with the given value "threshold" to the predicted probability.
    The resulting metrics can be averaged to get a global value for each metric.

    Parameters:
    ----------
    prediction : numpy.ndarray
        The predicted probability of the model for each pixel of the inpt image
    target: numpy.ndarray
        Ground-truth of the image

    Returns
    -------
    metrics_dict : Dictionary
        Dictionary with the computed metrics (roc=ROC, acc=Accuracy, f1=F1 score, recall=Recall, prec=Precision)
    """
    pred_class = prediction > pred_threshold
    try:
        roc = roc_auc_score(target, prediction)
    except:
        roc = -1
    
    try:
        f1 = f1_score(target, pred_class)
    except:
        f1 = -1

    try:
        acc = accuracy_score(target, pred_class)
    except:
        acc = -1
    
    try:
        recall = recall_score(target, pred_class)
    except:
        recall = -1
    
    try:
        prec = precision_score(target, pred_class)
    except:
        prec = -1

    metrics_dict = dict(roc=roc, f1=f1, acc=acc, recall=recall, prec=prec)

    return metrics_dict


def metrics(args):
    """ Compute the average perfomance in terms of different metrics for a set of images.
    Metrics computed: 
        Area under the ROC curve
        Binary accuracy
        Precision
        Recall
        F1 Score
    """
    # Override the destination format to 'zarr_memory' in case that it was given
    args.destination_format = 'zarr_memory'

    # Override 'is_labeled' to True, in order to get the segmentation response along with its respective ground-truth
    args.is_labeled = True
    args.mode = 'testing'

    utils.setup_logger(args)
    logger = logging.getLogger(args.mode + '_log')

    avg_metrics = dict(roc=0, acc=0, prec=0, recall=0, f1=0, time=0)

    all_preds = []
    all_targets = []

    e_time = perf_counter()
    for i, group in enumerate(segment.segment(args)):
        e_time = perf_counter() - e_time
        prediction = group['0/0'][:].flatten()
        target = group['1/0'][:].flatten()
        
        all_preds.append(prediction)
        all_targets.append(target)

        scores = metrics_image(prediction, target, pred_threshold=args.pred_threshold)

        for m_k in scores.keys():
            avg_metrics[m_k] = avg_metrics[m_k] + scores[m_k]
            logger.info('[Image %i] Metric %s: %0.4f' % (i+1, m_k, scores[m_k]))
        
        avg_metrics['time'] = avg_metrics['time'] + e_time
        logger.info('[Image %i] Execution time: %0.4f' % (i+1, e_time))
        
        # Time stamp for prediciton of next image 
        e_time = perf_counter()
    
    for m_k in avg_metrics.keys():
        avg_metrics[m_k] = avg_metrics[m_k] / (i+1)
        logger.info('Average metric %s: %0.4f' % (m_k, avg_metrics[m_k]))

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    global_scores = metrics_image(all_preds, all_targets, pred_threshold=args.pred_threshold)
    
    for m_k in global_scores.keys():
        logger.info('Global metric %s: %0.4f' % (m_k, global_scores[m_k]))
        
    logging.shutdown()
    return avg_metrics


if __name__ == '__main__':
    seg_parser = utils.get_segment_args(parser_only=True)
    
    parser = argparse.ArgumentParser(prog='Evaluate a model on a testing set (Segmentation models only)', parents=[seg_parser], add_help=False)
    
    parser.add_argument('-th', '--threshold', dest='pred_threshold', help='The prediced probability threshold used to determine if a pixel is part of the foreground or not.', default=0.5)
    
    args = utils.override_config_file(parser)
    
    avg_metrics = metrics(args)
