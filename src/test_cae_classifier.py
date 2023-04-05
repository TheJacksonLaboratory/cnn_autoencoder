import os
import logging

import torch
import zarr
import dask
import dask.array as da
from skimage.measure import label

from numcodecs import Blosc

import numpy as np

import models
import utils
from train_cae_ms import setup_network

from tqdm import tqdm


def save_pred2zarr(save_filename, im_id, x, target, pred, seg_threshold,
                   batch_size, 
                   patch_size,
                   num_classes,
                   compute_components_metrics,
                   save_input=False):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)
    
    if pred.ndim == 4:
        h, w = target.shape[-2:]
        if target.shape[1] > 1 and num_classes == 1:
            target = target[:, 1:]

        target = target.reshape(num_classes, h, w)
        target_chunks_shape = (1, patch_size, patch_size)

        pred = pred.reshape(num_classes, h, w)
        pred_chunks_shape = (num_classes, patch_size, patch_size)

    else:
        target = target.reshape(-1, 1)
        target_chunks_shape = (batch_size, 1)

        pred = pred.reshape(-1, num_classes)
        pred_chunks_shape = (batch_size, num_classes)

    if num_classes > 1:
        pred_scores = torch.softmax(pred, dim=1).numpy()
        pred_class = torch.argmax(pred, dim=1).reshape(-1, 1).numpy()
        target = target.numpy()

    else:
        pred_scores = torch.sigmoid(pred.detach()).numpy()
        pred_class = (pred > seg_threshold).to(torch.bool).numpy()
        target = target.to(torch.bool).numpy()

    z_grp = zarr.open(save_filename)

    if save_input:
        x = x.numpy()
        z_grp.create_dataset('input/%i/0' % im_id, data=x,
                         shape=x.shape,
                         chunks=True,
                         dtype=x.dtype,
                         compressor=compressor,
                         overwrite=True)

    z_grp.create_dataset('target/%i/0' % im_id, data=target,
                         shape=target.shape,
                         chunks=target_chunks_shape,
                         dtype=target.dtype,
                         compressor=compressor,
                         overwrite=True)

    z_grp.create_dataset('scores/%i/0' % im_id, data=pred_scores,
                         shape=pred_scores.shape,
                         chunks=pred_chunks_shape,
                         dtype=pred_scores.dtype,
                         compressor=compressor,
                         overwrite=True)

    z_grp.create_dataset('class/%i/0' % im_id, data=pred_class,
                         shape=pred_class.shape,
                         chunks=target_chunks_shape,
                         dtype=pred_class.dtype,
                         compressor=compressor,
                         overwrite=True)

    if compute_components_metrics:
        t_lab, n_objs = label(target, return_num=True, connectivity=2)

        for k in range(1, n_objs + 1):
            _, cc_y, cc_x = np.nonzero(t_lab == k)
            cc_bbox = (slice(None),
                       slice(max(0, cc_y.min() - 1),
                             min(h, cc_y.max() + 2),
                             1),
                       slice(max(0, cc_x.min() - 1),
                             min(w, cc_x.max() + 2),
                             1))

            target_k = target[cc_bbox]
            pred_scores_k = pred_scores[cc_bbox]
            pred_class_k = pred_class[cc_bbox]

            if save_input:
                x_k = x[cc_bbox]
                z_grp.create_dataset('input/%i/%i' % (im_id, k),
                                     chunks=True,
                                     data=x_k,
                                     shape=x_k.shape,
                                     dtype=x_k.dtype,
                                     compressor=compressor,
                                     overwrite=True)

            z_grp.create_dataset('target/%i/%i' % (im_id, k),
                                 chunks=True,
                                 data=target_k,
                                 shape=target_k.shape,
                                 dtype=target_k.dtype,
                                 compressor=compressor,
                                 overwrite=True)

            z_grp.create_dataset('scores/%i/%i' % (im_id, k),
                                 chunks=True,
                                 data=pred_scores_k,
                                 shape=pred_scores_k.shape,
                                 dtype=pred_scores_k.dtype,
                                 compressor=compressor,
                                 overwrite=True)

            z_grp.create_dataset('class/%i/%i' % (im_id, k),
                                 chunks=True,
                                 data=pred_class_k,
                                 shape=pred_class_k.shape,
                                 dtype=pred_class_k.dtype,
                                 compressor=compressor,
                                 overwrite=True)


def infer(model, test_data, args):
    """Inference for classification/segmentation models.

    Parameters
    ----------
    model : dict
        The modules of the model to be trained as a dictionary
    test_data : torch.utils.data.DataLoader or list[tuple]
        The input and respective labels of the test dataset.
    args : Namespace
        The input arguments passed at running time

    Returns
    -------
    completed : bool
        Whether the training was sucessfully completed or it was interrupted
    """
    logger = logging.getLogger(args.mode + '_log')
    completed = False

    log_str = 'Test metrics'

    if args.progress_bar:
        q = tqdm(total=len(test_data), desc=log_str, position=0)

    test_forward_step = models.decorate_trainable_modules(
        None, enabled_modules=model.keys())

    for i, (x, t) in enumerate(test_data):
        output = test_forward_step(x, model)

        if output.get('t_pred', None) is not None:
            pred = output['t_pred']
        else:
            pred = output['s_pred']

        save_filename = os.path.join(args.log_dir,
                                     f'output{args.log_identifier}.zarr')
        save_pred2zarr(save_filename, i, x.cpu(), t.cpu(), pred.detach().cpu(),
                       args.seg_threshold,
                       args.batch_size,
                       args.patch_size,
                       args.num_classes,
                       args.compute_components_metrics,
                       args.save_input)

        if (i % max(1, int(len(test_data) * 0.1))) == 0:
            log_str = 'Test metrics'
            metrics = utils.compute_metrics_per_image(
                pred, t, top_k=5, seg_threshold=args.seg_threshold)

            for k, m in metrics.items():
                m = np.nanmean(m)
                log_str += ' {}:{:.3f}'.format(k, m)

            logger.info(log_str)

        if args.progress_bar:
            q.set_description(log_str)
            q.update()

    else:
        completed = True

    if args.progress_bar:
        q.close()

    return completed
            

def compute_roc_curve(pred, target, component, args):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    pred = pred.compute()
    target = target.compute()

    fpr, tpr, thrsh, roc_auc = utils.compute_roc_curve(pred, target)

    save_filename = os.path.join(args.log_dir,
                                 f'output{args.log_identifier}.zarr')

    z_grp = zarr.open(save_filename)

    z_grp.create_dataset(component + '/tpr', data=tpr, shape=tpr.shape,
                         chunks=True,
                         dtype=tpr.dtype,
                         compressor=compressor,
                         overwrite=True)

    z_grp.create_dataset(component + '/fpr', data=fpr, shape=fpr.shape,
                         chunks=True,
                         dtype=fpr.dtype,
                         compressor=compressor,
                         overwrite=True)

    z_grp.create_dataset(component + '/thrsh', data=thrsh, shape=thrsh.shape,
                         chunks=True,
                         dtype=thrsh.dtype,
                         compressor=compressor,
                         overwrite=True)

    return roc_auc


def compute_metrics(args):
    logger = logging.getLogger(args.mode + '_log')
    completed = False

    all_targets = []
    all_pred_scores = []
    all_pred_classes = []

    all_targets_objs = []
    all_pred_scores_objs = []
    all_pred_classes_objs = []

    z = zarr.open(os.path.join(args.log_dir,
                               f'output{args.log_identifier}.zarr'), 'r')

    for i in z['target'].group_keys():
        for k in z['target/' + i].array_keys():
            pred_k = da.from_zarr(
                os.path.join(args.log_dir, f'output{args.log_identifier}.zarr'),
                component='scores/' + i + '/' + k)

            pred_class_k = da.from_zarr(
                os.path.join(args.log_dir, f'output{args.log_identifier}.zarr'),
                component='class/'  + i + '/' + k)

            target_k = da.from_zarr(
                os.path.join(args.log_dir, f'output{args.log_identifier}.zarr'),
                component='target/'  + i + '/' + k)

            if pred_k.ndim > 2:
                pred_k = np.moveaxis(pred_k, 0, -1)
                pred_k = np.reshape(pred_k, (-1, args.num_classes))

                pred_class_k = np.moveaxis(pred_class_k, 0, -1)
                pred_class_k = np.reshape(pred_class_k, (-1, args.num_classes))

                target_k = np.moveaxis(target_k, 0, -1)
                target_k = np.reshape(target_k, (-1, args.num_classes))

            if k == '0':
                all_pred_scores.append(pred_k)
                all_pred_classes.append(pred_class_k)
                all_targets.append(target_k)
            else:
                all_pred_scores_objs.append(pred_k)
                all_pred_classes_objs.append(pred_class_k)
                all_targets_objs.append(target_k)

    pred_scores = da.concatenate(all_pred_scores, axis=0)
    pred_class = da.concatenate(all_pred_classes, axis=0)
    target = da.concatenate(all_targets, axis=0)

    top_k = min(5, args.num_classes)
    pred_class_top = da.topk(pred_scores, top_k, axis=-1)

    metrics = utils.compute_class_metrics_dask(pred_class, target,
                                               args.num_classes,
                                               pred_class_top)

    if args.num_classes == 1:
        metrics['auc'] = compute_roc_curve(pred_scores, target, 'image_level',
                                           args)

    log_str = f'Test metrics per image'
    for m, v in metrics.items():
        log_str += ' {}:{:.3f}'.format(m, v)

    logger.info(log_str)

    if len(all_pred_scores_objs):
        pred_scores_obj = da.concatenate(all_pred_scores_objs, axis=0)
        pred_class_obj = da.concatenate(all_pred_classes_objs, axis=0)
        target_obj = da.concatenate(all_targets_objs, axis=0)

        pred_class_top_obj = da.topk(pred_scores_obj, top_k, axis=-1)

        metrics_obj = utils.compute_class_metrics_dask(pred_class_obj,
                                                       target_obj,
                                                       args.num_classes,
                                                       pred_class_top_obj)
        if args.num_classes == 1:
            metrics_obj['auc'] = compute_roc_curve(pred_scores_obj, target_obj,
                                                   'object_level',
                                                   args)

        log_str = 'Test metrics per object'
        for m, v in metrics_obj.items():
            log_str += ' {}:{:.3f}'.format(m, v)

        logger.info(log_str)

    # Return True if the testing finished sucessfully
    return completed


def test(args):
    """Set up the testing environment

    Parameters
    ----------
    args : dict or Namespace
        The set of parameters passed to the different constructors to set up
        the convolutional autoencoder training.
    """
    logger = logging.getLogger(args.mode + '_log')
    args.criterion = "CE"

    test_data, num_classes = utils.get_data(args)
    args.num_classes = num_classes

    model = setup_network(args, train=False)

    # Log the training setup
    logger.info('Network architecture:')
    for k in model.keys():
        logger.info('\n{}'.format(k))
        logger.info(model[k])

    infer(model, test_data, args)
    compute_metrics(args)


if __name__ == '__main__':
    args = utils.get_args(task='encoder', mode='test')

    utils.setup_logger(args)

    test(args)

    logging.shutdown()
