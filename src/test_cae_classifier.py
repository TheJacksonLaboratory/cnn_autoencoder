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
                   top_k=5,
                   save_input=False):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    top_k = min(top_k, num_classes)
    if pred.ndim == 4:
        h, w = target.shape[-2:]
        if target.shape[1] > 1 and num_classes == 1:
            target = target[:, 1:]

        target_chunks_shape = (1, 1, patch_size, patch_size)
        pred_chunks_shape = (1, num_classes, patch_size, patch_size)

    else:
        target = target.reshape(-1, 1)
        target_chunks_shape = (batch_size, 1)

        pred = pred.reshape(-1, num_classes)
        pred_chunks_shape = (batch_size, num_classes)

    if num_classes > 1:
        pred_scores = torch.softmax(pred, dim=1).numpy()
        pred_class = torch.argmax(pred, dim=1).unsqueeze(1).numpy()
        pred_class_top = torch.topk(pred, k=top_k, dim=1)[1].numpy()
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

    if num_classes > 1:
        z_grp.create_dataset('topk/%i/0' % im_id, data=pred_class_top,
                             shape=pred_class_top.shape,
                             chunks=target_chunks_shape,
                             dtype=pred_class_top.dtype,
                             compressor=compressor,
                             overwrite=True)

    if compute_components_metrics:
        t_lab, n_objs = label(target, return_num=True, connectivity=2)

        for k in range(1, n_objs + 1):
            _, _, cc_y, cc_x = np.nonzero(t_lab == k)
            cc_bbox = (slice(None),
                       slice(None),
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

            if num_classes > 1:
                pred_class_top_k = pred_class_top[cc_bbox]
                z_grp.create_dataset('topk/%i/%i' % (im_id, k),
                                     chunks=True,
                                     data=pred_class_top_k,
                                     shape=pred_class_top_k.shape,
                                     dtype=pred_class_top_k.dtype,
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
        save_pred2zarr(save_filename, i, x.cpu().mul(255).to(torch.uint8),
                       t.cpu(),
                       pred.detach().cpu(),
                       args.seg_threshold,
                       args.batch_size,
                       args.patch_size,
                       args.num_classes,
                       args.compute_components_metrics,
                       top_k=5,
                       save_input=args.save_input)

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


def compute_metrics(args, object_level=False):
    logger = logging.getLogger(args.mode + '_log')
    completed = False

    all_targets = []
    all_pred_scores = []
    all_pred_classes = []
    all_pred_classes_top = []

    type_level = 'object_level' if object_level else 'image_level'

    z = zarr.open(os.path.join(args.log_dir,
                               f'output{args.log_identifier}.zarr'), 'r')

    for i in z['target'].group_keys():
        if object_level:
            array_keys = set(z['target/' + i].array_keys()) - set('0')
        else:
            array_keys = ['0']

        for k in array_keys:
            if args.num_classes == 1:
                pred_k = da.from_zarr(
                    os.path.join(args.log_dir,
                                 f'output{args.log_identifier}.zarr'),
                    component='scores/' + i + '/' + k)
            else:
                pred_k = None

            pred_class_k = da.from_zarr(
                os.path.join(args.log_dir,
                             f'output{args.log_identifier}.zarr'),
                component='class/'  + i + '/' + k)

            target_k = da.from_zarr(
                os.path.join(args.log_dir,
                             f'output{args.log_identifier}.zarr'),
                component='target/'  + i + '/' + k)

            if 'topk' in z.group_keys():
                pred_class_top_k = da.from_zarr(
                    os.path.join(args.log_dir,
                                 f'output{args.log_identifier}.zarr'),
                    component='topk/'  + i + '/' + k)
            else:
                pred_class_top_k = None

            if object_level:
                pred_class_k = np.moveaxis(pred_class_k, 1, -1).reshape(-1, 1)
                target_k = np.moveaxis(target_k, 1, -1).reshape(-1, 1)

            all_pred_classes.append(pred_class_k)
            all_targets.append(target_k)

            if pred_k is not None:
                if object_level:
                    pred_k = np.moveaxis(pred_k, 1, -1)
                    pred_k = pred_k.reshape(-1, args.num_classes)

                all_pred_scores.append(pred_k)

            if pred_class_top_k is not None:
                if object_level:
                    pred_class_top_k = np.moveaxis(pred_class_top_k, 1, -1)
                    pred_class_top_k = pred_class_top_k.reshape(-1, top_k)

                all_pred_classes_top.append(pred_class_top_k)

    pred_class = da.concatenate(all_pred_classes, axis=0)
    target = da.concatenate(all_targets, axis=0)

    if not object_level:
        pred_class = np.moveaxis(pred_class, 1, -1).reshape(-1, 1)
        target = np.moveaxis(target, 1, -1).reshape(-1, 1)

    if len(all_pred_classes_top):
        pred_class_top = da.concatenate(all_pred_classes_top, axis=0)
        top_k = pred_class_top.shape[1]
        if not object_level:
            pred_class_top = np.moveaxis(pred_class_top, 1, -1)
            pred_class_top = pred_class_top.reshape(-1, top_k)

    else:
        pred_class_top = None

    metrics = utils.compute_class_metrics_dask(pred_class, target,
                                               args.num_classes,
                                               pred_class_top)

    if args.num_classes == 1:
        pred_scores = da.concatenate(all_pred_scores, axis=0)
        if not object_level:
            pred_scores = np.moveaxis(pred_scores, 1, -1)
            pred_scores = pred_scores.reshape(-1, args.num_classes)

        metrics['auc'] = compute_roc_curve(pred_scores, target,
                                           type_level,
                                           args)

    log_str = 'Test metrics at ' + type_level
    for m, v in metrics.items():
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

    if not args.metrics_only:
        infer(model, test_data, args)

    compute_metrics(args, object_level=False)

    if args.compute_components_metrics:
        compute_metrics(args, object_level=True)


if __name__ == '__main__':
    args = utils.get_args(task='encoder', mode='test')

    utils.setup_logger(args)

    test(args)

    logging.shutdown()
