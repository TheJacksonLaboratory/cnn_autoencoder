import os
import logging

import torch
import zarr
from numcodecs import Blosc

import numpy as np

import models
import utils
from train_cae_ms import setup_network

from tqdm import tqdm


def test(model, test_data, args):
    """Training loop by steps.
    This loop involves validation and network training checkpoint creation.

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

    if args.progress_bar:
        q = tqdm(total=len(test_data), desc="Testing", position=0)

    test_forward_step = models.decorate_trainable_modules(
        None, enabled_modules=model.keys())

    rec_metrics = None
    true_pos = 0
    true_pos_top = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    n_samples = 0

    for i, (x, t) in enumerate(test_data):
        output = test_forward_step(x, model)

        if output.get('t_pred', None) is not None:
            pred = output['t_pred']
        else:
            pred = output['s_pred']

        metrics = utils.compute_class_metrics(pred, target=t, top_k=5,
                                              num_classes=args.num_classes,
                                              seg_threshold=args.seg_threshold)

        log_str = ''
        for k, m in metrics.items():
            log_str += ' {}:{:.3f}'.format(k, m)

        true_pos += metrics['tp']
        true_pos_top += metrics['tp_top']
        true_neg += metrics['tn']
        false_pos += metrics['fp']
        false_neg += metrics['fn']
        n_samples += metrics['p'] + metrics['n']

        if i % max(1, int(0.1 * len(test_data))) == 0:
            logger.debug(log_str)
            if rec_metrics is None:
                rec_metrics = dict((m, []) for m in metrics.keys())

            for m, v in metrics.items():
                rec_metrics[m].append(v)

        # End of training step
        if args.progress_bar:
            q.set_description(log_str)
            q.update()

        if args.save_prediction:
            compressor = Blosc(cname='zlib', clevel=9,
                               shuffle=Blosc.BITSHUFFLE)
            for k in range(x.size(0)):
                x_k = x[k].cpu().mul(255).to(torch.uint8).unsqueeze(0).unsqueeze(2).numpy()
                t_k = t[k].mul(255).to(torch.uint8).unsqueeze(0).unsqueeze(2).numpy()

                pred_k = pred[k].detach().cpu().unsqueeze(0).unsqueeze(2)
                if args.num_classes == 1:
                    pred_k = pred_k.sigmoid()
                else:
                    pred_k = pred_k.softmax(dim=1)

                pred_class_k = pred_k > args.seg_threshold
                pred_class_k = pred_class_k.mul(255).to(torch.uint8).numpy()
                pred_k = pred_k.numpy()

                z_grp = zarr.open(os.path.join(args.log_dir, 'output.zarr'))
                z_grp.create_dataset('input/%i' % (k + i * args.batch_size),
                                     data=x_k,
                                     shape=(1, 3, 1, args.patch_size, 
                                            args.patch_size),
                                     chunks=(1, 3, 1, args.patch_size,
                                             args.patch_size),
                                     dtype=np.uint8,
                                     compressor=compressor,
                                     overwrite=True)
                z_grp.create_dataset('target/%i' % (k + i * args.batch_size),
                                     data=t_k,
                                     shape=(1, 1, 1, args.patch_size, 
                                            args.patch_size),
                                     chunks=(1, 1, 1, args.patch_size,
                                             args.patch_size),
                                     dtype=np.uint8,
                                     compressor=compressor,
                                     overwrite=True)
                z_grp.create_dataset('pred/%i' % (k + i * args.batch_size),
                                     data=pred_k,
                                     shape=(1, args.num_classes, 1,
                                            args.patch_size,
                                            args.patch_size),
                                     chunks=(1, args.num_classes, 1,
                                             args.patch_size,
                                             args.patch_size),
                                     dtype=np.float32,
                                     compressor=compressor,
                                     overwrite=True)
                z_grp.create_dataset('class/%i' % (k + i * args.batch_size),
                                     data=pred_class_k,
                                     shape=(1, args.num_classes, 1,
                                            args.patch_size,
                                            args.patch_size),
                                     chunks=(1, args.num_classes, 1,
                                             args.patch_size,
                                             args.patch_size),
                                     dtype=np.uint8,
                                     compressor=compressor,
                                     overwrite=True)

    else:
        completed = True

    if args.progress_bar:
        q.close()

    acc = (true_pos + true_neg) / n_samples
    acc_top = true_pos_top / n_samples

    rec = true_pos / (true_pos + false_neg)
    prec = true_pos / (true_pos + false_pos)
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg)

    log_string = f'Test metrics acc:{acc:0.3f}'
    if args.num_classes > 1:
        log_string += f' acc_top:{acc_top:0.3f}'

    log_string += f' rec:{rec:0.3f} prec:{prec:0.3f} f1:{f1:0.3f}'

    logger.info(log_string)
    log_string = 'Test average metrics'
    for k, m in rec_metrics.items():
        m = np.nanmean(m)
        log_string += ' {}:{:.3f}'.format(k, m)

    logger.info(log_string)

    # Return True if the testing finished sucessfully
    return completed


def main(args):
    """Set up the testing environment

    Parameters
    ----------
    args : dict or Namespace
        The set of parameters passed to the different constructors to set up
        the convolutional autoencoder training.
    """
    logger = logging.getLogger(args.mode + '_log')

    test_data, num_classes = utils.get_data(args)
    args.num_classes = num_classes

    model = setup_network(args, train=False)

    # Log the training setup
    logger.info('Network architecture:')
    for k in model.keys():
        logger.info('\n{}'.format(k))
        logger.info(model[k])

    test(model, test_data, args)


if __name__ == '__main__':
    args = utils.get_args(task='encoder', mode='test')

    utils.setup_logger(args)

    main(args)

    logging.shutdown()
