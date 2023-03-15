import logging

import torch
import numpy as np

import utils
from train_cae_ms import resume_checkpoint, setup_network, forward_step

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

    for k in model.keys():
        model[k].eval()

    if args.progress_bar:
        q = tqdm(total=len(test_data), desc="Testing", position=0)

    n_samples = 0
    true_pos = 0
    true_pos_top = 0

    for x, t in test_data:
        output = forward_step(x, model, trainable_modules=None)
        pred = output['t_pred'].detach().cpu().argmax(dim=1)
        pred_top = torch.topk(output['t_pred'].detach().cpu(), k=5)[1]

        true_pos += (pred == t).sum().item()
        true_pos_top += (pred_top == t.view(-1, 1)).any(dim=1).sum().item()
        n_samples += x.size(0)

        # End of training step
        if args.progress_bar:
            q.set_description('Test metrics acc:{:.3f} acc_top:{:.3f}'.format(
                true_pos / n_samples, true_pos_top / n_samples))
            q.update()

    else:
        completed = True

    if args.progress_bar:
        q.close()

    acc = true_pos / n_samples
    acc_top = true_pos_top / n_samples
    log_string = 'Test metrics acc:{:.3f} acc_top:{:.3f}'.format(acc, acc_top)

    logger.info(log_string)

    # Return True if the training finished sucessfully
    return completed


def main(args):
    """Set up the training environment

    Parameters
    ----------
    args : dict or Namespace
        The set of parameters passed to the different constructors to set up
        the convolutional autoencoder training.
    """
    logger = logging.getLogger(args.mode + '_log')

    test_data, num_classes = utils.get_data(args)
    args.num_classes = num_classes

    checkpoint = utils.load_state(args)
    model = setup_network(checkpoint['args'], use_gpu=args.gpu)

    resume_checkpoint(model, None, None, args.checkpoint,
                      gpu=args.gpu,
                      resume_optimizer=False)

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
