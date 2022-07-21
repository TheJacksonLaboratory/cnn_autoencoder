import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel
import torch.optim as optim

from models import InceptionV3Age, MobileNetAge, ResNetAge, ViTAge, EarlyStoppingPatience
from utils import get_testing_args, setup_logger, get_data

from functools import reduce
from inspect import signature

scheduler_options = {"ReduceOnPlateau": optim.lr_scheduler.ReduceLROnPlateau}
model_options = {"InceptionV3": InceptionV3Age, "MobileNet": MobileNetAge, "ResNet": ResNetAge, "ViT": ViTAge}


def test_step(age_model, data, criterion, args):
    """ Test step.
    Evaluates the performance of the network in its current state using the full set of validation elements.

    Parameters
    ----------
    age_model : torch.nn.Module
        The network model in the current state
    data : torch.utils.data.DataLoader or list[tuple]
        The validation dataset. Because the target is recosntruct the input, the label associated is ignored
    criterion : function or torch.nn.Module
        The loss criterion to evaluate the network's performance
    args: Namespace
        The input arguments passed at running time
    
    Returns
    -------
    mean_loss : float
        Mean value of the criterion function over the full set of validation elements
    """
    logger = logging.getLogger(args.mode + '_log')

    age_model.eval()
    sum_loss = 0
    sum_acc = 0
    total_examples = 0

    with torch.no_grad():
        for i, (x, t) in enumerate(data):
            y = age_model(x)

            if isinstance(y, tuple):
                y, aux = y

            loss = criterion(y.squeeze(), t.long())
            loss = torch.mean(loss)
            sum_loss += loss.item()

            curr_acc = torch.sum(y.detach().cpu().squeeze().argmax(dim=1) == t.long())
            curr_batch_size = x.size(0)

            sum_acc += curr_acc
            total_examples += curr_batch_size

            if i % max(1, int(0.1 * len(data))) == 0:
                logger.debug('\t[{:04d}/{:04d}] Testing Loss {:.4f} ({:.4f}). Accuracy {:.4f} ({:.4f})'.format(
                        i, len(data), 
                        loss.item(), sum_loss / (i+1),
                        curr_acc / curr_batch_size, sum_acc / total_examples
                        )
                    )

    mean_loss = sum_loss / len(data)
    acc = sum_acc / total_examples
    return mean_loss, acc


def setup_network(checkpoint):
    """ Setup a nerual network for image compression/decompression.

    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. All the parameters are passed directly to the model constructor.
        This way, the constructor can take the parameters needed that have been passed by the user.
    
    Returns
    -------
    age_model : nn.Module
        The convolutional neural network autoencoder model.
    """

    # The autoencoder model contains all the modules
    age_model = model_options[checkpoint['args']['model_type']](**checkpoint['args'])
    age_model.load_state_dict(checkpoint['model'])

    # If there are more than one GPU, DataParallel handles automatically the distribution of the work
    age_model = nn.DataParallel(age_model)
    if args.gpu:
        age_model.cuda()


    return age_model


def setup_criteria(args, gpu=True):
    """ Setup a loss function for the neural network optimization.

    Parameters
    ----------
    args : Namespace
        The input arguments passed at running time. All the parameters are passed directly to the criteria constructors.
            
    Returns
    -------
    criterion : nn.Module
        The loss function that is used as target to optimize the parameters of the nerual network.
    """

    # Loss function
    if args['criterion'] == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError('Criterion \'%s\' not supported' % args['criterion'])

    criterion = nn.DataParallel(criterion)
    if gpu:
        criterion = criterion.cuda()

    return criterion


def get_checkpoint(checkpoint, gpu=True):
    """ Resume training from a previous checkpoint

    Parameters
    ----------
    checkpoint : str
        Path to a previous training checkpoint
    gpu : bool
        Wether use GPUs to train the neural network or not    
    """

    if not gpu:
        checkpoint_state = torch.load(checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint_state = torch.load(checkpoint)
    
    return checkpoint_state


def main(args):
    """ Set up the training environment

    Parameters
    ----------
    args : dict or Namespace
        The set of parameters passed to the different constructors to set up the convolutional autoencoder training
    """
    logger = logging.getLogger(args.mode + '_log')

    training_checkpoint = get_checkpoint(args.trained_model, gpu=args.gpu)    
    args.merge_labels = training_checkpoint['args']['merge_labels']

    age_model = setup_network(training_checkpoint)
    criterion = setup_criteria(training_checkpoint['args'], gpu=args.gpu)

    # Log the training setup
    logger.info('Network architecture:')
    logger.info(age_model)
    
    logger.info('\nCriterion:')
    logger.info(criterion)
    
    test_data = get_data(args)
    test_loss, test_acc = test_step(age_model, test_data, criterion, args)
    logger.debug('Testing Loss {:.4f}. Accuracy {:.4f}'.format(
            test_loss, test_acc)
        )

if __name__ == '__main__':
    args = get_testing_args()
    args.task = 'classification'
    args.map_labels = True
    args.num_classes = 4

    setup_logger(args)

    main(args)
    
    logging.shutdown()