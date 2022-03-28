from .datasets import get_zarr_dataset, get_MNIST, get_ImageNet


def get_data(args):
    """ Retrieve a queue of data pairs (input, target) in mini-batches.
    The number of outputs can vary according to the use mode (training, testing, compression, decompression, etc).
    When used for training, two data queues are returned, one for training and the other for validation.
    Otherwise, only one data queue is returned.
    """
    args_dict = args if isinstance(args, dict) else args.__dict__

    # The arguments parser stores the data dir path as a list in case that more than one path is given
    # However, when a directory is given, it should be taken directly as the root directory of the dataset
    if isinstance(args_dict['data_dir'], list) and len(args_dict['data_dir']) == 1:
        args_dict['data_dir'] = args_dict['data_dir'][0]
    
    if args.dataset == 'MNIST':
        return get_MNIST(**args_dict)

    elif args.dataset == 'ImageNet':
        return get_ImageNet(**args_dict)

    elif args.dataset in ['Zarr', 'Histology']:
        return get_zarr_dataset(**args_dict)

    else:
        raise ValueError('The dataset \'%s\' is not available for training.' % args.dataset)
