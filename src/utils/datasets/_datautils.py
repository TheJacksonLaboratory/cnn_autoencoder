import os
from torch.utils.data import DataLoader, random_split

from ._augs import (get_zarr_transform,
                    get_mnist_transform,
                    get_imagenet_transform,
                    get_cifar_transform)
from ._cifar import CIFAR10, CIFAR100
from ._mnist import MNIST, EMNIST
from ._zarrbased import (zarrdataset_worker_init,
                         ZarrDataset,
                         LabeledZarrDataset)
from ._imagenet import ImageFolder, ImageS3


def get_MNIST(data_dir='.', batch_size=1, val_batch_size=1, workers=0, mode='training', normalize=True, **kwargs):
    prep_trans = get_mnist_transform(mode, normalize)

    # If testing the model, return the test set from MNIST
    if mode != 'training':
        mnist_data = MNIST(root=data_dir, train=False, download=True, transform=prep_trans)
        test_queue = DataLoader(mnist_data, batch_size=batch_size, shuffle=False, num_workers=workers)
        return test_queue

    mnist_data = MNIST(root=data_dir, train=True, download=True, transform=prep_trans)

    train_ds, valid_ds = random_split(mnist_data, (55000, 5000))
    train_queue = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_queue = DataLoader(valid_ds, batch_size=val_batch_size, shuffle=False, num_workers=workers)

    return train_queue, valid_queue, 10


def get_EMNIST(data_dir='.', batch_size=1, val_batch_size=1, workers=0, mode='training', normalize=True, **kwargs):
    prep_trans = get_mnist_transform(mode, normalize)

    # If testing the model, return the test set from MNIST
    if mode != 'training':
        mnist_data = EMNIST(root=data_dir, split='byclass', train=False, download=True, transform=prep_trans)
        test_queue = DataLoader(mnist_data, batch_size=batch_size, shuffle=False, num_workers=workers)
        return test_queue

    mnist_data = EMNIST(root=data_dir, split='byclass', train=True, download=True, transform=prep_trans)

    train_ds, valid_ds = random_split(mnist_data, (628132, 69800))
    train_queue = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_queue = DataLoader(valid_ds, batch_size=val_batch_size, shuffle=False, num_workers=workers)

    return train_queue, valid_queue, 62


def get_CIFAR10(data_dir='.', batch_size=1, val_batch_size=1, workers=0,
                mode='training',
                normalize=True,
                **kwargs):
    prep_trans = get_cifar_transform(mode, normalize)

    # If testing the model, return the test set from MNIST
    if mode != 'training':
        cifar_data = CIFAR10(root=data_dir, train=False, download=True, transform=prep_trans)
        test_queue = DataLoader(cifar_data, batch_size=batch_size, shuffle=False, num_workers=workers)
        return test_queue

    cifar_data = CIFAR10(root=data_dir, train=True, download=True, transform=prep_trans)

    train_ds, valid_ds = random_split(cifar_data, (45000, 5000))
    train_queue = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_queue = DataLoader(valid_ds, batch_size=val_batch_size, shuffle=False, num_workers=workers)

    return train_queue, valid_queue, 10


def get_CIFAR100(data_dir='.', batch_size=1, val_batch_size=1, workers=0,
                 mode='training',
                 normalize=True,
                 **kwargs):
    prep_trans = get_cifar_transform(mode, normalize)

    # If testing the model, return the test set from MNIST
    if mode != 'training':
        cifar_data = CIFAR100(root=data_dir, train=False, download=True, transform=prep_trans)
        test_queue = DataLoader(cifar_data, batch_size=batch_size, shuffle=False, num_workers=workers)
        return test_queue

    cifar_data = CIFAR100(root=data_dir, train=True, download=True, transform=prep_trans)

    train_ds, valid_ds = random_split(cifar_data, (45000, 5000))
    train_queue = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_queue = DataLoader(valid_ds, batch_size=val_batch_size, shuffle=False, num_workers=workers)

    return train_queue, valid_queue, 100


def get_ImageNet(data_dir='.', batch_size=1, val_batch_size=1, workers=0,
                 mode='training',
                 normalize=True,
                 patch_size=128,
                 **kwargs):
    prep_trans = get_imagenet_transform(mode, normalize, patch_size)

    if isinstance(data_dir, list) and len(data_dir) == 1:
        data_dir = data_dir[0]

    if (isinstance(data_dir, list)
       and (data_dir[0].endswith('txt')
       or data_dir[0].startswith('s3')
       or data_dir[0].startswith('http'))
       or data_dir.endswith('txt')):

        image_dataset = ImageS3

        # If testing the model, return the validation set from MNIST
        if mode != 'training':
            imagenet_data = image_dataset(root=data_dir, transform=prep_trans)
            test_queue = DataLoader(imagenet_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
            return test_queue

        trn_data_dir = [fn for fn in data_dir if 'train' in fn][0]
        val_data_dir = [fn for fn in data_dir if 'val' in fn][0]

        train_ds = image_dataset(root=trn_data_dir, transform=prep_trans)
        valid_ds = image_dataset(root=val_data_dir, transform=prep_trans)

        train_queue = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        valid_queue = DataLoader(valid_ds, batch_size=val_batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    else:
        image_dataset = ImageFolder
        data_dir = os.path.join(data_dir, 'ILSVRC/Data/CLS-LOC/test')

        # If testing the model, return the validation set from MNIST
        if mode != 'training':
            imagenet_data = image_dataset(root=data_dir, transform=prep_trans)
            test_queue = DataLoader(imagenet_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
            return test_queue

        imagenet_data = image_dataset(root=data_dir, transform=prep_trans)

        train_size = int(len(imagenet_data) * 0.96)
        val_size = len(imagenet_data) - train_size

        train_ds, valid_ds = random_split(imagenet_data, (train_size, val_size))
        train_queue = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        valid_queue = DataLoader(valid_ds, batch_size=val_batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_queue, valid_queue, 1000


def get_zarr_dataset(data_dir='.', task='autoencoder', batch_size=1,
                     val_batch_size=1,
                     workers=0,
                     data_mode='training',
                     shuffle_train=True,
                     shuffle_val=True,
                     shuffle_test=False,
                     train_dataset_size=-1,
                     val_dataset_size=-1,
                     test_dataset_size=-1,
                     gpu=False,
                     mode='training',
                     num_classes=None,
                     **kwargs):
    """Creates a data queue using pytorch\'s DataLoader module to retrieve
    patches from images stored in zarr format.
    """

    (prep_trans,
     input_target_trans,
     target_trans) = get_zarr_transform(data_mode=data_mode, **kwargs)

    if task == 'autoencoder':
        histo_dataset = ZarrDataset

    elif task == 'segmentation':
        if mode in ['training', 'test']:
            histo_dataset = LabeledZarrDataset
        else:
            histo_dataset = ZarrDataset

    elif task == 'classification':
        if mode in ['training', 'test']:
            histo_dataset = LabeledZarrDataset
        else:
            histo_dataset = ZarrDataset

    # Modes can vary from testing, segmentation, compress, decompress, etc. For this reason, only when it is properly training, two data queues are returned, otherwise, only one queue is returned.
    if 'train' not in data_mode:
        zarr_data = histo_dataset(root=data_dir,
                                  dataset_size=test_dataset_size,
                                  data_mode='test',
                                  transform=prep_trans,
                                  intput_target_transform=input_target_trans,
                                  target_transform=target_trans,
                                  workers=workers,
                                  **kwargs)
        test_queue = DataLoader(zarr_data, batch_size=batch_size,
                                shuffle=shuffle_test,
                                num_workers=min(workers, 
                                                len(zarr_data._filenames)),
                                pin_memory=gpu,
                                worker_init_fn=zarrdataset_worker_init)
        return test_queue

    zarr_train_data = histo_dataset(root=data_dir,
                                    dataset_size=train_dataset_size,
                                    data_mode='train',
                                    transform=prep_trans,
                                    input_target_transform=input_target_trans,
                                    target_transform=target_trans,
                                    workers=workers,
                                    **kwargs)
    zarr_valid_data = histo_dataset(root=data_dir,
                                    dataset_size=val_dataset_size,
                                    data_mode='val',
                                    transform=prep_trans,
                                    input_target_transform=input_target_trans,
                                    target_transform=target_trans,
                                    workers=workers,
                                    **kwargs)

    # When training a network that expects to receive a complete image divided into patches, it is better to use shuffle_trainin=False to preserve all patches in the same batch.
    train_queue = DataLoader(zarr_train_data, batch_size=batch_size, shuffle=shuffle_train, num_workers=min(workers, len(zarr_train_data._filenames)), pin_memory=gpu, worker_init_fn=zarrdataset_worker_init)
    valid_queue = DataLoader(zarr_valid_data, batch_size=val_batch_size, shuffle=shuffle_val, num_workers=min(workers, len(zarr_valid_data._filenames)), pin_memory=gpu, worker_init_fn=zarrdataset_worker_init)

    return train_queue, valid_queue, num_classes


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

    if args_dict['dataset'] == 'MNIST':
        return get_MNIST(**args_dict)

    if args_dict['dataset'] == 'EMNIST':
        return get_EMNIST(**args_dict)

    elif args_dict['dataset'] == 'CIFAR10':
        return get_CIFAR10(**args_dict)

    elif args_dict['dataset'] == 'CIFAR100':
        return get_CIFAR100(**args_dict)

    elif args_dict['dataset'] in ['ImageNet', 'ImageNet.S3']:
        return get_ImageNet(**args_dict)

    elif args_dict['dataset'] in ['Zarr', 'Histology']:
        return get_zarr_dataset(**args_dict)

    else:
        raise ValueError('The dataset \'%s\' is not available for training.' % args_dict['dataset'])
