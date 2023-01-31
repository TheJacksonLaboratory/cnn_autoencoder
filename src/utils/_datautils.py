import os
from torch.utils.data import DataLoader, random_split

from .datasets import (get_zarr_transform,
                       get_mnist_transform,
                       get_imagenet_transform,
                       MNIST,
                       zarrdataset_worker_init,
                       ZarrDataset,
                       LabeledZarrDataset,
                       ImageFolder,
                       ImageS3)


def get_MNIST(data_dir='.', batch_size=1, val_batch_size=1, workers=0,
              mode='training',
              normalize=True,
              **kwargs):
    prep_trans = get_mnist_transform(mode, normalize)

    # If testing the model, return the test set from MNIST
    if mode != 'training':
        mnist_data = MNIST(root=data_dir, train=False, download=False,
                           transform=prep_trans)
        test_queue = DataLoader(mnist_data, batch_size=batch_size,
                                shuffle=False,
                                num_workers=workers)
        return test_queue

    mnist_data = MNIST(root=data_dir, train=True, download=False,
                       transform=prep_trans)

    train_ds, valid_ds = random_split(mnist_data, (55000, 5000))
    train_queue = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=workers)
    valid_queue = DataLoader(valid_ds, batch_size=val_batch_size,
                             shuffle=False,
                             num_workers=workers)

    return train_queue, valid_queue


def get_ImageNet(data_dir='.', batch_size=1, val_batch_size=1, workers=0,
                 mode='training',
                 normalize=True,
                 **kwargs):
    prep_trans = get_imagenet_transform(mode, normalize, **kwargs)

    if (isinstance(data_dir, list)
       and (data_dir[0].endswith('txt')
       or data_dir[0].startswith('s3')
       or data_dir[0].startswith('http'))
       or data_dir.endswith('txt')):
        if isinstance(data_dir, list) and data_dir[0].endswith('txt'):
            data_dir = data_dir[0]
        image_dataset = ImageS3
    else:
        image_dataset = ImageFolder
        data_dir = os.path.join(data_dir, 'ILSVRC/Data/CLS-LOC/test')

    # If testing the model, return the validation set from MNIST
    if mode != 'training':
        imagenet_data = image_dataset(root=data_dir, transform=prep_trans)
        test_queue = DataLoader(imagenet_data, batch_size=batch_size,
                                shuffle=False,
                                num_workers=workers,
                                pin_memory=True)
        return test_queue

    imagenet_data = image_dataset(root=data_dir, transform=prep_trans)

    train_size = int(len(imagenet_data) * 0.96)
    val_size = len(imagenet_data) - train_size

    train_ds, valid_ds = random_split(imagenet_data, (train_size, val_size))
    train_queue = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=workers,
                             pin_memory=True)
    valid_queue = DataLoader(valid_ds, batch_size=val_batch_size,
                             shuffle=False,
                             num_workers=workers,
                             pin_memory=True)

    return train_queue, valid_queue


def get_zarr_dataset(data_dir='.', task='autoencoder', batch_size=1,
                     val_batch_size=1,
                     workers=0,
                     data_mode='training',
                     normalize=True,
                     compressed_input=False,
                     shuffle_train=True,
                     shuffle_val=True,
                     shuffle_test=False,
                     train_dataset_size=-1,
                     val_dataset_size=-1,
                     test_dataset_size=-1,
                     gpu=False,
                     rotation=False,
                     elastic_deformation=False,
                     map_labels=False,
                     merge_labels=None,
                     mode='training',
                     add_noise=False,
                     **kwargs):
    """Creates a data queue using pytorch\'s DataLoader module to retrieve
    patches from images stored in zarr format.
    """

    (prep_trans,
     input_target_trans,
     target_trans) = get_zarr_transform(
        data_mode=data_mode,
        normalize=normalize,
        compressed_input=compressed_input,
        rotation=rotation,
        elastic_deformation=elastic_deformation,
        map_labels=map_labels,
        merge_labels=merge_labels,
        add_noise=add_noise)

    if task in ['autoencoder', 'segmentation'] and mode in ['training', 'test']:
        histo_dataset = LabeledZarrDataset
    else:
        histo_dataset = ZarrDataset

    # Modes can vary from testing, segmentation, compress, decompress, etc.
    # For this reason, only when it is properly training, two data queues are
    # returned, otherwise, only one queue is returned.
    if 'train' not in data_mode:
        zarr_data = histo_dataset(root=data_dir,
                                  dataset_size=test_dataset_size,
                                  data_mode='test',
                                  transform=prep_trans,
                                  intput_target_transform=input_target_trans,
                                  target_transform=target_trans,
                                  compressed_input=compressed_input,
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
                                    compressed_input=compressed_input,
                                    workers=workers,
                                    **kwargs)
    zarr_valid_data = histo_dataset(root=data_dir,
                                    dataset_size=val_dataset_size,
                                    data_mode='val',
                                    transform=prep_trans,
                                    input_target_transform=input_target_trans,
                                    target_transform=target_trans,
                                    compressed_input=compressed_input,
                                    workers=workers,
                                    **kwargs)

    # When training a network that expects to receive a complete image divided
    # into patches, it is better to use shuffle_trainin=False to preserve all
    # patches in the same batch.
    train_queue = DataLoader(zarr_train_data, batch_size=batch_size,
                             shuffle=shuffle_train,
                             num_workers=min(workers,
                                             len(zarr_train_data._filenames)),
                             pin_memory=gpu,
                             worker_init_fn=zarrdataset_worker_init)
    valid_queue = DataLoader(zarr_valid_data, batch_size=val_batch_size,
                             shuffle=shuffle_val,
                             num_workers=min(workers,
                                             len(zarr_valid_data._filenames)),
                             pin_memory=gpu,
                             worker_init_fn=zarrdataset_worker_init)

    return train_queue, valid_queue


def get_data(args):
    """ Retrieve a queue of data pairs (input, target) in mini-batches.
    The number of outputs can vary according to the use mode (training,
    testing, compression, decompression, etc).
    When used for training, two data queues are returned, one for training and
    the other for validation.
    Otherwise, only one data queue is returned.
    """
    args_dict = args if isinstance(args, dict) else args.__dict__

    # The arguments parser stores the data dir path as a list in case that more
    # than one path is given. However, when a directory is given, it should be
    # taken directly as the root directory of the dataset
    if (isinstance(args_dict['data_dir'], list)
      and len(args_dict['data_dir']) == 1):
        args_dict['data_dir'] = args_dict['data_dir'][0]

    if args_dict['dataset'] == 'MNIST':
        return get_MNIST(**args_dict)

    elif args_dict['dataset'] in ['ImageNet', 'ImageNet.S3']:
        return get_ImageNet(**args_dict)

    elif args_dict['dataset'] in ['Zarr', 'Histology']:
        return get_zarr_dataset(**args_dict)

    else:
        raise ValueError(f'The dataset \'{args_dict["dataset"]}\''
                         f' is not available for training.')
