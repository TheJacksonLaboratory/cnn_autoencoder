import argparse
import os

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from PIL import Image


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


def get_imagenet_transform(mode='training', normalize=True):
    prep_trans_list = [
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float32)
        ]

    if mode == 'training':
        prep_trans_list.append(AddGaussianNoise(0., 0.01))
        prep_trans_list.append(transforms.RandomCrop((128, 128), pad_if_needed=True))

    if normalize:
        # prep_trans_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        prep_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))
        
    return transforms.Compose(prep_trans_list)


def get_ImageNet(data_dir='.', batch_size=1, val_batch_size=1, workers=0, mode='training', normalize=True, **kwargs):
    prep_trans = get_imagenet_transform(mode, normalize)

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

    return train_queue, valid_queue


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser('Test ImageNet dataloading (from S3 bucket)')
    
    parser.add_argument('-ds', '--dataset', nargs='+', dest='dataset_filenames', help='URL to the filenames in S3 storage')
    parser.add_argument('-m', '--mode', dest='mode', help='Mode of use of the dataset', choices=['training', 'validation', 'testing'], default='training')
    parser.add_argument('-bs', '--batch-size', type=int, dest='batch_size', help='Size of the batch retrieved', default=8)
    parser.add_argument('-nw', '--num-workers', type=int, dest='num_workers', help='Number of workers', default=0)

    args = parser.parse_args()

    transform = get_imagenet_transform(mode='training', normalize=True)
    
    trn_queue, val_queue = get_ImageNet(data_dir=args.dataset_filenames, batch_size=args.batch_size, val_batch_size=args.batch_size, workers=args.num_workers, mode=args.mode, normalize=True)

    print('Data set sizes: training %i, validation %i' % (len(trn_queue), len(val_queue)))
    
    for im, _ in trn_queue:
        print(im.size())
    
    for im, _ in val_queue:
        print(im.size())
        