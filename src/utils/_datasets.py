import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, ImageFolder

from PIL import Image


def get_data(args, normalize=True):
    if args.dataset == 'MNIST':
        return get_MNIST(args, normalize)

    elif args.dataset == 'ImageNet':
        return get_ImageNet(args, normalize)

    else:
        raise ValueError('The dataset \'%s\' is not available for training.' % args.dataset)


def get_MNIST(args, normalize=True):
    prep_trans_list = [transforms.Pad(2),
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float32)
        ]
    
    if normalize:
        # prep_trans_list.append(transforms.Normalize(mean=0.0, std=1.0))
        prep_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))
            
    prep_trans = transforms.Compose(prep_trans_list)

    # If testing the model, return the test set from MNIST
    if args.mode != 'training':
        mnist_data = MNIST(root=args.data_dir, train=False, download=args.download_data, transform=prep_trans)
        test_queue = DataLoader(mnist_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        return test_queue

    mnist_data = MNIST(root=args.data_dir, train=True, download=args.download_data, transform=prep_trans)

    train_ds, valid_ds = random_split(mnist_data, (55000, 5000))
    train_queue = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_queue = DataLoader(valid_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers)

    return train_queue, valid_queue


def get_ImageNet(args, normalize=True):
    prep_trans_list = [
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float32)
        ]

    if args.mode == 'training':
        prep_trans_list.append(transforms.RandomCrop((128, 128), pad_if_needed=True))

    if normalize:
        # prep_trans_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        prep_trans_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
    prep_trans = transforms.Compose(prep_trans_list)

    # If testing the model, return the validation set from MNIST
    if args.mode != 'training':
        imagenet_data = ImageFolder(root=os.path.join(args.data_dir, 'ILSVRC/Data/CLS-LOC/test'), transform=prep_trans)
        test_queue = DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        return test_queue

    imagenet_data = ImageFolder(root=os.path.join(args.data_dir, 'ILSVRC/Data/CLS-LOC/train'), transform=prep_trans)
    
    train_size = int(len(imagenet_data) * 0.96)
    val_size = len(imagenet_data) - train_size
    
    train_ds, valid_ds = random_split(imagenet_data, (train_size, val_size))
    train_queue = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_queue = DataLoader(valid_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers)

    return train_queue, valid_queue


def open_image(filename, compression_level):
    img = Image.open(filename)

    height, width = img.size
    height_offset = (height % (2 ** compression_level))
    width_offset = (width % (2 ** compression_level))

    trans_comp = [transforms.Pad([height_offset // 2, height_offset // 2 + height_offset % 2, width_offset // 2, width_offset // 2 + width_offset % 2]),
                  transforms.PILToTensor(),
                  transforms.ConvertImageDtype(torch.float32)
                ]

    channels = len(img.getbands())

    if channels == 3:
        # The ImageNet original normalization parameters
        # trans_comp.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        # The paper normalization parameters
        trans_comp.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    elif channels == 1:
        trans_comp.append(transforms.Normalize(mean=0.5, std=0.5))
    else:
        raise ValueError('The image has an unsoported number of channels.')
    
    prep_trans = transforms.Compose(trans_comp)

    img = prep_trans(img).unsqueeze(dim=0)

    return img


def save_image(filename, img):
    img = img.clip(0.0, 1.0) * 255.0

    post_trans = transforms.ToPILImage()
    img = post_trans(img.squeeze().to(torch.uint8))
    img.save(filename)
    return img


def open_compressed(filename):
    img = torch.load(filename).to(torch.float32)
    return img


def save_compressed(filename, img):
    torch.save(img, filename)
