import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def get_data(args):
    if args.dataset == 'MNIST':
        return get_MNIST(args)

    else:
        raise ValueError('The dataset \'%s\' is not available for training.' % args.dataset)


def get_MNIST(args):    
    prep_trans = transforms.Compose(        
        [transforms.Pad(2),
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float32),
         transforms.Normalize(mean=0.0, std=1.0)
        ]
    )

    mnist_data = MNIST(root=args.data_dir, train=True, download=False, transform=prep_trans)

    train_ds, valid_ds = random_split(mnist_data, (55000, 5000))
    train_queue = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=args.workers)
    valid_queue = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=args.workers)

    return train_queue, valid_queue