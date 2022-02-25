import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def get_mnist_transform(mode='training', normalize=True):
    prep_trans_list = [transforms.Pad(2),
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float32)
        ]
    
    if normalize:
        # prep_trans_list.append(transforms.Normalize(mean=0.0, std=1.0))
        prep_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))
            
    return transforms.Compose(prep_trans_list)


def get_MNIST(data_dir, batch_size=1, val_batch_size=1, workers=0, mode='training', normalize=True, **kwargs):
    prep_trans = get_mnist_transform(mode, normalize)

    # If testing the model, return the test set from MNIST
    if mode != 'training':
        mnist_data = MNIST(root=data_dir, train=False, download=False, transform=prep_trans)
        test_queue = DataLoader(mnist_data, batch_size=batch_size, shuffle=False, num_workers=workers)
        return test_queue

    mnist_data = MNIST(root=data_dir, train=True, download=False, transform=prep_trans)

    train_ds, valid_ds = random_split(mnist_data, (55000, 5000))
    train_queue = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_queue = DataLoader(valid_ds, batch_size=val_batch_size, shuffle=False, num_workers=workers)

    return train_queue, valid_queue
