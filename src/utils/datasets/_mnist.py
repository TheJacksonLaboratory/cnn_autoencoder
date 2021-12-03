import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


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
