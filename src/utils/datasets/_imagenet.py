import os

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_ImageNet(args, normalize=True):
    prep_trans_list = [
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float32)
        ]
    
        
    if args.mode == 'training':
        prep_trans_list.append(AddGaussianNoise(0., 0.1))
        prep_trans_list.append(transforms.RandomCrop((128, 128), pad_if_needed=True))

    if normalize:
        # prep_trans_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        prep_trans_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
    prep_trans = transforms.Compose(prep_trans_list)

    # If testing the model, return the validation set from MNIST
    if args.mode != 'training':
        imagenet_data = ImageFolder(root=os.path.join(args.data_dir, 'ILSVRC/Data/CLS-LOC/test'), transform=prep_trans)
        test_queue = DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        return test_queue

    imagenet_data = ImageFolder(root=os.path.join(args.data_dir, 'ILSVRC/Data/CLS-LOC/train'), transform=prep_trans)
    
    train_size = int(len(imagenet_data) * 0.96)
    val_size = len(imagenet_data) - train_size
    
    train_ds, valid_ds = random_split(imagenet_data, (train_size, val_size))
    train_queue = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_queue = DataLoader(valid_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    return train_queue, valid_queue