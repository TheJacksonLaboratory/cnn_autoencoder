import argparse
import os

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from PIL import Image


try:
    import boto3
    from io import BytesIO

    ImageS3_Implemented = True

    class ImageS3(Dataset):
        def __init__(self, root, transform=None, endpoint=None, bucket_name=None):
            
            if isinstance(root, list):
                self._s3_urls = root
            
            elif root.endswith('.txt'):
                with open(root, 'r') as f:
                    self._s3_urls = [l.strip() for l in f.readlines()]
            else:
                raise ValueError('Root %s not supported for retrieving images from an s3 bucket' % root)

            # If endpoint is none, determine the end point from the file names
            if endpoint is None:
                endpoint = '/'.join(self._s3_urls[0].split('/')[:3])
                self._remove_endpoint = True

            else:
                self._remove_endpoint = False            
        
            if bucket_name is not None:
                self._bucket_name = bucket_name
            else:
                self._bucket_name = self._s3_urls[0].split('/')[3]
            
            # Access the bucket anonymously
            self._s3 = boto3.client('s3', aws_access_key_id='', aws_secret_access_key='', region_name='us-east-2', endpoint_url=endpoint)
            
            self._s3._request_signer.sign = (lambda *args, **kwargs: None)

            self._transform = transform

        def __getitem__(self, index):
            if self._remove_endpoint:
                fn = '/'.join(self._s3_urls[index].split('/')[4:])
            else:
                fn = self._s3_urls[index]
            
            im_bytes = self._s3.get_object(Bucket=self._bucket_name, Key=fn)['Body'].read()
            im_s3 = Image.open(BytesIO(im_bytes))

            # Copy and close the connection with the original image in the cloud bucket. Additionally, convert any grayscale image to RGB (replicate it to have three channels)
            im = im_s3.copy().convert('RGB')
            im_s3.close()

            if self._transform is not None:
                im = self._transform(im)
            
            return im, [0]

        def __len__(self):
            return len(self._s3_urls)

except ModuleNotFoundError:
    print('Loading ImageNet from S3 bucket requires boto3 to be installed; however, it was not found. ImageNet from S3 not supported in this session.')
    ImageS3_Implemented = False
    ImageS3 = None


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

    if ImageS3_Implemented and (isinstance(data_dir, list) and (data_dir[0].endswith('txt') or data_dir[0].startswith('s3') or data_dir[0].startswith('http'))) or data_dir.endswith('txt'):
        if isinstance(data_dir, list) and data_dir[0].endswith('txt'):
            data_dir = data_dir[0]
        image_dataset = ImageS3
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
        