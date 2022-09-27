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
