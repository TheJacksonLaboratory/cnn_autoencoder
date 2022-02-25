from PIL import Image

import torch
import torchvision.transforms as transforms

from .datasets import get_zarr_dataset, get_MNIST, get_ImageNet


def get_data(args):
    args_dict = args if isinstance(args, dict) else args.__dict__

    if args.dataset == 'MNIST':
        return get_MNIST(**args_dict)

    elif args.dataset == 'ImageNet':
        return get_ImageNet(**args_dict)

    elif args.dataset == 'Zarr':
        return get_zarr_dataset(**args_dict)

    else:
        raise ValueError('The dataset \'%s\' is not available for training.' % args.dataset)


def open_image(filename, compression_level):
    img = Image.open(filename)

    height, width = img.size
    height_offset = (height % (2 ** compression_level))
    width_offset = (width % (2 ** compression_level))

    trans_comp = [transforms.Pad([0, 0, width_offset, height_offset]),
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
