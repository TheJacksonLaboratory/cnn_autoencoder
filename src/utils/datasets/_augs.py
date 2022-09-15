from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from elasticdeform import deform_grid
from scipy.ndimage import rotate


merge_funs = {'mean': np.mean, 'max': np.max, 'median':np.median}


def unfold_input(x, in_channels, patch_size, stride, padding):
    # Axes of x is expected to be in TCZYX order.
    x_type = x.dtype
    x_t = np.pad(x, padding, mode='reflect')
    x_t = torch.from_numpy(x_t)
    batch = F.unfold(x_t[0].float(), kernel_size=patch_size,
                     padding=0,
                     stride=stride)
    batch = batch.reshape(in_channels, patch_size, patch_size, -1).numpy()
    batch = np.moveaxis(batch, -1, 0).astype(x_type)
    return batch


def fold_input(x, in_channels, patch_size, np_H, np_W, H, W):
    x_type = x.dtype
    x_t = torch.from_numpy(x)
    x_t = x_t.permute(1, 2, 3, 0).reshape(in_channels * patch_size ** 2, -1)
    z = F.fold(x_t.float(), output_size=(np_H * patch_size, np_W * patch_size),
               kernel_size=patch_size,
               padding=0,
               stride=patch_size).int()
    z = z[..., :H, :W].reshape(1, -1, 1, H, W).numpy().astype(x_type)
    return z


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class RandomElasticDeformationInputTarget(object):
    def __init__(self, sigma=10):
        self._sigma = sigma

    def __call__(self, patch_target):
        patch, target = patch_target

        points = [3] * 2
        displacement = np.random.randn(2, *points) * self._sigma

        if not isinstance(patch, np.ndarray):
            patch = patch.numpy()

        if not isinstance(target, np.ndarray):
            target = target.numpy()

        patch = torch.from_numpy(deform_grid(patch, displacement, order=3,
                                             mode='reflect',
                                             axis=(1, 2))).float()
        target = torch.from_numpy(deform_grid(target, displacement, order=0,
                                              mode='reflect',
                                              axis=(1, 2))).float()

        return patch, target


class RandomRotationInputTarget(object):
    def __init__(self, degrees=90):
        self._degrees = degrees

    def __call__(self, patch_target):
        patch, target = patch_target

        angle = np.random.rand() * self._degrees

        if not isinstance(patch, np.ndarray):
            patch = patch.numpy()

        if not isinstance(target, np.ndarray):
            target = target.numpy()

        # rotate the input patch with bicubic interpolation, reflect the edges
        # to preserve the content in the image.
        patch = torch.from_numpy(rotate(patch.transpose(1, 2, 0), angle,
                                        order=3,
                                        reshape=False,
                                        mode='reflect').transpose(2, 0, 1)
                                 ).float()

        # rotate the target patch with nearest neighbor interpolation.
        target = torch.from_numpy(rotate(target.transpose(1, 2, 0), angle,
                                         order=0,
                                         reshape=False,
                                         mode='reflect').transpose(2, 0, 1)
                                  ).float()

        return patch, target


class MapLabels(object):
    """
    This mapping can handle the following case.
        0,0,0 -> 0
        1,0,0 -> 1
        1,1,0 -> 2
        1,1,1 -> 3
    """
    def __call__(self, target):
        mapped_labels = np.sum(target, axis=0)
        return mapped_labels


class MergeLabels(object):
    def __init__(self, merge_type):
        self._merge_fun = merge_funs[merge_type]

    def __call__(self, target):
        merge_axis = tuple(range(target.ndim-2, target.ndim))
        merged_labels = self._merge_fun(target, axis=merge_axis)
        return merged_labels


def get_zarr_transform(data_mode='testing', normalize=True,
                       compressed_input=False,
                       rotation=False,
                       elastic_deformation=False,
                       map_labels=None,
                       merge_labels=None,
                       add_noise=False,
                       **kwargs):
    """Define the transformations that are commonly applied to zarr-based
    datasets.

    When the input is compressed, it has a range of [0, 255], which is
    convenient to shift into a range of [-127.5, 127.5]. If the input is a
    color image (RGB) stored as zarr, it is normalized into the range [-1, 1].
    """
    prep_trans_list = [transforms.ToTensor(),
                       transforms.ConvertImageDtype(torch.float32)]

    if normalize:
        # The ToTensor transforms the input into the range [0, 1]. However, if
        # the input is compressed, it is required in the range [-127.5, 127.5].
        if compressed_input:
            prep_trans_list.append(transforms.Normalize(mean=0.5, std=1/255))
        else:
            prep_trans_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                        std=(0.5, 0.5, 0.5)))

    if add_noise:
        prep_trans_list.append(AddGaussianNoise(0., 0.1))

    prep_trans = transforms.Compose(prep_trans_list)

    input_target_trans_list = []
    if rotation:
        input_target_trans_list.append(RandomRotationInputTarget(degrees=30.))

    if elastic_deformation:
        input_target_trans_list.append(RandomElasticDeformationInputTarget(sigma=10))

    if len(input_target_trans_list) > 0:
        input_target_trans = transforms.Compose(input_target_trans_list)
    else:
        input_target_trans = None

    target_trans_list = []
    if map_labels:
        target_trans_list.append(MapLabels())

    if merge_labels is not None:
        target_trans_list.append(MergeLabels(merge_labels))

    if len(target_trans_list) > 0:
        target_trans = transforms.Compose(target_trans_list)
    else:
        target_trans = None

    return prep_trans, input_target_trans, target_trans


def get_imagenet_transform(mode='training', normalize=True, patch_size=128):
    prep_trans_list = [
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float32)
        ]

    if mode == 'training':
        prep_trans_list.append(AddGaussianNoise(0., 0.01))

    prep_trans_list.append(transforms.RandomCrop((patch_size, patch_size), pad_if_needed=True))

    if normalize:
        # prep_trans_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        prep_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))

    return transforms.Compose(prep_trans_list)


def get_mnist_transform(mode='training', normalize=True):
    prep_trans_list = [transforms.Pad(2),
                       transforms.PILToTensor(),
                       transforms.ConvertImageDtype(torch.float32)]

    if normalize:
        prep_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))

    return transforms.Compose(prep_trans_list)