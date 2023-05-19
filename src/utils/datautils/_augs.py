from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from elasticdeform import deform_grid
from scipy.ndimage import rotate, label, distance_transform_edt
from zarrdataset import SelectAxes, ZarrToArray

merge_funs = {'mean': np.mean, 'max': np.max, 'median':np.median}


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noisy_tensor = tensor + torch.randn(tensor.size()) * self.std \
                       + self.mean
        noisy_tensor.clip_(0, 1)
        return noisy_tensor


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
                                        order=4,
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


class RandomElasticDeformationInput(object):
    def __init__(self, sigma=10):
        self._sigma = sigma

    def __call__(self, patch):
        points = [3] * 2
        displacement = np.random.randn(2, *points) * self._sigma

        if not isinstance(patch, np.ndarray):
            patch = patch.numpy()

        patch = torch.from_numpy(deform_grid(patch, displacement, order=3,
                                             mode='reflect',
                                             axis=(1, 2))).float()
        return patch


class WeightsDistances(object):
    """Computes the weight associated to each pixel on the label image to the
    clossest object.
    """
    def __init__(self, class_weights, sigma=5, w_0=10):
        self.class_weights = class_weights
        self.sigma_2 = 2 * sigma ** 2
        self.w_0 = w_0
        self.SE = np.ones((3, 3))

    def __call__(self, target):
        w_x = np.take(self.class_weights, target.astype(np.int32))
        w_x = w_x.astype(dtype=np.float32)

        num_objects = target.sum()
        if num_objects > 0:
            target_labels, num_objects = label(target[0], structure=self.SE)
            dist = []
            for l in range(1, num_objects + 1):
                target_remaining = np.ones_like(target[0])
                target_remaining[target_labels == l] = 0
                d2_rem = distance_transform_edt(target_remaining)
                dist.append(d2_rem.astype(np.float32))

            dist = np.stack(dist)
            dist = np.sort(dist, axis=0)

            if num_objects > 1:
                w_1 = np.exp(-(dist[0] + dist[1]) ** 2 / self.sigma_2)
            else:
                w_1 = np.exp(-dist[0] ** 2 / self.sigma_2)

            w_x = w_x + self.w_0 * w_1

        return np.concatenate((w_x, target), axis=0)


class MapLabels(object):
    """ This mapping can handle the following cases.
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


class ExpandTensor(object):
    def __init__(self, ndim=3):
        self._ndim = ndim

    def __call__(self, image):
        if self._ndim == 0:
            return image.view(-1).squeeze()

        if image.ndim < self._ndim:
            sizes = tuple([1] * (self._ndim - image.ndim) + [-1] * image.ndim)
            image = image.expand(*sizes)

        elif image.ndim > self._ndim:
            dim_slice = slice(image.ndim - self._ndim, image.ndim, None)
            sizes = tuple([-1] + list(image.shape[dim_slice]))
            image = image.view(*sizes)

        return image


class ConvertTensorDtype(object):
    def __init__(self, dtype):
        if dtype is torch.float32:
            self.__tofun = lambda image: image.float()
        elif dtype is torch.int64:
            self.__tofun = lambda image: image.long()
        else:
            raise ValueError(
                "Convertion to data type {} not supported".format(dtype))

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        return self.__tofun(image)


def zarr_input_transform(patch_size, data_axes, add_noise, data_mode,
                         normalize,
                         rotation,
                         label_density,
                         elastic_deformation):

    prep_trans_list = [SelectAxes(source_axes=data_axes,
                                  axes_selection={"T": 0, "Z": 0},
                                  target_axes="YXC"),
                       ZarrToArray(dtype=None),
                       transforms.ToTensor(),
                       transforms.ConvertImageDtype(torch.float32)]

    if add_noise:
        prep_trans_list.append(AddGaussianNoise(0., 0.001))

    if 'train' in data_mode:
        prep_trans_list.append(transforms.RandomCrop((patch_size, patch_size),
                                                     pad_if_needed=True))
    elif 'test' in data_mode:
        prep_trans_list.append(transforms.CenterCrop((patch_size, patch_size)))

    # The ToTensor transforms the input into the range [0, 1]. However, if
    # the input is compressed, it is required in the range [-127.5, 127.5].
    if normalize:
        prep_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))

    input_target_trans_list = []
    if rotation:
        if label_density == 2:
            input_target_trans_list.append(
                RandomRotationInputTarget(degrees=30.))
        else:
            prep_trans_list.append(
                transforms.RandomRotation(
                    degrees=30.,
                    interpolation=transforms.InterpolationMode.BILINEAR))

    if elastic_deformation:
        if label_density == 2:
            input_target_trans_list.append(
                RandomElasticDeformationInputTarget(sigma=10))
        else:
            prep_trans_list.append(
                RandomElasticDeformationInput(sigma=10))

    prep_trans = transforms.Compose(prep_trans_list)

    if len(input_target_trans_list) > 0:
        input_target_trans = transforms.Compose(input_target_trans_list)
    else:
        input_target_trans = None

    return prep_trans, input_target_trans

def zarr_target_transform(labels_data_axes,
                          target_data_type,
                          label_density,
                          map_labels,
                          merge_labels,
                          class_weights,
                          weights_map_sigma,
                          weights_map_w):
    
    target_trans_list = []
    if map_labels:
        target_trans_list.append(MapLabels())

    if merge_labels is not None:
        target_trans_list.append(MergeLabels(merge_labels))

    if (class_weights is not None
      and weights_map_sigma is not None
      and weights_map_w is not None):
        target_trans_list.append(WeightsDistances(class_weights=class_weights,
                                                  sigma=weights_map_sigma,
                                                  w_0=weights_map_w))

    target_trans_list.append(SelectAxes(source_axes=labels_data_axes,
                                        axes_selection={"T": 0, "Z": 0},
                                        target_axes="CYX"))

    target_trans_list.append(ZarrToArray(dtype=None))

    if target_data_type is not None:
        target_trans_list.append(ConvertTensorDtype(target_data_type))

    if len(target_trans_list) > 0:
        target_trans = transforms.Compose(target_trans_list)
    else:
        target_trans = None

    return target_trans


def get_zarr_transform(data_mode='test', data_axes=None, labels_data_axes=None,
                       normalize=False,
                       rotation=False,
                       elastic_deformation=False,
                       target_data_type=None,
                       label_density=0,
                       map_labels=None,
                       merge_labels=None,
                       add_noise=False,
                       patch_size=128,
                       weights_map_sigma=None,
                       weights_map_w=None,
                       class_weights=None,
                       **kwargs):
    """Define the transformations that are commonly applied to zarr-based
    datasets.

    When the input is compressed, it has a range of [0, 255], which is
    convenient to shift into a range of [-127.5, 127.5]. If the input is a
    color image (RGB) stored as zarr, it is normalized into the range [-1, 1].
    """
    (prep_trans,
     input_target_trans) = zarr_input_transform(patch_size, data_axes,
                                                add_noise,
                                                data_mode,
                                                normalize,
                                                rotation,
                                                label_density,
                                                elastic_deformation)

    target_trans = zarr_target_transform(labels_data_axes, target_data_type,
                                         label_density,
                                         map_labels,
                                         merge_labels,
                                         class_weights,
                                         weights_map_sigma,
                                         weights_map_w)

    return prep_trans, input_target_trans, target_trans


def get_imagenet_transform(data_mode='training', normalize=False,
                           patch_size=128):
    prep_trans_list = [
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float32)
        ]

    if 'train' in data_mode:
        prep_trans_list.append(AddGaussianNoise(0., 0.01))
    
        prep_trans_list.append(transforms.RandomCrop((patch_size, patch_size),
                                                        pad_if_needed=True))
    elif 'test' in data_mode:
        prep_trans_list.append(transforms.CenterCrop((patch_size, patch_size)))

    if normalize:
        prep_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))

    return transforms.Compose(prep_trans_list)


def get_mnist_transform(data_mode='training', normalize=True):
    prep_trans_list = [transforms.Pad(2),
                       transforms.PILToTensor(),
                       transforms.ConvertImageDtype(torch.float32)]

    if normalize:
        prep_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))

    return transforms.Compose(prep_trans_list)


def get_cifar_transform(data_mode='training', normalize=True):
    prep_trans_list = [transforms.PILToTensor(),
                       transforms.ConvertImageDtype(torch.float32)]

    if normalize:
        prep_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))

    return transforms.Compose(prep_trans_list)
