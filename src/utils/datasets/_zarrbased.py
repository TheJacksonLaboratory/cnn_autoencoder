import math
import os
from functools import reduce

import math
import numpy as np
import zarr

from PIL import Image

from numcodecs import Blosc

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from elasticdeform import deform_grid, deform_random_grid
from scipy.ndimage import rotate


merge_funs = {'mean': np.mean, 'max': np.max, 'median':np.median}


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

        patch = torch.from_numpy(deform_grid(patch, displacement, order=3, mode='reflect', axis=(1, 2))).float()
        target = torch.from_numpy(deform_grid(target, displacement, order=0, mode='reflect', axis=(1, 2))).float()

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

        # rotate the input patch with bicubic interpolation, reflect the edges to preserve the content in the image
        patch = torch.from_numpy(rotate(patch.transpose(1, 2, 0), angle, order=3, reshape=False, mode='reflect').transpose(2, 0, 1)).float()

        # rotate the target patch with nearest neighbor interpolation
        target = torch.from_numpy(rotate(target.transpose(1, 2, 0), angle, order=0, reshape=False, mode='reflect').transpose(2, 0, 1)).float()

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


def parse_roi(filename):
    """ Parse the filename and ROIs from \'filename\'.
    The filename and ROIs must be separated by a semicolon (;). Any number of ROIs are accepted.
    ROIs are expected to be passed as (start_coords:axis_lengths), in the axis order of XYZCT.

    Example:
    test_file.zarr;(0, 10, 0, 0, 0):(10, 10, 1, 1, 1)
    Will parse a ROI from \'test_file\' from 0:10 in the first axis, 10:20 in the second axis, 0:1 in the third to fifth axes.

    Parameters:
    ----------
    filename : str, numpy.ndarray, zarr.Array, or zarr.Group
        Path to the image
    
    Returns
    -------
    fn : str
    rois : list of tuples
    """
    rois = []
    if isinstance(filename, (zarr.Array, np.ndarray)):
        fn = filename

    elif isinstance(filename, zarr.Group):
        fn = filename
        rois = filename.attrs['rois']

    elif isinstance(filename, str) and filename.lower().endswith('.zarr') and not ';' in filename:
        # The input is a zarr file, and the rois should be taken from it
        fn = filename
        z = zarr.open(filename, 'r')         
        rois = z.attrs.get('rois', [])
    
    elif isinstance(filename, str):
        broken_filename = filename.split(";")
        fn = broken_filename[0]
        rois_str = broken_filename[1:]
    
        for roi in rois_str:
            start_coords, axis_lengths = roi.split(':')
            start_coords = tuple([int(c.strip('\n\r ()')) for c in start_coords.split(',')])
            axis_lengths = tuple([int(l.strip('\n\r ()')) for l in axis_lengths.split(',')])

            rois.append((start_coords, axis_lengths))

    return fn, rois


def load_image(filename):
    """ Load the image at \'filename\' using the Image class from the PIL library and returns it as a numpy array.

    Parameters:
    ----------
    filename : str
        Path to the image
    
    Returns
    -------
    arr : numpy.array
    """
    im = Image.open(filename, mode="r").convert('RGB')
    arr = np.array(im)

    # Complete the number of dimensions to match the expected axis ordering (from OMERO)
    arr = arr.transpose(2, 0, 1)[np.newaxis, :, np.newaxis, ...]

    return arr
    

def compute_grid(index, imgs_shapes, imgs_sizes, patch_size):
    """ Compute the coordinate on a grid of indices corresponding to 'index'.
    The indices are in the form of [i, tl_x, tl_y], where 'i' is the file index.
    tl_x and tl_y are the top left coordinates of the patched image.
    To get a patch from any image, tl_y and tl_x must be multiplied by patch_size.
    
    Parameters:
    ----------
    index : int
        Index of the patched dataset Between 0 and 'total_patches'-1
    imgs_shapes : list of ints
        Shapes of each image in the dataset
    imgs_sizes : list of ints
        Number of patches that can be obtained from each image in the dataset
    patch_size : int
        The size of each squared patch
        
    Returns
    -------
    i : int
    tl_y : int
    tl_x : int
    """
    # This allows to generate virtually infinite data from bootstrapping the same data
    index %= imgs_sizes[-1]

    # Get the file index among the available file names
    i = list(filter(lambda l_h: l_h[1][0] <= index < l_h[1][1], enumerate(zip(imgs_sizes[:-1], imgs_sizes[1:]))))[0][0]
    index -= imgs_sizes[i]
    _, W = imgs_shapes[i]
    
    # Get the patch position in the file
    tl_y = index // int(math.ceil(W / patch_size))
    tl_x = index % int(math.ceil(W / patch_size))

    return i, tl_y, tl_x


def get_patch(z, tl_y, tl_x, patch_size, offset=0):
    """
    Gets a squared region from an array z (numpy or zarr).

    Parameters:
    ----------
    z : dask.array.core.Array, numpy.array or zarr.array
        A full array from where to take a patch
    tl_y : int
        Top left coordinate in the y-axis
    tl_x : int
        Top left coordinate in the x-axis
    patch_size : int
        Sice of the squared patch to extract from the input array 'z'
    offset : int
        Offset padding added to the retrieved patch.

    Returns
    -------
    patch : numpy.array
    """
    tl_y *= patch_size
    tl_x *= patch_size

    # TODO extract this information from the zarr metadata. For now, the color channel is considered to be in the second axis
    c = z.shape[1 if z.ndim > 3 else 0]
    H, W = z.shape[-2:]

    tl_y_offset = tl_y - offset
    tl_x_offset = tl_x - offset
    br_y_offset = tl_y + patch_size + offset
    br_x_offset = tl_x + patch_size + offset

    tl_y = max(tl_y_offset, 0)
    tl_x = max(tl_x_offset, 0)
    br_y = min(br_y_offset, H)
    br_x = min(br_x_offset, W)

    patch = z[..., tl_y:br_y, tl_x:br_x].squeeze()

    if c == 1:
        patch = patch[np.newaxis, ...]

    # In the case that the input patch contains more than three dimensions, pad the leading dimensions with (0, 0)
    leading_padding = [(0, 0)] * (patch.ndim - 2)

    # Pad the patch using the symmetric mode
    if offset > 0 or (patch.shape[-2] < patch_size or patch.shape[-1] < patch_size):
        pad_up = tl_y - tl_y_offset
        pad_down = br_y_offset - br_y
        pad_left = tl_x - tl_x_offset
        pad_right = br_x_offset - br_x

        valid_pad_up = min(pad_up, br_y - tl_y)
        valid_pad_down = min(pad_down, br_y - tl_y)
        valid_pad_left = min(pad_left, br_x - tl_x)
        valid_pad_right = min(pad_right, br_x - tl_x)

        patch  = np.pad(patch, 
            (*leading_padding, 
             (valid_pad_up, valid_pad_down),
             (valid_pad_left, valid_pad_right)),
            mode='symmetric', reflect_type='odd')

    return patch


def zarrdataset_worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset_obj = worker_info.dataset

    filenames_rois = list(map(parse_roi, dataset_obj._filenames))
    
    if len(filenames_rois) > 1 and len(filenames_rois) >= worker_info.num_workers:
        num_files_per_worker = int(math.ceil(len(filenames_rois) / worker_info.num_workers))
        curr_worker_filenames = dataset_obj._filenames[worker_id*num_files_per_worker:(worker_id+1)*num_files_per_worker]
        curr_worker_rois = None
    elif len(filenames_rois) == 1 and len(filenames_rois[0][1]) >= worker_info.num_workers:
        num_files_per_worker = int(math.ceil(len(filenames_rois[0][1]) / worker_info.num_workers))
        curr_worker_filenames = [filenames_rois[0][0]]
        curr_worker_rois = [filenames_rois[0][1][worker_id*num_files_per_worker:(worker_id+1)*num_files_per_worker]]
    else:
        raise ValueError('Missmatching number of workers and input files/ROIs')

    dataset_obj._z_list, dataset_obj._rois_list, dataset_obj._imgs_orginal_shapes = dataset_obj._preload_files(curr_worker_filenames, group='0', rois=curr_worker_rois)
    if hasattr(dataset_obj, '_lab_list'):
        dataset_obj._lab_list, _, _ = dataset_obj._preload_files(curr_worker_filenames, group='1', rois=curr_worker_rois)
    
    _, dataset_obj._max_H, dataset_obj._max_W, dataset_obj._org_channels, dataset_obj._imgs_sizes, dataset_obj._imgs_shapes = dataset_obj._compute_size(dataset_obj._z_list, dataset_obj._rois_list)
    dataset_obj._dataset_size //= worker_info.num_workers


class ZarrDataset(Dataset):
    """ A zarr-based dataset.
        The structure of the zarr file is considered fixed, and only the component '0/0' is used.
        Only two-dimensional (+color channels) data is supported by now.
    """
    def __init__(self, root, patch_size=128, dataset_size=-1, pyramid_level=0, mode='train', offset=0, transform=None, source_format='zarr', workers=0, data_axes='TCZYX', **kwargs):
        self._patch_size = patch_size
        self._dataset_size = dataset_size
        self._transform = transform
        self._offset = offset
        self._data_axes = data_axes

        self._pyramid_level = pyramid_level
        self._source_format = source_format.lower()
        if not self._source_format.startswith('.'):
            self._source_format = '.' + self._source_format

        self._mode = mode
        self._requires_split = False

        self._filenames = self._split_dataset(root)

        if workers == 0:
            self._z_list, self._rois_list, self._imgs_orginal_shapes = self._preload_files(self._filenames, group='0')
            dataset_size, self._max_H, self._max_W, self._org_channels, self._imgs_sizes, self._imgs_shapes = self._compute_size(self._z_list, self._rois_list)
        else:
            self._z_list = None
            self._rois_list = None
            self._imgs_orginal_shapes = None
            self._max_H = None
            self._max_W = None
            self._org_channels = None
            self._imgs_sizes = None
            self._imgs_shapes = None
        
        if self._dataset_size < 0:
            self._dataset_size = dataset_size

    def _get_filenames(self, source):
        if (isinstance(source, str) and self._source_format in source.lower()) or isinstance(source, (zarr.Group, zarr.Array, np.ndarray)):
            # If the input is a zarr group, zarr array, or numpy array return it as it is
            return [source]

        elif isinstance(source, list):
            # If the input is a list of any supported inputs, iterate each element
            # Check if an element in the list corresponds to the current data mode
            source_mode = list(filter(lambda fn: self._mode in fn, source))
            
            if len(source_mode) > 0:
                # Only if there is at least one element specific to the data mode, use it.
                # Otherwise, recurse the original source list
                source = source_mode
            
            return reduce(lambda l1, l2: l1 + l2, map(self._get_filenames, source), [])
        
        elif isinstance(source, str) and source.lower().endswith('txt'):
            self._requires_split = not self._mode.lower() in source.lower()

            # If the input is a text file with a list of url/paths or directories, recurse to get the filenames from the text file content
            with open(source, mode='r') as f:
                filenames = [l.strip('\n\r') for l in f.readlines()]

            return self._get_filenames(filenames)

        elif isinstance(source, str):
            self._requires_split = not self._mode.lower() in source.lower()

            # Otherwise, the input is a directory, create the filenames list from each element in that directory that meets the criteria
            return reduce(lambda l1, l2: l1 + l2, 
                    map(self._get_filenames, 
                        filter(lambda fn: self._source_format in fn.lower(), 
                            map(lambda fn: 
                                os.path.join(source, fn), sorted(os.listdir(source)))
                            )
                        )
                    )

        # If the source file/path does not meet the criteria, return an empty list
        return []
    
    def _split_dataset(self, root):
        """ Identify are the inputs being passed and split the data according to the mode.
        The datasets will be splitted into 70% training, 10% validation, and 20% testing.
        """
        # Get the set of filenames/arrays from the source input
        filenames = self._get_filenames(root)

        if self._requires_split:
            if self._mode == 'train':
                # Use 70% of the data for traning
                filenames = filenames[:int(0.7 * len(filenames))]
            elif self._mode == 'val':
                # Use 10% of the data for validation
                filenames = filenames[int(0.7 * len(filenames)):int(0.8 * len(filenames))]
            elif self._mode == 'test':
                # Use 20% of the data for testing
                filenames = filenames[int(0.8 * len(filenames)):]
        
        return filenames

    def _preload_files(self, filenames, group='0', rois=None):
        if rois is None:
            filenames_rois = list(map(parse_roi, filenames))
        else:
            filenames_rois = zip(filenames, rois)

        imgs_orginal_shapes = []
        z_list = []
        rois_list = []
        
        for id, (arr_src, rois) in enumerate(filenames_rois):
            org_height, org_width = None, None
            if isinstance(arr_src, zarr.Group) or (isinstance(arr_src, str) and '.zarr' in self._source_format):
                # If the passed object is a zarr group/file, open it and extract the level from the specified group
                if isinstance(arr_src, str):
                    arr_src = zarr.open(arr_src, mode='r')
                
                arr = arr_src['%s/%s' % (group, self._pyramid_level)]

                # The original height and width is stored as an attribute when compressing the image with the CAE
                org_height = arr_src.attrs.get('height', None)
                org_width = arr_src.attrs.get('width', None)

            elif isinstance(arr_src, str) and '.zarr' not in self._source_format:
                # If the input is a path to an image stored in a format supported by PIL, open it and use it as a numpy array
                arr = load_image(arr_src)
            else:
                # Otherwise, use directly the zarr array
                arr = arr_src
            
            # Store the original image's shapes
            if org_height is None:
                org_height = arr.shape[-2]
            if org_width is None:
                org_width = arr.shape[-1]

            imgs_orginal_shapes.append((org_height, org_width))

            z_list.append(arr)

            # List all ROIs in this image
            if len(rois) > 0:
                for (cx, cy, cz, cc, ct), (lx, ly, lz, lc, lt) in rois:
                    roi = [
                        slice(cx, cx+lx, 1),
                        slice(cy, cy+ly, 1),
                        slice(cz, cz+lz, 1),
                        slice(cc, cc+lc, 1),
                        slice(ct, ct+lt, 1)
                        ]
                    
                    # Because data could have been passed in a different axes ordering, slicing is reordered to match the input data axes ordering
                    roi = [roi['XYZCT'.index(a)] for a in self._data_axes]

                    # Take the ROI as the original size of the image
                    rois_list.append((id, tuple(roi)))

            else:
                roi = [slice(0, s, 1) for s in arr.shape]
                rois_list.append((id, tuple(roi)))
            
        return z_list, rois_list, imgs_orginal_shapes

    def _compute_size(self, z_list, rois_list):
        imgs_shapes = [((roi[-2].stop - roi[-2].start)//roi[-2].step, (roi[-1].stop - roi[-1].start)//roi[-1].step) for _, roi in rois_list]
        imgs_sizes = np.cumsum([0] + [int(np.ceil((H * W) / self._patch_size**2)) for H, W in imgs_shapes])
        
        # Get the upper bound of patches that can be obtained from all zarr files (images with smaller size will be padded)
        max_H, max_W = np.max(np.array(imgs_shapes), axis=0)
        
        max_H = self._patch_size * int(math.ceil(max_H / self._patch_size))
        max_W = self._patch_size * int(math.ceil(max_W / self._patch_size))
        
        # Compute the size of the dataset from the valid patches
        if z_list[0].ndim < 3:
            org_channels = 1
        elif z_list[0].ndim == 3:
            org_channels = z_list[0].shape[0]
        elif z_list[0].ndim > 3:
            org_channels = z_list[0].shape[1]

        # Return the dataset size and the information about the dataset
        return imgs_sizes[-1], max_H, max_W, org_channels, imgs_sizes, imgs_shapes

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        i, tl_y, tl_x = compute_grid(index, self._imgs_shapes, self._imgs_sizes, self._patch_size)
        id, roi = self._rois_list[i]
        patch = get_patch(self._z_list[id].get_orthogonal_selection(roi), tl_y, tl_x, self._patch_size, self._offset).squeeze()

        if self._transform is not None:
            patch = self._transform(patch.transpose(1, 2, 0))
        
        # Returns anything as label, to prevent an error during training
        return patch, [0]

    def get_channels(self):
        return self._org_channels

    def get_shape(self):
        return self._max_H, self._max_W

    def get_img_shape(self, i):
        return self._imgs_shapes[i]

    def get_img_original_shape(self, i):
        return self._imgs_orginal_shapes[i]


class LabeledZarrDataset(ZarrDataset):
    """ A labeled dataset based on the zarr dataset class.
        The densely labeled targets are extracted from group '1'.
    """
    def __init__(self, root, input_target_transform=None, target_transform=None, compression_level=0, compressed_input=False, **kwargs):
        super(LabeledZarrDataset, self).__init__(root, **kwargs)
        
        # Open the labels from group 1
        self._lab_list, _, _ = self._preload_files(self._filenames, group='1')

        self._compression_level = compression_level
        self._compressed_input = compressed_input

        # This is a transform that affects the geometry of the input, and then it has to be applied to the target as well
        self._input_target_transform = input_target_transform

        # This is a transform that only affects the target
        self._target_transform = target_transform
        
    def __getitem__(self, index):
        i, tl_y, tl_x = compute_grid(index, self._imgs_shapes, self._imgs_sizes, self._patch_size)
        id, roi = self._rois_list[i]        
        patch = get_patch(self._z_list[id].get_orthogonal_selection(roi), tl_y, tl_x, self._patch_size, self._offset).squeeze()

        if self._transform is not None:
            patch = self._transform(patch.transpose(1, 2, 0))
            
        patch_size = self._patch_size * ((2**self._compression_level) if self._compressed_input else 1)
        target = get_patch(self._lab_list[id].get_orthogonal_selection(roi), tl_y, tl_x, patch_size, 0).astype(np.float32)
        
        if self._input_target_transform:
            patch, target = self._input_target_transform((patch, target))

        if self._target_transform:
            target = self._target_transform(target)

        # Returns anything as label, to prevent an error during training
        return patch, target


def get_zarr_transform(mode='testing',
        normalize=True, compressed_input=False,
        rotation=False, elastic_deformation=False,
        map_labels=None, merge_labels=None, 
        add_noise=False, 
        **kwargs):
    """ Define the transformations that are commonly applied to zarr-based datasets.
    When the input is compressed, it has a range of [0, 255], which is convenient to shift into a range of [-127.5, 127.5].
    If the input is a color image (RGB) stored as zarr, it is normalized into the range [-1, 1].
    """
    prep_trans_list = [transforms.ToTensor(),
         transforms.ConvertImageDtype(torch.float32)
        ]

    if normalize:
        # The ToTensor transforms the input into the range [0, 1]. However, if the input is compressed, it is required in the range [-127.5, 127.5]
        if compressed_input:
            prep_trans_list.append(transforms.Normalize(mean=0.5, std=1/255))
        else:
            prep_trans_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

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


def get_zarr_dataset(
    data_dir='.', task='autoencoder', 
    batch_size=1, val_batch_size=1, workers=0,
    mode='training', normalize=True, compressed_input=False, 
    shuffle_training=True, shuffle_test=False, gpu=False,
    rotation=False, elastic_deformation=False,
    map_labels=False, merge_labels=None, **kwargs):
    """ Creates a data queue using pytorch\'s DataLoader module to retrieve patches from images stored in zarr format.
    The size of the data queue can be virtually infinite, for that reason, a conservative size has been defined using the following variables.
    1. TRAIN_DATASIZE: 1200000 for autoencoder models, and all available patches for segmenetation models
    2. VALID_DATASIZE: 50000 for autoencoder models, and all available patches for segmenetation models
    3. TEST_DATASIZE: 200000 for autoencoder models, and all available patches for segmenetation models
    """

    if task == 'autoencoder':
        prep_trans, input_target_trans, target_trans = get_zarr_transform(mode=mode, normalize=normalize, compressed_input=compressed_input)
        histo_dataset = ZarrDataset
        TRAIN_DATASIZE = 1200000
        VALID_DATASIZE = 50000
        
    elif task == 'segmentation':        
        prep_trans, input_target_trans, target_trans = get_zarr_transform(mode=mode, normalize=normalize, compressed_input=compressed_input, rotation=rotation, elastic_deformation=elastic_deformation)
        if 'train' in mode:
            histo_dataset = LabeledZarrDataset
        else:
            histo_dataset = ZarrDataset
        TRAIN_DATASIZE = -1
        VALID_DATASIZE = -1
        
    elif task == 'classification':
        prep_trans, input_target_trans, target_trans = get_zarr_transform(mode=mode, normalize=normalize, compressed_input=compressed_input, rotation=rotation, elastic_deformation=elastic_deformation, map_labels=map_labels, merge_labels=merge_labels)
        if 'train' in mode:
            histo_dataset = LabeledZarrDataset
        else:
            histo_dataset = ZarrDataset
        TRAIN_DATASIZE = 4000
        VALID_DATASIZE = 400
    
    # Modes can vary from testing, segmentation, compress, decompress, etc. For this reason, only when it is properly training, two data queues are returned, otherwise, only one queue is returned.
    if 'train' not in mode:
        hist_data = histo_dataset(root=data_dir, mode='test', transform=prep_trans, intput_target_transform=input_target_trans, target_trans=target_trans, compressed_input=compressed_input, **kwargs)
        test_queue = DataLoader(hist_data, batch_size=batch_size, shuffle=shuffle_test, num_workers=min(workers, len(hist_data._filenames)), pin_memory=gpu, worker_init_fn=zarrdataset_worker_init)
        return test_queue
    
    hist_train_data = histo_dataset(root=data_dir, dataset_size=TRAIN_DATASIZE, mode='train', transform=prep_trans, input_target_transform=input_target_trans, target_transform=target_trans, compressed_input=compressed_input, **kwargs)
    hist_valid_data = histo_dataset(root=data_dir, dataset_size=VALID_DATASIZE, mode='val', transform=prep_trans, input_target_transform=input_target_trans, target_transform=target_trans, compressed_input=compressed_input, **kwargs)

    # When training a network that expects to receive a complete image divided into patches, it is better to use shuffle_trainin=False to preserve all patches in the same batch.
    train_queue = DataLoader(hist_train_data, batch_size=batch_size, shuffle=shuffle_training, num_workers=min(workers, len(hist_train_data._filenames)), pin_memory=gpu, worker_init_fn=zarrdataset_worker_init)
    valid_queue = DataLoader(hist_valid_data, batch_size=val_batch_size, shuffle=False, num_workers=min(workers, len(hist_valid_data._filenames)), pin_memory=gpu, worker_init_fn=zarrdataset_worker_init)

    return train_queue, valid_queue


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser('Test zarr-based datasets generation and loading with a pytorch\'s DataLoader')

    parser.add_argument('-rs', '--seed', type=int, dest='seed', help='Random seed for the random number generator', default=-1)
    parser.add_argument('-dd', '--dir', dest='data_dir', nargs='+', help='Root directory where the zarr files are stored')
    parser.add_argument('-da', '--data-axes', type=str, dest='data_axes', help='Order of the axes in which the data is stored. For 5 channels: XYZCT', default='XYZCT')
    parser.add_argument('-nw', '--workers', type=int, dest='workers', help='Number of workers', default=0)
    parser.add_argument('-bs', '--batch-size', type=int, dest='batch_size', help='Batch size', default=8)
    parser.add_argument('-ps', '--patch-size', type=int, dest='patch_size', help='Size of the patch -> patch_size x patch_size', default=128)
    parser.add_argument('-shr', '--shuffle-trn', action='store_true', dest='shuffle_training', help='Shuffle training data?')
    parser.add_argument('-sht', '--shuffle-tst', action='store_true', dest='shuffle_test', help='Shuffle test data?')
    parser.add_argument('-t', '--task', dest='task', help='Task for what the data is used', choices=['autoencoder', 'segmentation', 'classification'], default='autoencoder')
    parser.add_argument('-ts', '--test-size', type=int, dest='test_size', help='Number of samples extracted from the test dataset', default=-1)
    parser.add_argument('-m', '--mode', dest='mode', help='The network use mode', choices=['train', 'val', 'test', 'all'], default='train')
    parser.add_argument('-o', '--offset', type=int, dest='offset', help='Offset added to the patches', default=0)
    parser.add_argument('-cl', '--comp-level', type=int, dest='compression_level', help='Compression level of the input', default=0)
    parser.add_argument('-if', '--src-format', type=str, dest='source_format', help='Format of the source files to compress', default='zarr')
    parser.add_argument('-mt', '--map-target', action='store_true', dest='map_labels', help='Map the target from vector form to a category label')
    parser.add_argument('-mtf', '--merge-target', type=str, dest='merge_labels', help='Merge a 2D labels map into a single label per patch', choices=list(merge_funs.keys()), default=None)

    args = parser.parse_args()

    if args.seed < 0:
        args.seed = np.random.randint(1, 100000)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)
    random.seed(args.seed + 2)

    args.compressed_input = args.compression_level > 0

    if args.mode in ['train', 'val']:
        dl, _ = get_zarr_dataset(**args.__dict__)
    else:
        dl = get_zarr_dataset(**args.__dict__)

    print('Batched dataset size:', len(dl))
    q = tqdm(total = len(dl))
    for i, (x, t) in enumerate(dl):
        if isinstance(t, list):
            q.set_description('Batch {} of size: {}, target: {}'.format(i, x.size(), len(t)))
        else:
            q.set_description('Batch {} of size: {}, target: {}'.format(i, x.size(), t.size()))
        q.update()

    q.close()