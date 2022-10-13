import math
import os
from functools import reduce

import numpy as np
import zarr

from PIL import Image

import torch
from torch.utils.data import Dataset


def image_to_zarr(arr_src, patch_size, source_format, data_group,
                  compressed_input=False):
    if (isinstance(arr_src, zarr.Group) or (isinstance(arr_src, str)
       and '.zarr' in source_format)):
        # If the passed object is a zarr group/file, open it and
        # extract the level from the specified group.
        if isinstance(arr_src, str):
            arr_src = zarr.open(arr_src, mode='r')

        arr = arr_src[data_group]
        comp_group = data_group.split('/')[0]

        if (compressed_input
           and comp_group in arr_src.keys()
           and 'compression_metadata' in arr_src[comp_group].attrs.keys()):
            comp_metadata = arr_src['compressed'].attrs['compression_metadata']
            compressed_channels = comp_metadata['compressed_channels']
            org_H = comp_metadata['height']
            org_W = comp_metadata['width']
            compression_level = comp_metadata['compression_level']
            arr_shape = (1, compressed_channels, 1, org_H, org_W)

        else:
            arr_shape = arr.shape
            compression_level = 0

    elif (isinstance(arr_src, str) and '.zarr' not in source_format):
        # If the input is a path to an image stored in a format
        # supported by PIL, open it and use it as a numpy array.
        arr = load_image(arr_src)
        arr = zarr.array(arr, chunks=(1, arr.shape[1], 1, patch_size,
                         patch_size))
        arr_shape = arr.shape
        compression_level = 0

    else:
        # Otherwise, use directly the zarr array
        arr = arr_src
        arr_shape = arr.shape
        compression_level = 0

    return arr, arr_shape, compression_level


def compute_num_patches(size, patch_size, padding, stride):
    """Compute the number of valid patches that can be extracted from the
    source image in a certain axis.

    Parameters:
    ----------
    size : int
        Size of the array in the given axis.
    patch_size : int
        The size of the patch extracted.
    padding : int
        Total padding added to the given axis of the source array.
    stride : int
        Stride between patches extracted in the given axis.

    Returns
    -------
    n_patches : int
        The number of complete patches that can be extracted from a certain
        axis with the given parameters.
    """
    n_patches = (size + padding - patch_size + stride) // stride
    n_patches = max(n_patches, 1)
    return n_patches


def parse_roi(filename):
    """Parse the filename and ROIs from `filename`.

    The filename and ROIs must be separated by a semicolon (;).
    Any number of ROIs are accepted. ROIs are expected to be passed as
    (start_coords:axis_lengths), in the axis order of XYZCT.

    Notes:
    ------
    An example of a ROI structure is the following.

    test_file.zarr;(0, 10, 0, 0, 0):(10, 10, 1, 1, 1)
    Will parse a ROI from \'test_file\' from 0:10 in the first axis, 10:20 in
    the second axis, 0:1 in the third to fifth axes.

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

    elif (isinstance(filename, str)
          and filename.lower().endswith('.zarr') and ';' not in filename):
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

            start_coords = tuple([int(c.strip('\n\r ()'))
                                  for c in start_coords.split(',')])

            axis_lengths = tuple([int(l.strip('\n\r ()'))
                                  for l in axis_lengths.split(',')])

            rois.append((start_coords, axis_lengths))

    return fn, rois


def get_filenames(source, source_format, data_mode):
    if (isinstance(source, str) and source_format in source.lower()
       or isinstance(source, (zarr.Group, zarr.Array, np.ndarray))):
        # If the input is a zarr group, zarr array, or numpy array return it as
        # it is.
        return [source]

    elif isinstance(source, list):
        # If the input is a list of any supported inputs, iterate each element
        # Check if an element in the list corresponds to the current data mode
        source_mode = list(filter(lambda fn: data_mode in fn, source))

        if len(source_mode) > 0:
            # Only if there is at least one element specific to the data mode,
            # use it. Otherwise, recurse the original source list.
            source = source_mode

        return reduce(lambda l1, l2:
                      l1 + l2,
                      map(get_filenames,
                          source,
                          [source_format] * len(source),
                          [data_mode] * len(source)),
                      [])

    elif isinstance(source, str) and source.lower().endswith('txt'):
        # If the input is a text file with a list of url/paths or directories,
        # recurse to get the filenames from the text file content.
        with open(source, mode='r') as f:
            filenames = [line.strip('\n\r ') for line in f.readlines()]

        return get_filenames(filenames, source_format, data_mode)

    elif isinstance(source, str):
        # Otherwise, the input is a directory, create the filenames list from
        # each element in that directory that meets the criteria.
        source = [os.path.join(source, fn)
                  for fn in sorted(os.listdir(source))
                  if source_format in fn.lower()]

        return reduce(lambda l1, l2:
                      l1 + l2,
                      map(get_filenames,
                          source,
                          [source_format] * len(source),
                          [data_mode] * len(source)))

    # If the source file/path does not meet the criteria, return an empty list
    return []


def load_image(filename):
    """Load the image at `filename` using the Image class from the PIL library
    and returns it as a numpy array.

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

    # Complete the number of dimensions to match the expected axis ordering
    # (from OMERO).
    arr = arr.transpose(2, 0, 1)[np.newaxis, :, np.newaxis, ...]

    return arr


def compute_grid(index, imgs_shapes, imgs_sizes, patch_size, padding, stride,
                 by_rows=True):
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
    padding : tuple of ints
        Padding added to the source image prior to retrieve the patch.
    stride : tuple of ints
        The spacing in each axis (x, y) between each patch retrieved.
    by_rows : bool
        Whether the patches are extracted by rows (left to right) of by columns
        (top to bottom).

    Returns
    -------
    i : int
    tl_y : int
    tl_x : int
    """
    # This allows to generate virtually infinite data from bootstrapping the same data
    index %= imgs_sizes[-1]

    # Get the file index among the available file names
    i = list(filter(lambda l_h:
                    l_h[1][0] <= index < l_h[1][1],
                    enumerate(zip(imgs_sizes[:-1], imgs_sizes[1:]))))[0][0]
    index -= imgs_sizes[i]
    H, W = imgs_shapes[i]

    # Get the patch position in the file
    if by_rows:
        n_patches_per_row = compute_num_patches(W, patch_size,
                                                padding[0] + padding[1],
                                                stride[0])
        tl_y = index // n_patches_per_row
        tl_x = index % n_patches_per_row
    else:
        n_patches_per_col = compute_num_patches(H, patch_size,
                                                padding[2] + padding[3],
                                                stride[1])
        tl_y = index % n_patches_per_col
        tl_x = index // n_patches_per_col
    return i, tl_y, tl_x


def get_patch(z, shape, tl_y, tl_x, patch_size, padding, stride,
              compression_level=0):
    """Get a squared region from an array z (numpy or zarr).

    Parameters:
    ----------
    z : dask.array.core.Array, numpy.array or zarr.array
        A full array from where to take a patch
    shape : tuple of ints
        Shape of the original array. For compressed representation of images,
        the shape of the uncompressed image.
    tl_y : int
        Top left coordinate in the y-axis
    tl_x : int
        Top left coordinate in the x-axis
    patch_size : int
        Sice of the squared patch to extract from the input array `z`
    padding : tuple of ints
        Padding added to the source image prior to retrieve the patch.
    stride : tuple of ints
        The spacing in each axis (x, y) between each patch retrieved.
    compression_level : int
        In case that the input is a comrpessed representation, pass the level
        of compression.

    Returns
    -------
    patch : numpy.array
    """
    tl_y *= stride[0]
    tl_x *= stride[1]

    # TODO extract this information from the zarr metadata. For now, the color
    # channel is considered to be in the second axis.
    c = z.shape[1 if z.ndim > 3 else 0]
    H, W = shape[-2:]

    tl_x_padding = tl_x - padding[0]
    br_x_padding = tl_x + patch_size + padding[1]
    tl_y_padding = tl_y - padding[2]
    br_y_padding = tl_y + patch_size + padding[3]

    tl_y = max(tl_y_padding, 0) // 2 ** compression_level
    tl_x = max(tl_x_padding, 0) // 2 ** compression_level
    br_y = min(br_y_padding, H) // 2 ** compression_level
    br_x = min(br_x_padding, W) // 2 ** compression_level
    tl_x_padding //= 2 ** compression_level
    br_x_padding //= 2 ** compression_level
    tl_y_padding //= 2 ** compression_level
    br_y_padding //= 2 ** compression_level

    patch = z[..., tl_y:br_y, tl_x:br_x].squeeze()

    if c == 1:
        patch = patch[np.newaxis, ...]

    # In the case that the input patch contains more than three dimensions, pad
    # the leading dimensions with (0, 0).
    leading_padding = [(0, 0)] * (patch.ndim - 2)

    # Pad the patch using the symmetric mode
    if (patch.shape[-2] < patch_size // 2 ** compression_level
       or patch.shape[-1] < patch_size // 2 ** compression_level):
        pad_up = tl_y - tl_y_padding
        pad_down = br_y_padding - br_y
        pad_left = tl_x - tl_x_padding
        pad_right = br_x_padding - br_x

        patch = np.pad(patch,
                       (*leading_padding,
                        (pad_up, pad_down),
                        (pad_left, pad_right)),
                       mode='symmetric',
                       reflect_type='even')

    return patch


def zarrdataset_worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset_obj = worker_info.dataset

    filenames_rois = list(map(parse_roi, dataset_obj._filenames))

    if (len(filenames_rois) > 1
       and len(filenames_rois) >= worker_info.num_workers):
        nf_worker = int(math.ceil(len(filenames_rois)
                                  / worker_info.num_workers))
        curr_worker_filenames = \
            dataset_obj._filenames[
                worker_id*nf_worker:(worker_id + 1)*nf_worker]
        curr_worker_rois = None
    elif (len(filenames_rois) == 1
          and len(filenames_rois[0][1]) >= worker_info.num_workers):
        nf_worker = int(math.ceil(len(filenames_rois[0][1])
                                  / worker_info.num_workers))
        curr_worker_filenames = [filenames_rois[0][0]]
        curr_worker_rois = [
            filenames_rois[0][1][worker_id*nf_worker:(worker_id+1)*nf_worker]]
    else:
        raise ValueError('Missmatching number of workers and input files/ROIs')

    (dataset_obj.z_list,
     dataset_obj._rois_list,
     dataset_obj._compression_level) = \
        dataset_obj._preload_files(
            curr_worker_filenames,
            data_group=dataset_obj._data_group,
            data_axes=dataset_obj._data_axes,
            compressed_input=dataset_obj._compressed_input,
            rois=curr_worker_rois)

    if hasattr(dataset_obj, '_lab_list'):
        (dataset_obj._lab_list,
         dataset_obj._lab_rois_list,
         _) = \
            dataset_obj._preload_files(
                curr_worker_filenames,
                data_group=dataset_obj._labels_group,
                data_axes=dataset_obj._labels_data_axes,
                compressed_input=False,
                rois=curr_worker_rois)

    (dataset_obj._dataset_size,
     dataset_obj._max_H,
     dataset_obj._max_W,
     dataset_obj._org_channels,
     dataset_obj._imgs_sizes,
     dataset_obj.imgs_shapes) = \
        dataset_obj._compute_size(dataset_obj.z_list, dataset_obj._rois_list)


class ZarrDataset(Dataset):
    """A zarr-based dataset.

        The structure of the zarr file is considered as it follows the OME-NGFF
        standard and the data from 'data_group' is hte one accessed and used.
        Only two-dimensional (+color channels) data is supported by now. This
        is because 2D image operations are used for pre-/post-processing.
    """
    def __init__(self, root,
                 patch_size=128,
                 dataset_size=-1,
                 data_mode='train',
                 padding=None,
                 stride=None,
                 transform=None,
                 source_format='zarr',
                 data_axes='TCZYX',
                 data_group='0/0',
                 compressed_input=False,
                 split_train=0.7,
                 split_val=0.1,
                 **kwargs):

        if padding is None:
            padding = (0, 0, 0, 0)
        elif isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        if stride is None:
            stride = (patch_size, patch_size)
        elif isinstance(stride, int):
            stride = (stride, stride)

        self._dataset_size = dataset_size
        self._split_train = split_train
        self._split_val = split_val
        self._transform = transform

        self._patch_size = patch_size
        self._padding = padding
        self._stride = stride

        self._data_axes = data_axes
        self._data_group = data_group

        self._compression_level = None
        self._compressed_input = compressed_input

        self._source_format = source_format.lower()
        if not self._source_format.startswith('.'):
            self._source_format = '.' + self._source_format

        self._data_mode = data_mode
        self._requires_split = False

        self._filenames = self._split_dataset(root)

        (self.z_list,
         self._rois_list,
         self._compression_level) = \
            self._preload_files(self._filenames,
                                data_group=self._data_group,
                                data_axes=self._data_axes,
                                compressed_input=self._compressed_input)

        (self._dataset_size,
         self._max_H,
         self._max_W,
         self._org_channels,
         self._imgs_sizes,
         self.imgs_shapes) = self._compute_size(self.z_list, self._rois_list)

    def _get_filenames(self, source):
        if (isinstance(source, str)
           and self._source_format not in source.lower()):
            if (self._data_mode.lower() != 'all'
               and not self._data_mode.lower() in source.lower()):
                self._requires_split = True

        # If the source file/path does not meet the criteria
        # return an empty list.
        return get_filenames(source, self._source_format, self._data_mode)

    def _split_dataset(self, root):
        """Identify are the inputs being passed and split the data according
        to the data mode.

        The datasets will be splitted into 70% training, 10% validation, and
        20% testing.
        """
        # Get the set of filenames/arrays from the source input
        filenames = self._get_filenames(root)

        if self._requires_split:
            if self._split_train <= 1.0:
                train_size = int(self._split_train * len(filenames))
            else:
                train_size = int(self._split_train)

            if self._split_val <= 1.0:
                val_size = int(self._split_val * len(filenames))
            else:
                val_size = self._split_val

            if 'train' in self._data_mode:
                # Use 70% of the data for traning
                filenames = filenames[:train_size]
            elif 'val' in self._data_mode:
                # Use 10% of the data for validation
                filenames = filenames[train_size:train_size+val_size]
            elif 'test' in self._data_mode:
                # Use 20% of the data for testing
                filenames = filenames[train_size+val_size:]

        return filenames

    def _preload_files(self, filenames, data_group='0/0', data_axes='TCZYX',
                       compressed_input=False, rois=None):
        if rois is None:
            filenames_rois = list(map(parse_roi, filenames))
        else:
            filenames_rois = zip(filenames, rois)

        z_list = []
        rois_list = []
        compression_level = 0

        for id, (arr_src, rois) in enumerate(filenames_rois):
            arr, arr_shape, compression_level = \
                image_to_zarr(arr_src, self._patch_size, self._source_format,
                              data_group, compressed_input)
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

                    # Because data could have been passed in a different axes
                    # ordering, slicing is reordered to match the input data
                    # axes ordering.
                    roi = [roi['XYZCT'.index(a)] for a in data_axes]

                    # Take the ROI as the original size of the image
                    rois_list.append((id, tuple(roi)))

            else:
                roi = [slice(0, s, 1) for s in arr_shape]
                rois_list.append((id, tuple(roi)))

        return z_list, rois_list, compression_level

    def _compute_size(self, z_list, rois_list):
        imgs_shapes = [((roi[-2].stop - roi[-2].start)//roi[-2].step,
                        (roi[-1].stop - roi[-1].start)//roi[-1].step)
                       for _, roi in rois_list]
        imgs_sizes = np.cumsum(
            [0]
            + [compute_num_patches(W, self._patch_size,
                                   self._padding[0] + self._padding[1],
                                   self._stride[0])
               * compute_num_patches(H, self._patch_size,
                                     self._padding[2] + self._padding[3],
                                     self._stride[1])
               for H, W in imgs_shapes])
        # Get the upper bound of patches that can be obtained from all zarr
        # files (images with smaller size will be padded).
        max_H, max_W = np.max(np.array(imgs_shapes), axis=0)

        max_H = (self._patch_size
                 * compute_num_patches(max_H, self._patch_size,
                                       self._padding[0] + self._padding[1],
                                       self._stride[0]))
        max_W = (self._patch_size
                 * compute_num_patches(max_W, self._patch_size,
                                       self._padding[0] + self._padding[1],
                                       self._stride[0]))

        # Compute the size of the dataset from the valid patches
        if 'C' not in self._data_axes:
            org_channels = 1
        else:
            org_channels = z_list[0].shape[self._data_axes.index('C')]

        # Return the dataset size and the information about the dataset
        return (imgs_sizes[-1],
                max_H,
                max_W,
                org_channels,
                imgs_sizes,
                imgs_shapes)

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        i, tl_y, tl_x = compute_grid(index, self.imgs_shapes, self._imgs_sizes,
                                     self._patch_size,
                                     self._padding,
                                     self._stride)
        id, roi = self._rois_list[i]
        patch = get_patch(self.z_list[id].get_orthogonal_selection(roi),
                          self.imgs_shapes[id],
                          tl_y,
                          tl_x,
                          self._patch_size,
                          self._padding,
                          self._stride,
                          self._compression_level).squeeze()

        if self._transform is not None:
            patch = self._transform(patch.transpose(1, 2, 0))

        # Returns anything as label, to prevent an error during training
        return patch, [0]

    def get_channels(self):
        return self._org_channels

    def get_shape(self):
        return self._max_H, self._max_W


class LabeledZarrDataset(ZarrDataset):
    """ A labeled dataset based on the zarr dataset class.
        The densely labeled targets are extracted from group '1'.
    """
    def __init__(self, root, input_target_transform=None,
                 target_transform=None,
                 labels_group='labels/0/0',
                 labels_data_axes=None,
                 **kwargs):
        super(LabeledZarrDataset, self).__init__(root, **kwargs)

        # Open the labels from the labels group
        self._labels_group = labels_group
        if labels_data_axes is None:
            labels_data_axes = self._data_axes
        self._labels_data_axes = labels_data_axes
        self._lab_list, self._lab_rois_list, _ = \
            self._preload_files(self._filenames, data_group=self._labels_group,
                                data_axes=self._labels_data_axes,
                                compressed_input=False)

        # This is a transform that affects the geometry of the input, and then
        # it has to be applied to the target as well
        self._input_target_transform = input_target_transform

        # This is a transform that only affects the target
        self._target_transform = target_transform

    def __getitem__(self, index):
        i, tl_y, tl_x = compute_grid(index, self.imgs_shapes, self._imgs_sizes,
                                     self._patch_size,
                                     self._padding,
                                     self._stride)
        id, roi = self._rois_list[i]
        patch = get_patch(self.z_list[id].get_orthogonal_selection(roi),
                          self.imgs_shapes[id],
                          tl_y,
                          tl_x,
                          self._patch_size,
                          self._padding,
                          self._stride,
                          self._compression_level).squeeze()

        if self._transform is not None:
            patch = self._transform(patch.transpose(1, 2, 0))

        id, roi = self._lab_rois_list[i]
        target = get_patch(self._lab_list[id].get_orthogonal_selection(roi),
                           self.imgs_shapes[id],
                           tl_y,
                           tl_x,
                           self._patch_size,
                           self._padding,
                           self._stride,
                           0).astype(np.float32)

        if self._input_target_transform:
            patch, target = self._input_target_transform((patch, target))

        if self._target_transform:
            target = self._target_transform(target)

        # Returns anything as label, to prevent an error during training
        return patch, target
