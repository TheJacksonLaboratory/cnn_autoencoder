import math
import os
import numpy as np
import zarr
from PIL import Image
from numcodecs import Blosc

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def load_image(filename, patch_size):
    im = Image.open(filename, mode="r")
    arr = np.array(im)
    
    if len(im.getbands()) == 1:
        arr = np.tile(np.array(im), (3, 1, 1))
    else:
        arr = arr.transpose(2, 0, 1)

    # Pad the image to the closest size multiple of 2
    H, W = arr.shape[1:]
    pad_bottom = int(math.ceil(H / patch_size) * patch_size) - H
    pad_right = int(math.ceil(W / patch_size) * patch_size) - W

    if pad_bottom > 0 or pad_right > 0:
        arr = np.pad(arr, ((0, 0), (0, pad_bottom), (0, pad_right)))
    
    return arr
    

def compute_grid(index, n_files, min_H, min_W, patch_size):
    """ Compute the coordinate on a grid of indices corresponding to 'index'.
    The indices are in the form of [i, tl_x, tl_y], where 'i' is the file index.
    tl_x and tl_y are the top left coordinates of the patched image.
    To get a patch from any image, tl_y and tl_x must be multiplied by patch_size.
    
    Parameters:
    ----------
    index : int
        Index of the patched dataset Between 0 and 'total_patches'-1
    n_files : int
        Number of image files in the dataset
    min_H : int
        Minimum image height among all images
    min_W : int
        Minimum image width among all images
    patch_size : int
        The size of each squared patch
        
    Returns
    -------
    i : int
    tl_y : int
    tl_x : int
    """
    # This allows to generate virtually infinite data from bootstrapping the same data
    index %= (n_files * min_H * min_W) // patch_size**2

    # Get the file index among the available file names
    i = index // ((min_H * min_W) // patch_size**2)
    index %= (min_H * min_W) // patch_size**2

    # Get the patch position in the file
    tl_y = index // (min_W // patch_size)
    tl_x = index % (min_W // patch_size)

    return i, tl_y, tl_x


def get_patch(z, tl_y, tl_x, patch_size, offset=0):
    """
    Gets a squared region from an array z (numpy or zarr).

    Parameters:
    ----------
    z : numpy.array or zarr.array
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

    # TODO extract this information from the zarr metadata
    c = max(z.shape[:-2])
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

    # Pad the patch using the reflect mode
    if offset > 0:
        patch = np.pad(patch, ((0, 0), (tl_y - tl_y_offset, br_y_offset - br_y), (tl_x - tl_x_offset, br_x_offset - br_x)), mode='reflect', reflect_type='even')

    return patch


class ZarrDataset(Dataset):
    """ A zarr-based dataset.
        The structure of the zarr file is considered fixed, and only the component '0/0' is used.
        Only two-dimensional (+color channels) data is supported by now.
    """
    def __init__(self, root, patch_size, dataset_size=-1, level=0, mode='train', offset=0, transform=None, source_format='zarr', **kwargs):
        # Get all the filenames in the root folder
        if isinstance(root, list):
            self._filenames = root
        
        elif root.lower().endswith(source_format):
            # If the input is a single zarr file, take it directly as the only file
            self._filenames = [root]
        
        else:
            # If a root directory was provided, create a dataset from the images contained by splitting the set into training, validation, and testing subsets.
            self._filenames = list(map(lambda fn: os.path.join(root, fn), [fn for fn in sorted(os.listdir(root)) if fn.lower().endswith(source_format)]))

            if mode == 'train':
                # Use 70% of the data for traning
                self._filenames = self._filenames[:int(0.7 * len(self._filenames))]
            elif mode == 'val':
                # Use 10% of the data for validation
                self._filenames = self._filenames[int(0.7 * len(self._filenames)):int(0.8 * len(self._filenames))]
            else:
                # Use 20% of the data for testing
                self._filenames = self._filenames[int(0.8 * len(self._filenames)):]

        self._patch_size = patch_size
        self._dataset_size = dataset_size
        self._transform = transform
        self._offset = offset
        
        self._n_files = len(self._filenames)
        self._level = level
        self._source_format = source_format

        self._z_list = self._preload_files()

        # Get the lower bound of patches that can be obtained from all zarr files
        min_H = min([z.shape[-2] for z in self._z_list])
        min_W = min([z.shape[-1] for z in self._z_list])
        
        self._min_H = self._patch_size * (min_H // self._patch_size)
        self._min_W = self._patch_size * (min_W // self._patch_size)
        
        if dataset_size < 0:
            self._dataset_size = int(np.ceil(self._min_H * self._min_W / self._patch_size**2)) * self._n_files
        else:
            self._dataset_size = dataset_size

    def _preload_files(self, group='0'):
        if self._source_format == 'zarr':
            # Open the tile downscaled to 'level'      
            z_list = [zarr.open(fn, mode='r')['%s/%s' % (group, self._level)] for fn in self._filenames]

        else:
            # Loading the images using PIL. This option is restricted to formats supported by PIL
            compressor = Blosc(cname='zlib', clevel=0, shuffle=Blosc.BITSHUFFLE)
            
            z_list = [zarr.array(load_image(fn, self._patch_size), chunks=(3, self._patch_size, self._patch_size), compressor=compressor)
                            for fn in self._filenames
                    ]

        return z_list

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        i, tl_y, tl_x = compute_grid(index, self._n_files, self._min_H, self._min_W, self._patch_size)

        patch = get_patch(self._z_list[i], tl_y, tl_x, self._patch_size, self._offset).squeeze()

        if self._transform is not None:
            patch = self._transform(patch.transpose(1, 2, 0))
        
        # Returns anything as label, to prevent an error during training
        return patch, [0]

    def get_shape(self):
        return self._min_H, self._min_W


class LabeledZarrDataset(ZarrDataset):
    """ A labeled dataset based on the zarr dataset class.        
        The dense labels are extracted from group '1'.
    """
    def __init__(self, root, patch_size, dataset_size=-1, level=0, mode='train', offset=0, transform=None, compression_level=0, compressed_input=False):
        super(LabeledZarrDataset, self).__init__(root, patch_size, dataset_size, level, mode, offset, transform)
        
        # Open the labels from group 1
        self._lab_list = self._preload_files(group='1')

        self._compression_level = compression_level
        self._compressed_input = compressed_input
        
    def __getitem__(self, index):
        i, tl_y, tl_x = compute_grid(index, self._n_files, self._min_H, self._min_W, self._patch_size)

        patch = get_patch(self._z_list[i], tl_y, tl_x, self._patch_size, self._offset).squeeze()

        if self._transform is not None:
            patch = self._transform(patch.transpose(1, 2, 0))
            
        patch_size = self._patch_size * ((2**self._compression_level) if self._compressed_input else 1)
        target = get_patch(self._lab_list[i], tl_y, tl_x, patch_size, 0).astype(np.float32)
        
        # Returns anything as label, to prevent an error during training
        return patch, target


def get_zarr_transform(normalize=True):
    prep_trans_list = [transforms.ToTensor(),
         transforms.ConvertImageDtype(torch.float32)
        ]
    
    if normalize:
        prep_trans_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
            
    return transforms.Compose(prep_trans_list)


def get_zarr_dataset(data_dir, task='autoencoder', patch_size=128, batch_size=1, val_batch_size=1, workers=0, mode='training', normalize=True, offset=0, gpu=False, pyramid_level=0, compressed_input=False, compression_level=0, **kwargs):
    """ Creates a data queue using pytorch\'s DataLoader module to retrieve patches from histology images.
    The size of the data queue can be virtually infinite, for that reason, a cnservative size has been defined using the following global variables.
    1. TRAIN_DATASIZE
    2. VALID_DATASIZE
    3. TEST_DATASIZE
    """

    if task == 'autoencoder':
        prep_trans = get_zarr_transform(normalize=normalize)
        histo_dataset = ZarrDataset
        TRAIN_DATASIZE = 1200000
        VALID_DATASIZE = 50000
        TEST_DATASIZE = 200000
        
    elif task == 'segmentation':
        prep_trans = get_zarr_transform(normalize=normalize)
        if mode == 'training':
            histo_dataset = LabeledZarrDataset
        else:
            histo_dataset = ZarrDataset
        TRAIN_DATASIZE = -1
        VALID_DATASIZE = -1
        TEST_DATASIZE = -1
    
    if mode != 'training':
        hist_data = histo_dataset(data_dir, patch_size=patch_size, dataset_size=TEST_DATASIZE, level=pyramid_level, mode='test', transform=prep_trans, offset=offset, compression_level=compression_level, compressed_input=compressed_input)
        test_queue = DataLoader(hist_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=gpu)
        return test_queue

    hist_train_data = histo_dataset(data_dir, patch_size=patch_size, dataset_size=TRAIN_DATASIZE, level=pyramid_level, mode='train', transform=prep_trans, offset=offset, compression_level=compression_level, compressed_input=compressed_input)
    hist_valid_data = histo_dataset(data_dir, patch_size=patch_size, dataset_size=VALID_DATASIZE, level=pyramid_level, mode='val', transform=prep_trans, offset=offset, compression_level=compression_level, compressed_input=compressed_input)

    train_queue = DataLoader(hist_train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=gpu)
    valid_queue = DataLoader(hist_valid_data, batch_size=val_batch_size, shuffle=False, num_workers=workers, pin_memory=gpu)

    return train_queue, valid_queue


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse
    from time import perf_counter

    parser = argparse.ArgumentParser('Test zarr-based datasets generation and loading with a pytorch\'s DataLoader')

    parser.add_argument('-d', '--dir', dest='root', help='Root directory where the zarr files are stored')
    parser.add_argument('-w', '--workers', type=int, dest='workers', help='Number of workers', default=0)
    parser.add_argument('-sb', '--batchsize', type=int, dest='batch_size', help='Batch size', default=16)
    parser.add_argument('-nb', '--nbatches', type=int, dest='n_batches', help='Number of batches to show', default=20)
    parser.add_argument('-p', '--patch', type=int, dest='patch_size', help='Size of the patch -> patch_size x patch_size', default=128)
    parser.add_argument('-sh', '--shuffled', action='store_true', dest='shuffled', help='Shuffle the data?')
    parser.add_argument('-t', '--task', dest='task', help='Task for what the data is used', choices=['autoencoder', 'segmentation'])
    parser.add_argument('-m', '--mode', dest='mode', help='The network use mode', choices=['train', 'val', 'test'])
    parser.add_argument('-o', '--offset', type=int, dest='offset', help='Offset added to the patches', default=0)
    parser.add_argument('-cl', '--comp-level', type=int, dest='compression_level', help='Compression level of the input', default=0)
    parser.add_argument('-ex', '--extension', type=str, dest='source_format', help='Format of the input files', default='.zarr')
    
    args = parser.parse_args()

    if args.task == 'autoencoder':
        dataset = ZarrDataset
    else:
        dataset = LabeledZarrDataset

    args.compressed_input = args.compression_level > 0
    ds = dataset(**args.__dict__)

    print('Dataset size:', len(ds))
    
    dl = DataLoader(ds, shuffle=args.shuffled, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers)
    print('Min image size: (%d, %d)' % (ds._min_H, ds._min_W))

    t_ini = perf_counter()

    for i, (x, t) in enumerate(dl):
        print('Batch {} of size: {}, target: {}'.format(i, x.size(), t.size() if isinstance(t, torch.torch.Tensor) else None))
        plt.imshow(x[0].permute(1, 2, 0))
        plt.show()

        if i >= args.n_batches:
            break

    e_time = perf_counter() - t_ini
    print('Elapsed time:', e_time)