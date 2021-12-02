import numpy as np
import zarr
import dask
import dask.array as da

import torch
from torch.utils.data import Dataset


def _compute_grid(n_files, patch_size, H, W):
    """ Pre-compute a grid of indices to extract the patches from a lazy loaded set of zarr arrays.
    The indices are in the form of [i, tl_x, tl_y], where 'i' is the file index.
    tl_x and tl_y are the top left coordinates of the patch in the original image.
    """
    indices_grid = [(i, tl_y, tl_x) for i in range(n_files) for tl_y in np.arange(0, H, patch_size) for tl_x in np.arange(0, W, patch_size)]

    return indices_grid


def _get_zarr(filenames, level=0):
    # Get the arrays from the zarr files in a lazy manner using dask
    lazy_arr = [da.from_zarr(fn, component='0/%s' % level) for fn in filenames]
    return da.stack(lazy_arr, axis=0)


class KOMP_zarr(Dataset):
    """ The KOMP dataset is converted from raw data to zarr using bioformats

    """
    def __init__(self, filenames, patch_size, dataset_size=-1, level=0, **kwargs):
        self._filenames = filenames
        self._patch_size = patch_size
        self._dataset_size = dataset_size

        self._lazy_arr = _get_zarr(filenames, level)
        
        n_files, _, _, _, H, W = self._lazy_arr.shape

        if dataset_size < 0:
            self._dataset_size = int(np.ceil(H * W / patch_size**2)) * n_files
        else:
            self._dataset_size = dataset_size

        self._grid = _compute_grid(n_files, patch_size, H, W)

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        i, tl_y, tl_x = self._grid[index]
        
        patch = self._lazy_arr[i, 0, :, 0, tl_y:(tl_y + self._patch_size), tl_x:(tl_x + self._patch_size)]
        
        return patch.compute()


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import argparse
    import os

    parser = argparse.ArgumentParser('Test histology datasets')

    parser.add_argument('-d', '--dir', dest='root_dir', help='Root directory where the zarr files are stored')

    args = parser.parse_args()

    filenames = list(map(lambda fn: os.path.join(args.root_dir, fn) ,filter(lambda fn: '.zarr' in fn, os.listdir(args.root_dir))))

    print('Found the following zarr files:\n', filenames)

    ds = KOMP_zarr(filenames, 512, -1, 0)

    print('Dataset size:', len(ds))

    dl = DataLoader(ds, shuffle=False, batch_size=8, pin_memory=True)

    for i, x in enumerate(dl):
        print('Batch %d of size: [%i, %i, %i, %i]' % (i, *x.size()))
        if i > 10:
            break

    