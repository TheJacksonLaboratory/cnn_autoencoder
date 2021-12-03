import os
import numpy as np
import zarr

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms


class Histology_zarr(Dataset):
    """ A histology dataset that has been converted from raw data to zarr using bioformats.
        The structure of the zarr file is considered fixed. For this reason, only the component '0/0' is used.
        From that component, temporal and layer dimensions are discarded, keeping only channels, and spatial dimension.
    """
    def __init__(self, root, patch_size, dataset_size=-1, level=0, transform=None, **kwargs):
        # Get all the filenames in the root folder
        self._filenames = list(map(lambda fn: os.path.join(root, fn), filter(lambda fn: '.zarr' in fn, sorted(os.listdir(root)))))

        self._patch_size = patch_size
        self._dataset_size = dataset_size
        self._transform = transform

        n_files = len(self._filenames)

        self._z_list = [zarr.open(fn, mode='r')['0/%s' % level] for fn in filenames]
        
        # Get the lower bound of patches that can be obtained from all zarr files
        min_H = min([z.shape[-2] for z in self._z_list])
        min_W = min([z.shape[-1] for z in self._z_list])

        self._min_H = patch_size * (min_H // patch_size)
        self._min_W = patch_size * (min_W // patch_size)

        if dataset_size < 0:
            self._dataset_size = int(np.ceil(self._min_H * self._min_W / patch_size**2)) * n_files
        else:
            self._dataset_size = dataset_size

    def __len__(self):
        return self._dataset_size

    def _compute_grid(self, index):
        """ Compute the coordinate on a grid of indices corresponding to 'index'.
        The indices are in the form of [i, tl_x, tl_y], where 'i' is the file index.
        tl_x and tl_y are the top left coordinates of the patch in the original image.
        """
        i = index // (self._min_H * self._min_W // self._patch_size**2)
        index %= (self._min_H * self._min_W // self._patch_size**2)
        tl_y = index // (self._min_W // self._patch_size)
        tl_x = index % (self._min_W // self._patch_size)

        return i, tl_y * self._patch_size, tl_x * self._patch_size

    def __getitem__(self, index):        
        i, tl_y, tl_x = self._compute_grid(index)
        patch = self._z_list[i][0, :, 0, tl_y:(tl_y + self._patch_size), tl_x:(tl_x + self._patch_size)]

        patch = self._transform(patch)

        return patch


def get_Histology(args, normalize):
    prep_trans_list = [transforms.Pad(2),
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float32)
        ]
    
    if normalize:
        # prep_trans_list.append(transforms.Normalize(mean=0.0, std=1.0))
        prep_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))
            
    prep_trans = transforms.Compose(prep_trans_list)

    # If testing the model, return the test set from MNIST
    if args.mode != 'training':
        mnist_data = Histology_zarr(filenames, train=False, transform=prep_trans)
        test_queue = DataLoader(mnist_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        return test_queue

    mnist_data = Histology_zarr(filenames, train=True, transform=prep_trans)

    train_ds, valid_ds = random_split(mnist_data, (55000, 5000))
    train_queue = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_queue = DataLoader(valid_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers)

    return train_queue, valid_queue


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import argparse
    import os
    from time import perf_counter

    parser = argparse.ArgumentParser('Test histology datasets')

    parser.add_argument('-d', '--dir', dest='root_dir', help='Root directory where the zarr files are stored')
    parser.add_argument('-w', '--workers', type=int, dest='workers', help='Number of workers', default=0)
    parser.add_argument('-sb', '--batchsize', type=int, dest='batch_size', help='Batch size', default=16)
    parser.add_argument('-nb', '--nbatches', type=int, dest='n_batches', help='Number of batches to show', default=20)
    parser.add_argument('-p', '--patch', type=int, dest='patch_size', help='Size of the patch -> patch_size x patch_size', default=128)
    parser.add_argument('-sh', '--shuffled', action='store_true', dest='shuffled', help='Shuffle the data?')

    args = parser.parse_args()

    filenames = list(map(lambda fn: os.path.join(args.root_dir, fn) ,filter(lambda fn: '.zarr' in fn, os.listdir(args.root_dir))))

    print('Found the following zarr files:\n', filenames)

    ds = Histology_zarr(filenames, args.patch_size, -1, 0)

    print('Dataset size:', len(ds))

    dl = DataLoader(ds, shuffle=args.shuffled, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers)

    t_ini = perf_counter()

    for i, x in enumerate(dl):
        print('Batch %d of size: [%i, %i, %i, %i]' % (i, *x.size()))
        if i >= args.n_batches:
            break

    e_time = perf_counter() - t_ini
    print('Elapsed time:', e_time)