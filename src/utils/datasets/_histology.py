import os
import numpy as np
import zarr

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class Histology_zarr(Dataset):
    """ A histology dataset that has been converted from raw data to zarr using bioformats.
        The structure of the zarr file is considered fixed. For this reason, only the component '0/0' is used.
        From that component, temporal and layer dimensions are discarded, keeping only channels, and spatial dimension.
    """
    def __init__(self, root, patch_size, dataset_size=-1, level=0, mode='train', offset=0, transform=None, **kwargs):
        # Get all the filenames in the root folder
        if isinstance(root, list):
            self._filenames = root
        
        elif root.endswith('.zarr'):
            # If the input is a single zarr file, take it directly as the only file
            self._filenames = [root]
        
        else:            
            self._filenames = list(map(lambda fn: os.path.join(root, fn), [fn for fn in sorted(os.listdir(root)) if '.zarr' in fn]))
            if mode == 'train':
                self._filenames = self._filenames[:int(0.7 * len(self._filenames))]
            elif mode == 'val':
                self._filenames = self._filenames[int(0.7 * len(self._filenames)):int(0.8 * len(self._filenames))]
            else:
                self._filenames = self._filenames[int(0.8 * len(self._filenames)):]

        self._patch_size = patch_size
        self._dataset_size = dataset_size
        self._transform = transform
        self._offset = offset
        
        self._n_files = len(self._filenames)

        # Open the tile downscaled to 'level'      
        self._z_list = [zarr.open(fn, mode='r')['0/%s' % level] for fn in self._filenames]

        # Get the lower bound of patches that can be obtained from all zarr files
        min_H = min([z.shape[-2] for z in self._z_list])
        min_W = min([z.shape[-1] for z in self._z_list])
        
        self._min_H = self._patch_size * (min_H // self._patch_size)
        self._min_W = self._patch_size * (min_W // self._patch_size)
        
        if dataset_size < 0:
            self._dataset_size = int(np.ceil(self._min_H * self._min_W / self._patch_size**2)) * self._n_files
        else:
            self._dataset_size = dataset_size

    def __len__(self):
        return self._dataset_size

    def _compute_grid(self, index):
        """ Compute the coordinate on a grid of indices corresponding to 'index'.
        The indices are in the form of [i, tl_x, tl_y], where 'i' is the file index.
        tl_x and tl_y are the top left coordinates of the patched image.
        """
                
        # This allows to generate virtually infinite data from bootstrapping the same data
        index %= (self._n_files * self._min_H * self._min_W) // self._patch_size**2

        # Get the file index among the available file names
        i = index // ((self._min_H * self._min_W) // self._patch_size**2)
        index %= (self._min_H * self._min_W) // self._patch_size**2

        # Get the patch position in the file
        tl_y = index // (self._min_W // self._patch_size)
        tl_x = index % (self._min_W // self._patch_size)

        return i, tl_y, tl_x

    def _get_patch(self, i, tl_y, tl_x, patch_size, z_list, offset_patch=True):
        tl_y *= patch_size
        tl_x *= patch_size
        
        # TODO extract this information from the zarr metadata
        c = max(z_list[i].shape[:-2])
        H, W = z_list[i].shape[-2:]

        offset = self._offset if offset_patch else 0
        tl_y_offset = tl_y - offset
        tl_x_offset = tl_x - offset
        br_y_offset = tl_y + patch_size + offset
        br_x_offset = tl_x + patch_size + offset
        
        tl_y = max(tl_y_offset, 0)
        tl_x = max(tl_x_offset, 0)
        br_y = min(br_y_offset, H)
        br_x = min(br_x_offset, W)

        patch = z_list[i][..., tl_y:br_y, tl_x:br_x].squeeze()

        if c == 1:
            patch = patch[np.newaxis, ...]

        # Pad the patch using the reflect mode
        if offset > 0:
            patch = np.pad(patch, ((0, 0), (tl_y - tl_y_offset, br_y_offset - br_y), (tl_x - tl_x_offset, br_x_offset - br_x)), mode='reflect', reflect_type='even')

        return patch

    def __getitem__(self, index):
        i, tl_y, tl_x = self._compute_grid(index)
        print('Retrieving %d, (%d, %d), from: %s' % (i, tl_y, tl_x, self._filenames[i]))

        patch = self._get_patch(i, tl_y, tl_x, self._patch_size, self._z_list).squeeze()

        if self._transform is not None:
            patch = self._transform(patch.transpose(1, 2, 0))
        
        # Returns anything as label, to prevent an error during training
        return patch, [0]


class Histology_seg_zarr(Histology_zarr):
    """ A histology dataset that has been converted from raw data to zarr using the convolutional autoencoder.
        The structure of the zarr file is considered fixed. For this reason, only the component '0/0' is used.
        The labels are extracted from group '1'.
    """
    def __init__(self, root, patch_size, dataset_size=-1, level=0, mode='train', offset=0, transform=None, compression_level=0, compressed_input=False):
        super(Histology_seg_zarr, self).__init__(root, patch_size, dataset_size, level, mode, offset, transform)
        
        # Open the labels from group 1
        self._lab_list = [zarr.open(fn, mode='r')['1/0'] for fn in self._filenames]
        self._compression_level = compression_level
        self._compressed_input = compressed_input
        
    def __getitem__(self, index):
        i, tl_y, tl_x = self._compute_grid(index)

        patch = self._get_patch(i, tl_y, tl_x, self._patch_size, self._z_list).squeeze()

        if self._transform is not None:
            patch = self._transform(patch.transpose(1, 2, 0))
            
        patch_size = self._patch_size * ((2**self._compression_level) if self._compressed_input else 1)
        target = self._get_patch(i, tl_y, tl_x, patch_size, self._lab_list, offset_patch=False).astype(np.float32)
        
        # Returns anything as label, to prevent an error during training
        return patch, target


def get_histo_transform(normalize=True):
    prep_trans_list = [transforms.ToTensor(),
         transforms.ConvertImageDtype(torch.float32)
        ]
    
    if normalize:
        prep_trans_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
            
    return transforms.Compose(prep_trans_list)


def get_Histology(args, offset=0, normalize=False):
    """ Creates a data queue using pytorch\'s DataLoader module to retrieve patches from histology images.
    The size of the data queue can be virtually infinite, for that reason, a cnservative size has been defined using the following global variables.
    1. TRAIN_DATASIZE
    2. VALID_DATASIZE
    3. TEST_DATASIZE
    """

    if not hasattr(args, 'patch_size') or args.patch_size < 0:
        patch_size = 512
    else:
        patch_size = args.patch_size
    
    if hasattr(args, 'pyramid_level'):
        level = args.pyramid_level
    else:
        level = 0

    if not hasattr(args, 'compressed_input'):
        compressed_input = False
    else:
        compressed_input = args.compressed_input
    
    if not hasattr(args, 'compression_level'):
        compression_level = 0
    else:
        compression_level = args.compression_level
    
    if not hasattr(args, 'normalize'):
        normalize = True
    else:
        normalize = args.normalize
    
    if args.task == 'autoencoder':
        prep_trans = get_histo_transform(normalize=normalize)
        histo_dataset = Histology_zarr
        TRAIN_DATASIZE = 1200000
        VALID_DATASIZE = 50000
        TEST_DATASIZE = 200000
        
    elif args.task == 'segmentation':
        prep_trans = get_histo_transform(normalize=normalize)
        if args.mode == 'training':
            histo_dataset = Histology_seg_zarr
        else:
            histo_dataset = Histology_zarr
        TRAIN_DATASIZE = -1
        VALID_DATASIZE = -1
        TEST_DATASIZE = -1
            
    if args.mode != 'training':
        print('Files\n', args.data_dir[:10])        
        hist_data = histo_dataset(args.data_dir, patch_size=patch_size, dataset_size=TEST_DATASIZE, level=level, mode='test', transform=prep_trans, offset=offset, compression_level=compression_level, compressed_input=compressed_input)
        test_queue = DataLoader(hist_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.gpu)
        return test_queue

    hist_train_data = histo_dataset(args.data_dir, patch_size=patch_size, dataset_size=TRAIN_DATASIZE, level=level, mode='train', transform=prep_trans, offset=offset, compression_level=compression_level, compressed_input=compressed_input)
    hist_valid_data = histo_dataset(args.data_dir, patch_size=patch_size, dataset_size=VALID_DATASIZE, level=level, mode='val', transform=prep_trans, offset=offset, compression_level=compression_level, compressed_input=compressed_input)

    train_queue = DataLoader(hist_train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.gpu)
    valid_queue = DataLoader(hist_valid_data, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.gpu)

    return train_queue, valid_queue


if __name__ == '__main__':
    import argparse
    from time import perf_counter

    parser = argparse.ArgumentParser('Test histology datasets generation and loading with a pytorch\'s DataLoader')

    parser.add_argument('-d', '--dir', dest='root_dir', help='Root directory where the zarr files are stored')
    parser.add_argument('-w', '--workers', type=int, dest='workers', help='Number of workers', default=0)
    parser.add_argument('-sb', '--batchsize', type=int, dest='batch_size', help='Batch size', default=16)
    parser.add_argument('-nb', '--nbatches', type=int, dest='n_batches', help='Number of batches to show', default=20)
    parser.add_argument('-p', '--patch', type=int, dest='patch_size', help='Size of the patch -> patch_size x patch_size', default=128)
    parser.add_argument('-sh', '--shuffled', action='store_true', dest='shuffled', help='Shuffle the data?')
    parser.add_argument('-t', '--task', dest='task', help='Task for what the data is used', choices=['autoencoder', 'segmentation'])
    parser.add_argument('-m', '--mode', dest='mode', help='The network use mode', choices=['train', 'val', 'test'])
    parser.add_argument('-o', '--offset', type=int, dest='offset', help='Offset added to the patches', default=0)
    parser.add_argument('-cl', '--comp-level', type=int, dest='compression_level', help='Compression level of the input', default=0)
    
    args = parser.parse_args()

    if args.task == 'autoencoder':
        histo_dataset = Histology_zarr
    else:
        histo_dataset = Histology_seg_zarr

    compressed_input = args.compression_level > 0
    ds = histo_dataset(args.root_dir, args.patch_size, mode=args.mode, dataset_size=-1, level=0, offset=args.offset, compression_level=args.compression_level, compressed_input=compressed_input)

    print('Dataset size:', len(ds))

    dl = DataLoader(ds, shuffle=args.shuffled, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers)

    t_ini = perf_counter()

    for i, (x, t) in enumerate(dl):
        x = x.cuda()
        print('Batch {} of size: {}, target: {}'.format(i, x.size(), t.size() if isinstance(t, torch.torch.Tensor) else None))
        if i >= args.n_batches:
            break

    e_time = perf_counter() - t_ini
    print('Elapsed time:', e_time)