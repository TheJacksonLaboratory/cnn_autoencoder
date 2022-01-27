import os

from matplotlib.pyplot import jet
from tqdm import tqdm
import argparse
import zarr

from numcodecs import Blosc


def save_group(fn, z, label, dst_dir):
    dst_fn = os.path.join(dst_dir, fn)
    group = zarr.open_group(dst_fn, mode='w')

    z_input = group.create_group('0', overwrite=True)
    z_label = group.create_group('1', overwrite=True)

    comp = Blosc(cname='zlib', clevel=5)
    z_input.create_dataset('0', data=z, compressor=comp)
    z_label.create_dataset('0', data=label)


def main(inputs_fn, labels_fn, dst_dir):
    z_inputs = zarr.open(inputs_fn, 'r')
    z_labels = zarr.open(labels_fn, 'r')
    print(z_inputs.info)

    q = tqdm(total=z_inputs.shape[-1])
    for i in range(z_inputs.shape[-1]):
        save_group('%04d.zarr' % i, z_inputs[..., i].transpose(2, 0, 1), z_labels[..., i].transpose(2, 0, 1), dst_dir)
        q.update()

    q.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert a set of arrays stored in zarr format into a dataset of independent zarr files')
    parser.add_argument('-i', '--input', type=str, dest='inputs_fn', help='A zarr file containing the input images')
    parser.add_argument('-l', '--label', type=str, dest='labels_fn', help='A zarr file containing the corresponding labels')
    parser.add_argument('-o', '--out-dir', type=str, dest='dst_dir', help='Path to a directory where to store the dataset')

    args = parser.parse_args()

    main(args.inputs_fn, args.labels_fn, args.dst_dir)
