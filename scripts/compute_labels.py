import argparse
import os

import aiohttp
import zarr

import tqdm

from numcodecs import Blosc


def label_zarr(z_url, label, output_filename):
    try:
        z_grp = zarr.open_consolidated(z_url, mode='r')

    except (KeyError, aiohttp.client_exceptions.ClientResponseError):
        z_grp = zarr.open(z_url, mode='r')

    mask_grp = z_grp["masks/0/0"]

    labeled_mask = mask_grp[:] * label

    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    z_grp = zarr.open(output_filename, mode="a")
    z_grp.create_dataset(name="masks/1/0", data=labeled_mask,
                         shape=labeled_mask.shape,
                         dtype=labeled_mask.dtype,
                         chunks=True,
                         compressor=compressor,
                         overwrite=True)
    z_grp["masks/1/0"].attrs.update({'label': label})


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute tissue mask from zarr files")
    parser.add_argument("-i", "--input", dest="inputs", type=str, nargs="+",
                        help="Input histology image stored in a zarr file and "
                             "its corresponding label")
    parser.add_argument("-o", "--output", dest="output_dir", type=str,
                        help="Output directory where to store the masks. "
                             "Ignore this to store the mask in the same input "
                             "zarr file",
                        default=None)
    args = parser.parse_args()

    if not isinstance(args.inputs, list):
        args.inputs = [args.inputs]

    fn_list = []
    for filename in args.inputs:
        if filename.lower().endswith(".txt"):
            with open(filename, mode="r") as fp:
                fn_list += [fn.strip(" \n") for fn in fp.readlines()]
        elif (os.path.isdir(filename)
          and ".zarr" not in filename.lower()):
            fn_list += [os.path.join(filename, fn)
                        for fn in os.listdir(filename)
                        if ".zarr" in fn.lower()]
        elif ".zarr" in filename.lower():
            fn_list.append(filename)

    q = tqdm.tqdm(total=len(fn_list))
    for filename in fn_list:
        filename, label = filename.split("::")
        label = int(label)

        if args.output_dir is not None:
            output_filename = os.path.join(args.output_dir,
                                           os.path.basename(filename))
        else:
            output_filename = filename

        label_zarr(filename, label, output_filename)

        q.set_description("Completed labeling file %s" % filename)
        q.update()

    q.close()
