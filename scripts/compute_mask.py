import argparse
import urllib.request
import urllib.error
import os

import math
import zarr
import dask.array as da
import numpy as np

from skimage import morphology, color, filters, transform

import tqdm

from numcodecs import Blosc


def get_mask(scaled_wsi):
    gray = color.rgb2gray(scaled_wsi)
    thresh = filters.threshold_otsu(gray)
    mask = gray > thresh

    mask = morphology.remove_small_objects(mask==0, min_size=16 * 16,
                                           connectivity=2)
    mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)

    mask = morphology.binary_dilation(mask, morphology.disk(16))

    return mask


def downscale_chunk(chunk, scale):
    h, w = chunk.shape[:2]
    output_shape = (int(math.ceil(scale * h)), int(math.ceil(scale * w)))

    dwn_chunk = transform.resize(chunk, output_shape=output_shape)
    return dwn_chunk


def compute_tissue_mask(zarr_url, mag=40, scaled_mag=1.25, data_axes="XYZCT",
                        array_keys=None):
    z = zarr.open(zarr_url, mode="r")

    a_ch, a_H, a_W = [data_axes.index(a) for a in "CYX"]

    shapes = []
    if array_keys is None:
        array_keys = z["0"].array_keys()

    for k in array_keys:
        shapes.append((z["0/" + k].shape[a_H], z["0/" + k].shape[a_W], k))

    H, W, _ = max(shapes)
    scaled_H = int(math.ceil(H * scaled_mag / mag))
    scaled_W = int(math.ceil(W * scaled_mag / mag))

    dist = [((scaled_H - h) ** 2 + (scaled_W - w) ** 2, h, w, k)
            for h, w, k in shapes]

    _, cls_H, cls_W, cls_key = min(dist)

    base_wsi = da.from_zarr(zarr_url, component="0/" + cls_key)

    # Reschunk input to have all channels inside each chunk
    chunks = (base_wsi.chunksize[a_W], base_wsi.chunksize[a_H], 1, 3, 1)
    chunks = [chunks["XYZCT".index(a)] for a in data_axes]
    base_wsi = base_wsi.rechunk(chunks=chunks)

    unused_axes = list(set(data_axes) - set("YXC"))
    transpose_order = [a_H, a_W, a_ch]
    transpose_order += [data_axes.index(a) for a in unused_axes]
    base_wsi = base_wsi.transpose(transpose_order)
    base_wsi = base_wsi.reshape(cls_H, cls_W, 3)

    scale = scaled_H / base_wsi.shape[0]
    scaled_wsi = base_wsi.map_blocks(downscale_chunk,
                                     scale=scale,
                                     dtype=np.uint8,
                                     drop_axis=(0, 1),
                                     new_axis=(0, 1),
                                     meta=np.empty((0), dtype=np.uint8))

    scaled_wsi = scaled_wsi.compute()

    mask_wsi = get_mask(scaled_wsi)
    return mask_wsi


def mask_zarr(z_url, output_filename, scaled_mag=1.25, default_mag=40,
              data_axes="XYZCT",
              array_keys=None):

    z_ome = None
    mag = None
    if len(urllib.parse.urlparse(z_url).scheme) > 0:
        try:
            with urllib.request.urlopen(os.path.join(z_url, "OME/METADATA.ome.xml")) as f:
                z_ome = f.read().decode('utf-8')
        except urllib.error.HTTPError:
            pass
    else:
        if os.path.isfile(os.path.join(z_url, "OME/METADATA.ome.xml")):
            with open(os.path.join(z_url, "OME/METADATA.ome.xml")) as f:
                z_ome = f.read()

    if z_ome is not None:
        mag_pos_ini = z_ome.find("AppMag")
        if mag_pos_ini >= 0:
            mag_pos_ini = z_ome.find("=", mag_pos_ini) + 1
            mag_pos_end = z_ome.find("|", mag_pos_ini)
            mag = float(z_ome[mag_pos_ini:mag_pos_end].strip(" "))

        mag_pos_ini = z_ome.find("Power")
        if mag_pos_ini >= 0:
            mag_pos_ini = z_ome.find("Value", mag_pos_ini)
            mag_pos_ini = z_ome.find(">", mag_pos_ini) + 1
            mag_pos_end = z_ome.find("<", mag_pos_ini)
            mag = float(z_ome[mag_pos_ini:mag_pos_end].strip(" "))

    if mag is None:
        print("Could not get magnification from metadata, setting "
              "magnification to default value %i" % default_mag)
        mag = default_mag


    mask_wsi = compute_tissue_mask(z_url, mag=mag,
                                   scaled_mag=scaled_mag,
                                   data_axes=data_axes,
                                   array_keys=array_keys)

    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

    z_grp = zarr.open(output_filename, mode="a")
    z_grp.create_dataset(name="masks/0/0", data=mask_wsi, shape=mask_wsi.shape,
                         dtype=mask_wsi.dtype,
                         chunks=True,
                         compressor=compressor,
                         overwrite=True)
    z_grp["masks/0/0"].attrs.update({'scaled_mag': scaled_mag, 
                                     'scale': scaled_mag / mag})


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute tissue mask from zarr files")
    parser.add_argument("-i", "--input", dest="inputs", type=str, nargs="+",
                        help="Input histology image stored in a zarr file")
    parser.add_argument("-o", "--output", dest="output_dir", type=str,
                        help="Output directory where to store the masks. "
                             "Ignore this to store the mask in the same input "
                             "zarr file",
                        default=None)
    parser.add_argument("-da", "--data-axes", dest="data_axes", type=str,
                        default="XYZCT",
                        help="Ordering of the axes in the zarr files")
    parser.add_argument("-s", "--scale", dest="scaled_mag", type=float,
                        default=1.25,
                        help="Magnification at whch the tissue is computed")
    parser.add_argument("-ak", "--array-keys", dest="array_keys", type=str,
                        default=None, nargs="+",
                        help="Keys where arrays are stored within the used "
                             "zarr group. This is used in case that the list "
                             "of keys cannot be read from the file (e.g., a "
                             "zarr file stored in a S3 bucket)")
    parser.add_argument("-p", "--progress-bar", dest="progress_bar",
                        action="store_true",
                        default=False,
                        help="Show progress bar per file processed")
    parser.add_argument("-dm", "--default-mag", dest="default_mag", type=float,
                        default=20,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not isinstance(args.inputs, list):
        args.inputs = [args.inputs]

    fn_list = []
    for filename in args.inputs:
        if filename.lower().endswith(".txt"):
            with open(filename, mode="r") as fp:
                fn_list += [fn.strip(" \n") for fn in fp.readlines()]
        elif (os.path.isdir(filename)
          and not filename.lower().endswith(".zarr")):
            fn_list += [os.path.join(filename, fn)
                        for fn in os.listdir(filename)
                        if fn.lower().endswith(".zarr")]
        elif filename.lower().endswith(".zarr"):
            fn_list.append(filename)

    if args.progress_bar:
        q = tqdm.tqdm(total=len(fn_list))

    for filename in fn_list:
        if args.output_dir is not None:
            output_filename = os.path.join(args.output_dir,
                                           os.path.basename(filename))
        else:
            output_filename = filename

        mask_zarr(filename, output_filename, scaled_mag=args.scaled_mag,
                  default_mag=args.default_mag,
                  data_axes=args.data_axes,
                  array_keys=args.array_keys)

        if args.progress_bar:
            q.set_description("Completed masking file %s" % filename)
            q.update()

    if args.progress_bar:
        q.close()
