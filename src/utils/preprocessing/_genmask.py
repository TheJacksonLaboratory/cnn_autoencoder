import argparse
import os
import ome_types

import math
import dask
import dask.array as da

import numpy as np
from skimage import filters, morphology
from dask_image import ndmeasure, ndfilters

from numcodecs import Blosc


def _read_magnification(filename, default_mag=20.):
    """Read the magnification scale from the metadata
    """
    xml_file_name = os.path.join(filename, 'OME/METADATA.ome.xml')

    if not os.path.exists(xml_file_name):
        # If the metadata is not present in the zarr file (e.g. it was not
        # generated using bioformats2raw), use the default value.
        nom_mag = default_mag
    else:
        ome_metadata = ome_types.from_xml(xml_file_name, parser='lxml',
                                          validate=False)
        objectives = ome_metadata.instruments[0].objectives[0]

        try:
            nom_mag = objectives.nominal_magnification
        except AttributeError:
            nom_mag = default_mag

    return nom_mag


def _clean_mask(labeled_mask):
    """Remove small objects and holes from the labeled mask.
    """
    mask = morphology.remove_small_objects(labeled_mask, min_size=16 * 16,
                                           connectivity=2)
    mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)
    mask = mask.astype(bool)
    mask = morphology.binary_dilation(mask, morphology.disk(16))

    return mask


def generate_mask(zarr_fn, data_group='0/0', data_axes='XYZCT',
                  dest_group='labels',
                  org_mag=None,
                  scale_mag=1.25):
    """Generates a foreground mask from the array stored in zarr_fn/data_group
    using the thresholdig method passed as argument.

    The generated mask is stored inside the same zarr file in the followeing
    data_group `zarr_fn`/`labels`/mask/`data_group`.

    Parameters:
    -----------
    zarr_fn: str
        The file name of an image stored as zarr file.
    data_group: str
        The data_group in the zarr file that will be masked.
    data_axes: str
        The order of the array axes stored in `data_group`.
    dest_group: str
        Destination data_group where to store the mask. The deault is `labels`
        data_group, so the mask will be stored in
        `zarr_fn`/`labels`/mask/`data_group`
    org_mag: float or None
        The original magnification scale of the image at group `data_group`.
        If `None`, the code will try to get it from the metadata, otherwise, it
        will be set to 20.0.
    scale_mag: float
        The input image is downsampled to that magnification level to be
        processed instead of using the original size.
    Returns:
    --------
    mask_comp: str
        The data_group/component where the mask has been saved.
    """
    if org_mag is None:
        org_mag = _read_magnification(zarr_fn)

    arr = da.from_zarr(zarr_fn, component=data_group)

    # Extract the color components according to the axes ordering of this array
    color_axis = data_axes.index('C')
    ch_R, ch_G, ch_B = [tuple(
        [slice(None)] * color_axis
        + [slice(c, c+1, 1)]
        + [slice(None)] * (len(data_axes) - color_axis - 1))
                        for c in range(3)]
    gray_arr = 0.299 * arr[ch_R] + 0.587 * arr[ch_G] + 0.114 * arr[ch_B]

    # Transpose the gray-level image to have its axes in the `YXC` order
    unused_axis = list(set(data_axes) - set('YXC'))
    transpose_order = [data_axes.index(a) for a in unused_axis]
    transpose_order += [data_axes.index(a) for a in 'YXC']
    gray_arr = np.transpose(gray_arr, transpose_order)
    gray_arr = np.squeeze(gray_arr)

    # Resize the input image.
    sample_size = int(math.ceil(org_mag / scale_mag))
    dwn_sigma = sample_size / math.sqrt(2*math.log(2))

    H, W = gray_arr.shape
    sigma_overlap = int(math.ceil(4 * dwn_sigma)) + 1

    H_out = max(1, int(math.ceil(H / sample_size))) * sample_size
    W_out = max(1, int(math.ceil(W / sample_size))) * sample_size
    H_out_overlap = max(1, int(math.ceil(sigma_overlap / sample_size))) * sample_size
    W_out_overlap = max(1, int(math.ceil(sigma_overlap / sample_size))) * sample_size

    H_pad = max(H_out, H_out_overlap) - H
    W_pad = max(W_out, W_out_overlap) - W

    gray_arr = da.pad(gray_arr, pad_width=((0, H_pad), (0, W_pad)))

    print(gray_arr)
    
    gray_arr = ndfilters.gaussian_filter(gray_arr, order=0, sigma=dwn_sigma,
                                         mode='reflect')
    gray_arr = gray_arr.astype(np.uint8)
    gray_arr = da.coarsen(np.mean, gray_arr, {0: sample_size, 1: sample_size})

    # Compute a threshold level to segment foreground from background
    gray_arr_hist = ndmeasure.histogram(gray_arr, min=np.min(gray_arr),
                                        max=np.max(gray_arr),
                                        bins=256)
    thresh_level = dask.delayed(filters.threshold_otsu)(gray_arr,
                                                        hist=gray_arr_hist)
    mask = gray_arr <= thresh_level.compute()
    mask = mask.astype(np.uint8)

    # Remove small objects and holes
    mask, _ = ndmeasure.label(mask)

    mask = da.from_delayed(dask.delayed(_clean_mask)(mask), shape=mask.shape,
                           meta=np.empty((), dtype=bool))
    mask = mask[:H_out, :W_out]

    # Return the axes order of the mask to the order given by `data_axes`.
    rem_axes = list(set(data_axes) - set('YX'))
    if len(rem_axes):
        mask = da.from_delayed(dask.delayed(np.expand_dims)(
            mask, axis=tuple(range(len(rem_axes)))),
            shape=[1] * len(rem_axes) + list(mask.shape),
            meta=np.empty((), dtype=bool))

    transpose_order = [(''.join(rem_axes) + 'YXC').index(a)
                       for a in data_axes]
    mask = da.transpose(mask, axes=transpose_order)

    # Store the generated mask
    compressor = Blosc(cname='lz4', clevel=9, shuffle=Blosc.SHUFFLE)
    mask_comp = dest_group + '/mask/' + data_group
    mask.to_zarr(zarr_fn, component=mask_comp, overwrite=True,
                 compressor=compressor)

    return mask_comp


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Mask generation for zarr images")
    parser.add_argument("-i", "--input", type=str, dest="input_filaneme",
                        help="Path to input zarr file")
    parser.add_argument("-dg", "--data-group", type=str, dest="data_group",
                        help="The zarr data_group used to generate the mask")
    parser.add_argument("-da", "--data-axes", type=str, dest="data_axes",
                        help="Order of the axes in the group array (default is "
                             "the biofromats2raw output `XYZCT`)",
                        default="XYZCT")
    parser.add_argument("-dst", "--dest-group", type=str, dest="dest_group",
                        help="Group inside the zarr file where to store the "
                             "generated mask",
                        default="labels")
    args = parser.parse_args()

    mask_comp = generate_mask(args.input_filaneme, data_group=args.data_group,
                              data_axes=args.data_axes,
                              dest_group=args.dest_group)
