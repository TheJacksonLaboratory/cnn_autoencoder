import os
import zarr
import numpy as np
from PIL import Image

from tqdm import tqdm
import argparse


def zarr2png(in_fn, src_dir, out_dir, seed, group):
    z = zarr.open(os.path.join(src_dir, in_fn), "r")
    seg = z[group][0, 0, 0] * 255.0
    seg = seg.astype(np.uint8)
    fn = in_fn.split(".zarr")[0]

    im = Image.fromarray(seg)
    im.save(os.path.join(out_dir, fn + "_%s.png" % seed),
            quality_opts={'compress_level': 5, 'optimize': False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert images from zarr to png format")
    parser.add_argument('-d', '--src-dir', dest='src_dir', type=str,
                        help='Source directory')
    parser.add_argument('-o', '--out-dir', dest='out_dir', type=str,
                        help='Output directory')
    parser.add_argument('-s', '--seed', dest='seeds', nargs="+", type=str,
                        help='Seed used to train the segmentation model')

    args = parser.parse_args()

    for seed in args.seeds:
        in_fns = list(filter(lambda fn: '.zarr' in fn, os.listdir(args.src_dir)))

        out_dir = os.path.join(args.out_dir, seed)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        q = tqdm(total=len(in_fns))
        for in_fn in in_fns:
            q.set_description("Converting %s -> %s" % (in_fn, out_dir))
            group = "labels/segmentation_%s/0/0" % seed
            zarr2png(in_fn, src_dir=args.src_dir, out_dir=out_dir, group=group, 
                     seed=seed)
            q.update()

        q.close()
