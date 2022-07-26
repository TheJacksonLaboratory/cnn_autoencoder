import argparse
import os
from tqdm import tqdm

from PIL import Image


format_dict = {'JPEG2000': 'jp2', 'JPEG': 'jpeg', 'PNG':'png'}


def convert(src_filename, dst_filename, file_format, **kwargs):
    im = Image.open(src_filename, mode='r')
    im.save(dst_filename, format=file_format, **kwargs)
    im.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert images to a different image format')

    parser.add_argument('-sd', '--src-dir', type=str, dest='src_dir', help='Source directory', default='.')
    parser.add_argument('-dd', '--dst-dir', type=str, dest='dst_dir', help='Destination directory', default='.')
    parser.add_argument('-sf', '--src-format', type=str, dest='src_format', help='Source image format')
    parser.add_argument('-df', '--dst-format', type=str, dest='dst_format', help='Destination image format', choices=list(format_dict.keys()))

    args = parser.parse_args()

    in_filenames = ['.'.join(fn.split('.')[:-1]) for fn in os.listdir(args.src_dir) if fn.lower().endswith(format_dict[args.src_format])]
    quality_opts = {}

    q = tqdm()
    for in_fn in in_filenames:
        for iq in range(0, 101, 10):
            if 'JPEG' in args.dst_format:
                quality_opts = {'quality': iq}
            elif 'PNG' in args.dst_format:
                quality_opts = {'compress_level': 9 - iq // 10, 'optimize': False}
                if iq == 100: break

            out_fn = os.path.join(args.dst_dir, '%s_%03d.%s' % (in_fn, iq, format_dict[args.dst_format]))
            q.set_description('Converting %s to %s' % (in_fn, out_fn))
            convert(os.path.join(args.src_dir, in_fn + '.%s' % (format_dict[args.src_format])), out_fn, args.dst_format, **quality_opts)
            q.update()
    
    q.close()
