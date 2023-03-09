import argparse
import os

import torch

from itertools import product

stat_keys = ['avg', 'std']


def dump_metrics(metrics_filename, out_filename, mode='w'):
    basename, extension = os.path.splitext(out_filename)
    out_filename_per_img = basename + '_per_img' + extension

    out_file_exists = os.path.isfile(out_filename)
    metrics = torch.load(metrics_filename)
    metric_keys = list(metrics.keys())
    metric_keys.remove('codec')

    codec = metrics['codec']
    if codec == 'CAE':
        identifier = metrics['seed']
        metric_keys.remove('seed')
    else:
        identifier = metrics['quality']
        metric_keys.remove('quality')

    metric_keys = list(filter(lambda fn: 'stats' not in fn, metric_keys))

    with open(out_filename, mode) as o_f, \
         open(out_filename_per_img, mode) as o_i_f:
        if 'w' in mode or ('a' in mode and not out_file_exists):
            o_f.write('Method,quality')
            for k, s in product(metric_keys, stat_keys):
                o_f.write(',%s_%s' % (k, s))
            o_f.write('\n')
        if 'w' in mode or ('a' in mode and not out_file_exists):
            o_i_f.write('Method,quality,id')
            for k in metric_keys:
                o_i_f.write(',%s' % k)
            o_i_f.write('\n')

        # Insert overall metrics
        o_f.write('%s,%s' % (codec, identifier))
        for k, s in product(metric_keys, stat_keys):
            o_f.write(',%f' % metrics['%s_stats' % k][s])

        o_f.write('\n')

        # Insert per-image metrics
        for i in range(len(metrics[metric_keys[0]])):
            o_i_f.write('%s,%s,%i' % (codec, identifier, i))
            for k in metric_keys:
                o_i_f.write(',%f' % metrics[k][i])
            o_i_f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dump metrics')

    parser.add_argument('-d', '--dir', type=str, dest='root_dir', help='Folder where the metric outputs are stored')
    parser.add_argument('-m', '--method', type=str, dest='method', help='The method used to compute the metrics (JPEG, PNG)')
    parser.add_argument('-o', '--out-file', type=str, dest='out_filename', help='Filename where to store the dumped metrics')
    parser.add_argument('-fm', '--file-mode', type=str, dest='file_mode', help='Mode to open the output file', default='w')

    args = parser.parse_args()

    mode = args.file_mode
    for fn in os.listdir(args.root_dir):
        metric_filename = os.path.join(args.root_dir, fn)
        if not '_%s_' % args.method in fn or not fn.endswith('.pth'):
            continue
        print('Opening file', metric_filename)
        dump_metrics(metric_filename, args.out_filename, mode)
        mode = 'a'
