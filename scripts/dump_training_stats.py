import argparse
import os
import re

import torch
import numpy as np


def save_training_stats_checkpoint(trained_model, output_filename):
    """Extract and save the training and validation performance into a csv file
    """
    save_fn = os.path.join(trained_model)

    if not torch.cuda.is_available():
        state = torch.load(save_fn, map_location=torch.device('cpu'))
    else:
        state = torch.load(save_fn)

    if not output_filename.endswith(".csv"):
        output_filename = os.path.join(output_filename, "training_stats.csv")

    with open(output_filename, "w") as fp:
        fp.write("Model,seed,step,train_loss,val_loss\n")
        for s, (trn_loss, val_loss) in enumerate(zip(state["train_loss"],
                                                     state["valid_loss"])):
            fp.write("%s,%s,%i,%f,%f\n" % (state["args"]["model_type"],
                                           state["args"]["seed"],
                                           s * state["args"]["checkpoint_steps"],
                                           trn_loss,
                                           val_loss))


def save_training_stats_log(training_log, output_filename):
    """Extract and save the training and validation performance into a csv file
    """
    if not output_filename.lower().endswith(".csv"):
        output_filename = os.path.join(output_filename, "training_stats.csv")

    metrics_headers = 0
    last_step = 0
    stats = {'training': {},
             'validation': {}
             }

    with open(training_log, "r") as i_fp, open(output_filename, "w") as o_fp:
        o_fp.write("seed,mode,step,")

        log_entry = i_fp.readline()
        seed = int(log_entry[log_entry.find('seed') + len('seed: '):])

        for log_entry in i_fp:
            mode = ''

            if 'Training Loss' in log_entry:
                mode = 'training'

            if 'Validation Loss' in log_entry:
                mode = 'validation'

            if len(mode):
                metrics = log_entry.strip('\n').split(' ')[8:]
                metrics[0] = 'Loss=' + metrics[0]
                metrics_dict = {}
                for m in metrics:
                    if len(m) == 0:
                        continue

                    m = list(filter(len, re.split('[,=:\[\]]', m)))
                    if m[0] == 'lr':
                        m = m[1:]

                    if len(m) == 2:
                        metrics_dict[m[0]] = m[1]

                    elif len(m) == 3:
                        metrics_dict[m[0] + '_min'] = m[1]
                        metrics_dict[m[0] + '_max'] = m[2]

                    elif len(m) == 4:
                        metrics_dict[m[0] + '_q1'] = m[1]
                        metrics_dict[m[0] + '_q2'] = m[2]
                        metrics_dict[m[0] + '_q3'] = m[3]

                    elif len(m) == 5:
                        metrics_dict[m[0] + '_min'] = m[1]
                        metrics_dict[m[0] + '_max'] = m[2]
                        metrics_dict[m[0] + '_std'] = m[4]

                for m, v in metrics_dict.items():
                    if stats[mode].get(m, None) is None:
                        stats[mode][m] = []

                    stats[mode][m].append(float(v))

                if metrics_headers == 0:
                    o_fp.write(','.join(metrics_dict.keys()) + '\n')
                    metrics_headers = len(metrics_dict.keys())

            if 'Step' in log_entry:
                step_pos = log_entry.find('Step') + len('Step')
                decision_pos = log_entry[step_pos:].find('(')
                last_step = int(log_entry[step_pos:step_pos + decision_pos])

                for md in ['training', 'validation']:
                    metrics = []
                    for k in stats[md]:
                        metrics.append('%0.4f' % np.mean(stats[md][k]))

                    metrics += [''] * (metrics_headers - len(metrics))

                    metrics = ','.join(metrics)
                    o_fp.write('{:d},{},{:d},{}\n'.format(seed, md, last_step,
                                                          metrics))

                    for k in stats[md]:
                        stats[md][k] = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save the stats from the training stage '
                                     'of a cnn autoencoder model')

    parser.add_argument('-m', '--model', type=str, dest='checkpoint',
                        help='The checkpoint of the model to be tested')
    parser.add_argument('-o', '--output', type=str, dest='output',
                        help='The output filename to store the training stats',
                        default='training_stats.csv')

    args = parser.parse_args()
    args.mode = 'save_stats'

    if args.checkpoint.lower().endswith('.pth'):
        save_training_stats_checkpoint(args.checkpoint, args.output)
    else:
        save_training_stats_log(args.checkpoint, args.output)
