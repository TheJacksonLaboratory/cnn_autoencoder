import argparse
import os

import torch


def save_training_stats(trained_model, output_filename):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save the factorized entropy model learned during a cnn autoencoder training')

    parser.add_argument('-m', '--model', type=str, dest='trained_model', help='The checkpoint of the model to be tested')
    parser.add_argument('-o', '--output', type=str, dest='output', help='The output filename to store the cdf', default='cdf.pth')

    args = parser.parse_args()
    args.mode = 'save_cdf'

    save_training_stats(args.trained_model, args.output)
