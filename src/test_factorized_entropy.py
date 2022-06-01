import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from models import FactorizedEntropy, FactorizedEntropyLaplace

model_types = {
    'FactorizedEntropy': FactorizedEntropy,
    'Laplacian': FactorizedEntropyLaplace    
}


def test_factorized_entropy(size=1000, epochs=100, batch=10, channels=1, modes=2):
    """ Test the functionality of the factorized entropy approximator module.
    This model allows to model the cummulative distribution function of a random variable of unknown distribution.
    By approximating this function, it is possible to compute the entropy of the variable and use this metric as loss function. 

    Parameters
    ----------
    size : torch.nn.Module
        The network model in the current state

    """

    # Prepare a mix of two normal distributions as "unknown" actual distribution
    x = []
    for c in range(channels):
        mix_size = np.array([0] + list(range(size//modes, size, size//modes)))
        mix_size[-1] = size
        if modes > 1:
            mix_size = mix_size[1:] - mix_size[:-1]

        x_new = []
        for s in mix_size:
            var, mean = torch.rand(1) * 10, torch.rand(1) * 20
            x_new.append(torch.randn([s, 1, 1, 1]) * var + mean)
        x_new = torch.cat(x_new, dim=0)
        x.append(x_new)
    
    x = torch.cat(x, dim=1)
    x.clip_(0, 255)
    x = x - x.mean(dim=0).unsqueeze(dim=0)

    print('Sample stats\n Mean:', x.mean(dim=0), '\n Var:', x.var(dim=0), '\n Range: [', x.min(), x.max(), ']')
    x = x.round()

    # Use a Factorized entropy model to approximate the real distribution
    fact_entropy = model_types[args.model_type](channels_bn=channels, K=4, r=3)
    print(fact_entropy)
    for par in fact_entropy.parameters():
        print('\t', par.size())

    # Optimize the model parameters through stochastic gradient descent
    optimizer = optim.SGD(params=fact_entropy.parameters(), lr=1e-3)

    # Generate a grid to show the approximated distribution every 10 epochs
    sample_idx = np.tile(np.random.permutation(size).reshape(-1, batch),(epochs, 1))

    sampled_space = torch.linspace(-127.5, 127.5, 512).reshape(-1, 1, 1, 1).tile([1, channels, 1, 1])

    mean_loss = 0

    for s, s_i in enumerate(sample_idx):

        optimizer.zero_grad()
        
        fact_entropy.reset(x[s_i])

        p = fact_entropy(x[s_i] + 0.5) - fact_entropy(x[s_i] - 0.5) + 1e-10        

        loss = torch.mean(torch.sum(-torch.log2(p), dim=1))

        loss.backward()
        mean_loss += loss.item()

        optimizer.step()

        if s % (size // batch) == 0:
            # Plot the current approximation of the real distribution
            print('[{}] Mean loss: {}, P [{}, {}]'.format(s, batch * mean_loss / size, p.min(), p.max()))
            mean_loss = 0

    cols = int(np.ceil(np.sqrt(channels)))
    rows = int(np.ceil(channels // cols))
    print('Ploting results [%d, %d]' % (rows, cols))
    fig, ax = plt.subplots(cols, rows)
    ax = ax.reshape(rows, cols)
    with torch.no_grad():
        sampled_p = fact_entropy(sampled_space + 0.5) - fact_entropy(sampled_space - 0.5)
        for c in range(channels):
            rr = c // cols
            cc = c % cols
            ax[rr, cc].hist(x[:, c].squeeze().numpy(), density=True)
            ax[rr, cc].plot(sampled_space[:, c].squeeze().numpy(), sampled_p[:, c].squeeze().numpy(), 'r--')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test the Factorized prior based on an univariate non-parametric density model')
    parser.add_argument('-s', '--size', type=int, dest='size', help='Size of the population drawn from a mix of two normal distributions', default=1000)
    parser.add_argument('-e', '--epochs', type=int, dest='epochs', help='Number of epochs to train the factorized entropy model', default=100)
    parser.add_argument('-b', '--batch', type=int, dest='batch', help='Batch size', default=10)
    parser.add_argument('-c', '--channels', type=int, dest='channels', help='Number of channels for the compressed representation', default=1)
    parser.add_argument('-m', '--modes', type=int, dest='modes', help='Number of modes for the mix of gaussian distributions for each channel', default=2)
    parser.add_argument('-mt', '--model-type', type=str, dest='model_type', help='Model to approximate the entropy model of the latent layer', choices=model_types.keys(), default=model_types.keys()[0])

    args = parser.parse_args()

    test_factorized_entropy(size=args.size, epochs=args.epochs, batch=args.batch, channels=args.channels, modes=args.modes)