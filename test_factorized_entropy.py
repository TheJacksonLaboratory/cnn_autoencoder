import argparse

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from src.models import FactorizedEntropy


def test_factorized_entropy(size=1000, epochs=100):
    """ Test the functionality of the factorized entropy approximator module.
    This model allows to model the cummulative distribution function of a random variable of unknown distribution.
    By approximating this function, it is possible to compute the entropy of the variable and use this metric as loss function. 

    Parameters
    ----------
    size : torch.nn.Module
        The network model in the current state

    """

    # Prepare a mix of two normal distributions as "unknown" actual distribution
    x1 = torch.randn([size//2, 1, 1, 1]) * 2 + 5
    x2 = torch.randn([size//2, 1, 1, 1]) * 3 - 3
    x = torch.cat((x1, x2), dim=0)
    plt.hist(x.squeeze().numpy(), density=True)

    # Use a Factorized entropy model to approximate the real distribution
    fact_entropy = FactorizedEntropy(channels_bn=1, K=4, r=3)

    # Optimize the model parameters through stochastic gradient descent
    optimizer = optim.SGD(fact_entropy.parameters(), lr=1e-3)

    # Generate a grid to show the approximated distribution every 10 epochs
    sample_idx = np.random.permutation(size).reshape(10, -1)
    sampled_space = torch.linspace(x.min()-1, x.max()+1, 100).reshape(-1, 1, 1, 1)
    
    for e in range(epochs):
        mean_loss = 0

        for s_i in sample_idx:
            optimizer.zero_grad()
            p = fact_entropy(x[s_i] + 0.5) - fact_entropy(x[s_i] - 0.5)
            loss = -torch.mean(torch.log2(p + 1e-10))
            
            loss.backward()
            mean_loss += loss.item()

            optimizer.step()

        print('[{}] Mean loss: {}'.format(e, mean_loss / sample_idx.shape[1]))

        if e % 10 == 0:
            # Plot the current approximation of the real distribution
            with torch.no_grad():
                sampled_p = fact_entropy(sampled_space + 0.5) - fact_entropy(sampled_space - 0.5)
                plt.plot(sampled_space.squeeze().numpy(), sampled_p.squeeze().numpy(), 'r--')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test the Factorized prior based on an univariate non-parametric density model')
    parser.add_argument('-s', '--size', dest='size', help='Size of the population drawn from a mix of two normal distributions', default=1000)
    parser.add_argument('-e', '--epochs', dest='epochs', help='Number of epochs to train the factorized entropy model', default=100)
    args = parser.parse_args()

    test_factorized_entropy(**args)