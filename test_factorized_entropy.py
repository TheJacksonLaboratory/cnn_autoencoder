import numpy as np
import torch
from .src.models import FactorizedEntropy


def test_factorized_entropy(size=1000):
    import torch.optim as optim
    import matplotlib.pyplot as plt

    print('Test the Factorized prior based on a univariate non-parametric density model')    
    x1 = torch.randn([size//2, 1, 1, 1]) * 2 + 5
    x2 = torch.randn([size//2, 1, 1, 1]) * 3 - 3
    x = torch.cat((x1, x2), dim=0)
    plt.hist(x.squeeze().numpy(), density=True)

    fact_entropy = FactorizedEntropy(channels=1, K=4, r=3)
    optimizer = optim.SGD(fact_entropy.parameters(), lr=1e-3)

    sample_idx = np.random.permutation(size).reshape(10, -1)
    sampled_space = torch.linspace(x.min()-1, x.max()+1, 100).reshape(-1, 1, 1, 1)
    
    for e in range(100):
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
            with torch.no_grad():
                sampled_p = fact_entropy(sampled_space + 0.5) - fact_entropy(sampled_space - 0.5)
                plt.plot(sampled_space.squeeze().numpy(), sampled_p.squeeze().numpy(), 'r--')

    plt.show()


if __name__ == '__main__':
    test_factorized_entropy()