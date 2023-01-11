from functools import partial
import models
import utils
from train_cae_ms import setup_network, resume_checkpoint, setup_criteria

import lc
from lc.torch import ParameterTorch as Param, AsVector, AsIs
from lc.compression_types import ConstraintL0Pruning, LowRank, RankSelection, AdaptiveQuantization

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm


def compute_mse_loss(forward_fun, data):
    """Performance evaluation.
    Evaluates the performance of the network in its current state using the
    full set of validation elements.

    Parameters
    ----------
    forward_fun : function
        The forward function applied to the input data to get its reconstructed
        version after compressing it.
    data : torch.utils.data.DataLoader or list[tuple]
        The validation dataset. Because the target is reconstruct the input,
        the label associated is ignored.

    Returns
    -------
    mean_mse : float
        Mean value of the MSE between the input and its reconstruction
    mean_loss : float
        Mean value of the criterion function over the full set of validation
        elements
    """
    sum_loss = 0
    sum_mse = 0
    total_count = 0

    with torch.no_grad():
        q = tqdm(desc='Performance evaluation', total=len(data))
        for i, (x, _) in enumerate(data):
            mse, loss = forward_fun(x)

            # MSE and Loss values come averaged from the forward pass, so these
            # are weighted for the final average computation.
            sum_loss += loss * x.size(0)
            sum_mse += mse * x.size(0)
            total_count += x.size(0)

            q.update()

        q.close()
    avg_loss = sum_loss / total_count
    avg_mse = sum_mse / total_count

    return avg_mse, avg_loss


def train_test_eval_base(net, criterion, args):
    train_loader, val_loader = utils.get_data(args)

    def forward_func(x):
        x_r, y, p_y = net(x)
        loss_dict = criterion(x=x, y=y, x_r=x_r, p_y=p_y, net=net)

        loss = torch.mean(loss_dict['dist_rate_loss'])
        mse = loss_dict['dist'][0]

        return mse, loss

    mse_train, loss_train = compute_mse_loss(forward_func, train_loader)
    mse_val, loss_val = compute_mse_loss(forward_func, val_loader)

    print(f"Train MSE: {mse_train:.4f}, train loss: {loss_train}")
    print(f"Validation MSE: {mse_val:.2f}, validation loss: {loss_val}")


def load_reference_cae(args):
    net = setup_network(args)
    resume_checkpoint(net, None, None, args.resume, gpu=args.gpu,
                      resume_optimizer=False)
    return net


def my_l_step_base(model, lc_penalty, step, criterion, args):
    train_loader, _ = utils.get_data(args)
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    lr = 0.0001*(0.98**step)
    optimizer = optim.Adam(params, lr=lr)

    print(f'L-step #{step} with lr: {lr:.5f}')
    epochs_per_step_ = 7

    if step == 0:
        epochs_per_step_ = epochs_per_step_ * 2

    for epoch in range(epochs_per_step_):
        avg_loss = []
        q = tqdm(desc='L-step Epoch %i' % epoch, total=len(train_loader))
        for x, _ in train_loader:
            optimizer.zero_grad()
            x = x.to('cuda' if args.gpu else 'cpu')
            x_r, y, p_y = model(x)

            loss_dict = criterion(x=x, y=y, x_r=x_r, p_y=p_y, net=model)
            loss = loss_dict['dist_rate_loss'] + lc_penalty()

            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            q.set_description(f'Loss {loss:0.4f}')
            q.update()

        q.close()
        
        print(f"\tepoch #{epoch} is finished.")
        print(f"\t  avg. train loss: {np.mean(avg_loss):.6f}")


def run_lc_algorithm(args):
    device = torch.device('cuda' if args.gpu else 'cpu')

    mu_s = [9e-5 * (1.1 ** n) for n in range(20)]

    net = load_reference_cae(args)
    criterion, _ = setup_criteria(args)

    untracked_layers = [lambda x=x: getattr(x, 'weight') for x in net.modules() if hasattr(x, 'weight') and not isinstance(x, nn.Conv2d) and not isinstance(x, nn.ConvTranspose2d)]
    untracked_layers += [lambda x=x: getattr(x, 'bias') for x in net.modules() if hasattr(x, 'bias') and x.bias is not None and not isinstance(x, nn.Conv2d) and not isinstance(x, nn.ConvTranspose2d)]
    untracked_pars = sum(map(lambda l: l().numel(), untracked_layers))

    layers = [((lambda x=x: getattr(x, 'weight')), x) for x in net.modules() if (isinstance(x, nn.Conv2d) or isinstance(x, nn.ConvTranspose2d)) and (x.weight.ndim == 4 or x.weight.ndim == 2)]
    n_pars = sum(map(lambda l: l[0]().numel(), layers))

    net = utils.add_flops_counting_methods(net)
    net.start_flops_count()
    _ = net(torch.rand(1, 3, 128, 128))
    compressed_flops = net.compute_average_flops_cost()
    net.stop_flops_count()

    compression_tasks = {}
    for i, (w, module) in enumerate(layers):
        compression_tasks[Param(w, device)] = (AsIs,
                                               RankSelection(
                                                   conv_scheme='scheme_1',
                                                   alpha=10e-10,
                                                   criterion='flopgs',
                                                   normalize=True,
                                                   module=module),
                                               f"task_{i}")

    my_l_step = partial(my_l_step_base, criterion=criterion, args=args)
    train_test_eval = partial(train_test_eval_base, criterion=criterion, args=args)

    lc_alg = lc.Algorithm(
        model=net,                            # model to compress
        compression_tasks=compression_tasks,  # specifications of compression
        l_step_optimization=my_l_step,        # implementation of L-step
        mu_schedule=mu_s,                     # schedule of mu values
        evaluation_func=train_test_eval       # evaluation function
    )
    lc_alg.run()
    
    print('=' * 100)
    print('Parameters count:', lc_alg.count_params())
    compressed_model_bits = lc_alg.count_param_bits() + untracked_pars * 32
    uncompressed_model_bits = (n_pars + untracked_pars) * 32
    compression_ratio = uncompressed_model_bits / compressed_model_bits
    print('Compression ratio achievied:', compression_ratio)


if __name__ == "__main__":
    args = utils.get_args(task='autoencoder', mode='training')

    utils.setup_logger(args)

    run_lc_algorithm(args)
