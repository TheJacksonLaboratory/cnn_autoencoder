"""
BSD 3-Clause License

Copyright (c) 2020, The Regents of the University of California
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

From the examples module of the LC-Model-Compression package
""" 

import logging
import torch
from torch import nn
from collections import OrderedDict
from scipy.linalg import svd
import numpy as np

from lc.compression_types.low_rank import matrix_to_tensor

from models.tasks._autoencoders import ColorEmbedding, DownsamplingUnit, UpsamplingUnit, ResidualDownsamplingUnit, ResidualUpsamplingUnit


class FullRankException(Exception):
    pass

class RankNotEfficientException(Exception):
    pass


def linear_layer_reparametrizer(sub_module, conv_scheme='scheme_1'):
    logger = logging.getLogger('training_log')
    W = sub_module.weight.data.cpu().numpy()

    init_shape = None
    n,m,d1,d2 = None, None, None, None
    if isinstance(sub_module, (nn.Conv2d, nn.ConvTranspose2d)):
        if conv_scheme == 'scheme_1':
            init_shape = W.shape
            reshaped = W.reshape([init_shape[0], -1])
            W = reshaped
        elif conv_scheme == 'scheme_2':
            [n, m, d1, d2] = W.shape
            swapped = W.swapaxes(1, 2)
            reshaped = swapped.reshape([n * d1, m * d2])
            W = reshaped

    u, s, v = svd(W, full_matrices=False)
    from numpy.linalg import matrix_rank

    r = sub_module.rank_ if hasattr(sub_module, 'rank_') \
        else sub_module.selected_rank_ if hasattr(sub_module, 'selected_rank_') \
        else int(matrix_rank(W))
    logger.debug(r)
    if 0 < r < np.min(W.shape):
        diag = np.diag(s[:r] ** 0.5)
        U = u[:, :r] @ diag
        V = diag @ v[:r, :]
        logger.debug(U.shape, V.shape)
        new_W = U @ V


        from numpy.linalg import norm
        logger.debug(f'Fro. norm of W: {norm(W)}')
        logger.debug(f'Fro. norm of diff: {norm(W - new_W)}')


        m,n = W.shape
        if r > np.floor(m*n/(m+n)):
            # raise RankNotEfficientException("Selected rank doesn't contribute to any savings")
            logger.debug("Selected rank doesn't contribute to any savings")
            return sub_module, None

        bias = sub_module.bias is not None
        if isinstance(sub_module, nn.Linear):
            l1 = nn.Linear(in_features=sub_module.in_features, out_features=r, bias=False)
            l2 = nn.Linear(in_features=r, out_features=sub_module.out_features, bias=bias)
            l1.weight.data = torch.from_numpy(V)
            l2.weight.data = torch.from_numpy(U)
            if bias:
                l2.bias.data = sub_module.bias.data
            return l1, l2
        else:
            if conv_scheme == 'scheme_1':
                if isinstance(sub_module, nn.ConvTranspose2d):
                    l1 = nn.ConvTranspose2d(in_channels=sub_module.in_channels,
                                            out_channels=r,
                                            kernel_size=sub_module.kernel_size,
                                            stride=sub_module.stride,
                                            padding=sub_module.padding,
                                            output_padding=sub_module.output_padding,
                                            dilation=sub_module.dilation,
                                            groups=sub_module.groups,
                                            bias=False)
                    l2 = nn.Conv2d(in_channels=r,
                                   out_channels=sub_module.out_channels,
                                   kernel_size=1,
                                   bias=bias)

                else:
                    l1 = nn.Conv2d(in_channels=sub_module.in_channels,
                                   out_channels=r,
                                   kernel_size=sub_module.kernel_size,
                                   stride=sub_module.stride,
                                   padding=sub_module.padding,
                                   dilation=sub_module.dilation,
                                   groups=sub_module.groups,
                                   bias=False)
                    l2 = nn.Conv2d(in_channels=r,
                                   out_channels=sub_module.out_channels,
                                   kernel_size=1,
                                   bias=bias)

                l1.weight.data = torch.from_numpy(V.reshape([-1, *init_shape[1:]]))
                l2.weight.data = torch.from_numpy(U[:, :, None, None])

                if bias:
                    l2.bias.data = sub_module.bias.data

                return l1, l2

            elif conv_scheme == 'scheme_2':
                if isinstance(sub_module, nn.ConvTranspose2d):
                    l1 = nn.ConvTranspose2d(in_channels=sub_module.in_channels,
                                   out_channels=r,
                                   kernel_size=(1, sub_module.kernel_size[1]),
                                   stride=(1, sub_module.stride[1]),
                                   padding=(0, sub_module.padding[1]),
                                   output_padding=(0, sub_module.output_padding[1]),
                                   dilation=sub_module.dilation,
                                   groups=sub_module.groups,
                                   bias=False)

                    l2 = nn.ConvTranspose2d(in_channels=r, 
                                            out_channels=sub_module.out_channels,
                                            kernel_size=(sub_module.kernel_size[0], 1),
                                            padding=(sub_module.padding[0], 0),
                                            output_padding=(sub_module.output_padding[0], 0),
                                            stride=(sub_module.stride[0], 1),
                                            bias=bias)
                else:
                    l1 = nn.Conv2d(in_channels=sub_module.in_channels,
                                   out_channels=r,
                                   kernel_size=(1, sub_module.kernel_size[1]),
                                   stride=(1, sub_module.stride[1]),
                                   padding=(0, sub_module.padding[1]),
                                   dilation=sub_module.dilation,
                                   groups=sub_module.groups,
                                   bias=False)

                    l2 = nn.Conv2d(in_channels=r, 
                                   out_channels=sub_module.out_channels,
                                   kernel_size=(sub_module.kernel_size[0], 1),
                                   padding=(sub_module.padding[0], 0),
                                   stride=(sub_module.stride[0], 1),
                                   bias=bias)

                l1.weight.data = matrix_to_tensor(torch.from_numpy(V).to(l1.weight.data.device), l1.weight.data.shape, conv_scheme)
                l2.weight.data = matrix_to_tensor(torch.from_numpy(U).to(l1.weight.data.device), l2.weight.data.shape, conv_scheme)

                if bias:
                    l2.bias.data = sub_module.bias.data

                return l1, l2

    else:
        return sub_module, None


def reparametrization_helper(sub_module_name, sub_module, conv_scheme, old_weight_decay=True):
    new_sequence = []
    decayed_values_repar = []
    decayed_valued_old = []

    if isinstance(sub_module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        l1, l2 = linear_layer_reparametrizer(sub_module, conv_scheme=conv_scheme)
        if l2 is None:
            new_sequence.append((sub_module_name, sub_module))
            decayed_valued_old.append(sub_module.weight)
        else:            
            new_sequence.append((sub_module_name + '_V', l1))
            new_sequence.append((sub_module_name + '_U', l2))
            decayed_values_repar.append((l1, l2))

    elif isinstance(sub_module, nn.Sequential):
        for m_name, m in sub_module._modules.items():
            m_decayed_values_repar, m_decayed_valued_old, reparametrized = \
                reparametrization_helper(m_name, m, conv_scheme, old_weight_decay=old_weight_decay)
            decayed_values_repar += m_decayed_values_repar
            decayed_valued_old += m_decayed_valued_old
            new_sequence += reparametrized
    else:
        new_sequence.append((sub_module_name, sub_module))
        if old_weight_decay and hasattr(sub_module, 'weight'):
            decayed_valued_old.append(sub_module.weight)

    return decayed_values_repar, decayed_valued_old, new_sequence


def reparametrize_blocks(block_name, block, conv_scheme='scheme_1', old_weight_decay=True):
    all_decayed_values_repar = []
    all_decayed_valued_old = []

    if isinstance(block, ColorEmbedding):
        decayed_values_repar, decayed_valued_old, reparametrized = \
            reparametrization_helper('embedding', block.embedding, conv_scheme, old_weight_decay=old_weight_decay)
        block.embedding = nn.Sequential(OrderedDict(reparametrized))

        all_decayed_values_repar += decayed_values_repar
        all_decayed_valued_old += decayed_valued_old

    elif isinstance(block, (DownsamplingUnit, UpsamplingUnit)):
        decayed_values_repar, decayed_valued_old, reparametrized = \
            reparametrization_helper('model', block.model, conv_scheme, old_weight_decay=old_weight_decay)
        block.model = nn.Sequential(OrderedDict(reparametrized))

        all_decayed_values_repar += decayed_values_repar
        all_decayed_valued_old += decayed_valued_old

    elif isinstance(block, (ResidualDownsamplingUnit, ResidualUpsamplingUnit)):
        decayed_values_repar, decayed_valued_old, reparametrized = \
            reparametrization_helper('res_model', block.res_model, conv_scheme, old_weight_decay=old_weight_decay)
        block.res_model = nn.Sequential(OrderedDict(reparametrized))

        all_decayed_values_repar += decayed_values_repar
        all_decayed_valued_old += decayed_valued_old

        decayed_values_repar, decayed_valued_old, reparametrized = \
            reparametrization_helper('model', block.model, conv_scheme, old_weight_decay=old_weight_decay)
        block.model = nn.Sequential(OrderedDict(reparametrized))

        all_decayed_values_repar += decayed_values_repar
        all_decayed_valued_old += decayed_valued_old

    elif isinstance(block, (nn.Sequential, nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        decayed_values_repar, decayed_valued_old, reparametrized = \
            reparametrization_helper(block_name, block, conv_scheme, old_weight_decay=old_weight_decay)
        all_decayed_values_repar += decayed_values_repar
        all_decayed_valued_old += decayed_valued_old
        block = nn.Sequential(OrderedDict(reparametrized))

    return block, all_decayed_values_repar, all_decayed_valued_old


def reparametrize_low_rank(model, conv_scheme='scheme_1', old_weight_decay=True):
    all_decayed_values_repar = []
    all_decayed_valued_old = []

    # Reparametrize the CAE model blocks
    (model.module.embedding,
     block_decayed_values_repar,
     block_decayed_valued_old) = reparametrize_blocks('color_embedding',
                                                      model.module.embedding,
                                                      conv_scheme,
                                                      old_weight_decay)
    all_decayed_values_repar += block_decayed_values_repar
    all_decayed_values_repar += block_decayed_valued_old

    for block_name, block in model.module.analysis.analysis_track._modules.items():
        (block,
         block_decayed_values_repar,
         block_decayed_valued_old) = reparametrize_blocks(block_name,
                                                          block,
                                                          conv_scheme,
                                                          old_weight_decay)
        model.module.analysis.analysis_track._modules[block_name] = block
        all_decayed_values_repar += block_decayed_values_repar
        all_decayed_values_repar += block_decayed_valued_old

    for block_name, block in model.module.synthesis.synthesis_track._modules.items():
        (block,
         block_decayed_values_repar,
         block_decayed_valued_old) = reparametrize_blocks(block_name,
                                                          block,
                                                          conv_scheme,
                                                          old_weight_decay)
        model.module.synthesis.synthesis_track._modules[block_name] = block
        all_decayed_values_repar += block_decayed_values_repar
        all_decayed_values_repar += block_decayed_valued_old

    for b_i, block in enumerate(model.module.synthesis.color_layers):
        (block,
         block_decayed_values_repar,
         block_decayed_valued_old) = reparametrize_blocks(str(b_i),
                                                          block,
                                                          conv_scheme,
                                                          old_weight_decay)
        model.module.synthesis.color_layers[b_i] = block
        all_decayed_values_repar += block_decayed_values_repar
        all_decayed_values_repar += block_decayed_valued_old

    def weight_decay():
        sum_ = torch.autograd.Variable(torch.FloatTensor([0.0], device='cuda' if torch.cuda.is_available() else 'cpu'))

        for x in all_decayed_valued_old:
            sum_ += torch.sum(x**2)

        for v, u in all_decayed_values_repar:
            v = v.weight
            u = u.weight
            u_ = u.view(u.size()[0], -1)
            v_ = v.view(u_.size()[1], -1)
            sum_ += torch.sum(torch.matmul(u_,v_)**2)

        return sum_

    model.weight_decay = weight_decay
