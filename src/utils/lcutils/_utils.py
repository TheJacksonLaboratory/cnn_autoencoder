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
import torch
from torch import nn
from lc.torch import ParameterTorch as LCParameterTorch, AsIs
from lc.compression_types.low_rank import RankSelection

from ._compflops import add_flops_counting_methods
from ._finetune import reparametrize_low_rank

from tqdm import tqdm


def format_time(seconds):
    if seconds < 60:
        return '{:.1f}s.'.format(seconds)
    if seconds < 3600:
        return '{:d}m. {}'.format(int(seconds//60), format_time(seconds%60))
    if seconds < 3600*24:
        return '{:d}h. {}'.format(int(seconds//3600), format_time(seconds%3600))

    return '{:d}d. {}'.format(int(seconds//(3600*24)), format_time(seconds%(3600*24)))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Recorder():
    def __init__(self):
        pass

    def record(self, tag, value):
        if hasattr(self, tag):
            self.__dict__[tag].append(value)
        else:
            self.__dict__[tag] = [value]


def compute_mse_rate_loss(forward_func, data_loader, print_log=False):
    total_cnt = 0
    ave_loss = 0
    ave_mse = 0
    ave_rate = 0

    if print_log:
        q_bar = tqdm(desc='Evaluating performance', total=len(data_loader), position=1, leave=None)

    for x, _ in data_loader:
        with torch.no_grad():
            mse, rate, loss = forward_func(x)
            ave_loss += loss.data.item() * x.size(0)
            ave_mse += mse.data.item() * x.size(0)
            ave_rate += rate.data.item() * x.size(0)
            total_cnt += x.size(0)

            if print_log:
                q_bar.update()

    if print_log:
        q_bar.close()

    ave_loss /= total_cnt
    ave_mse /= total_cnt
    ave_rate /= total_cnt

    return ave_mse, ave_rate, ave_loss


def create_lc_compression_task(config_, model=None, device='cpu',
                               eval_flops=False,
                               val_loader=None):
    if eval_flops and config_['criterion'] == "flops":
        model = add_flops_counting_methods(model)
        model.start_flops_count()

        for x, _ in val_loader:
            _ = model(x)
            break
        uncompressed_flops = model.compute_average_flops_cost()
        print('The number of FLOPS in model', uncompressed_flops)
        model.stop_flops_count()

    compression_tasks = {}
    for i, (w, module) in enumerate(
        [((lambda x=x: getattr(x, 'weight')), x)
         for x in model.modules() if isinstance(x, (nn.Conv2d,
                                                    nn.ConvTranspose2d,
                                                    nn.Linear))]):
        compression_tasks[LCParameterTorch(w, device)] = \
            (AsIs,
             RankSelection(conv_scheme=config_['conv_scheme'],
                           alpha=config_["alpha"],
                           criterion=config_["criterion"],
                           normalize=True,
                           module=module),
             f"task_{i}")

    return compression_tasks


def load_compressed_dict(model_base, checkpoint, compression_tasks, 
                         conv_scheme='scheme_2'):
    compression_tasks_state = checkpoint['best_compression_tasks_state']
    compression_tasks_info = checkpoint['best_compression_tasks_info']
    compression_tasks_lambdas = checkpoint['best_compression_tasks_lambdas']

    last_lc_it = checkpoint['last_step_number']

    for task_name, infos in compression_tasks_info.items():
        compression_tasks_info[task_name] = infos[last_lc_it]

    for i, module in enumerate([x for x in model_base.modules() if isinstance(x, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))]):
        module.selected_rank_ = compression_tasks_info[f"task_{i}"]['selected_rank']
        print(module.selected_rank_)

    for param in compression_tasks.keys():
        (view, compression, task_name) = compression_tasks[param]
        compression_state = compression_tasks_state[task_name]
        if "init_shape" not in compression_state:
            # we need to manually set init_shape since old version didn't save it
            x=param.vector_to_compression_view(param.w, view)
            compression_state["init_shape"] = x.shape
        compression.load_state_dict(compression_state)
        compression.info = compression_tasks_info[task_name]

        if isinstance(param, LCParameterTorch):
            param.retrieve(full=True)
            param.lambda_ = compression_tasks_lambdas[task_name]
            param.delta_theta = param.compression_view_to_vector(compression.uncompress_state(), view)
            print(param.delta_theta, compression, task_name)
        else:
            raise Exception("RankSelection contains non-torch LCparameters, critical error")

    reparametrize_low_rank(model_base, conv_scheme=conv_scheme)