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


from tqdm import tqdm
import torch


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


def compute_mse_rate_loss(forward_func, data_loader):
    total_cnt = 0
    ave_loss = 0
    ave_mse = 0
    ave_rate = 0

    q_bar = tqdm(desc='Evaluating performance', total=len(data_loader), position=1, leave=None)

    for x, _ in data_loader:
        with torch.no_grad():
            mse, rate, loss = forward_func(x)
            ave_loss += loss.data.item() * x.size(0)
            ave_mse += mse.data.item() * x.size(0)
            ave_rate += rate.data.item() * x.size(0)
            total_cnt += x.size(0)

            q_bar.update()

    q_bar.close()

    ave_loss /= total_cnt
    ave_mse /= total_cnt
    ave_rate /= total_cnt

    return ave_mse, ave_rate, ave_loss
