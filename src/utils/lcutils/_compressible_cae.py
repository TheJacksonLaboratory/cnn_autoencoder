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


import time
import torch
from torch import nn
from lc.torch import ParameterTorch as LCParameterTorch, AsIs
from lc.compression_types.low_rank import RankSelection
from lc.models.torch.utils import count_params
from ._compflops import add_flops_counting_methods
from ._finetune import reparametrize_low_rank
from ._utils import AverageMeter, Recorder, format_time, compute_mse_rate_loss

from tqdm import tqdm

class CompressibleCAE():
    def __init__(self, name, model, train_loader, val_loader, criterion):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.name = name
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion

    def lc_setup(self):
        def l_step_optimization(model, lc_penalty, step, config):
            all_start_time = config['all_start_time']

            lr_scheduler = None
            my_params = filter(lambda p: p.requires_grad, model.parameters())
            learning_rate = config['lr']

            if config['lr_decay_mode'] == 'after_l':
                learning_rate *= (config['lr_decay'] ** step)
                print(f"Current LR={learning_rate}")

            def constract_my_forward_lc_eval(lc_penalty):
                pen = lc_penalty()

                def my_forward_lc_eval(x):
                    x_r, y, p_y = model(x)
                    loss_dict = self.criterion(x=x, y=y, x_r=x_r, p_y=p_y, net=model)
                    mse = loss_dict['dist'][0]
                    rate = loss_dict['rate_loss']
                    loss = torch.mean(loss_dict['dist_rate_loss'])
                    return mse, rate, loss + pen

                return my_forward_lc_eval

            optimizer = torch.optim.Adam(my_params, learning_rate)

            if config['lr_decay_mode'] == 'restart_on_l':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])

            if 'lr_trick' in config:
                l_trick_value = 0.1
                print('LR trick in play. first epoch is trained with LR of {:.4e}'.format(config['lr'] * l_trick_value))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['lr'] * l_trick_value
                # TODO: revert back the lr_trick?

            steps_in_this_it = config['steps'] // config['print_freq'] if step > 0 else \
                config['first_mu_epochs'] if 'first_mu_epochs' in config else config['steps'] // config['print_freq']
            print('Steps in this iteration is :', steps_in_this_it)
            model.eval()

            lc_evaluator = constract_my_forward_lc_eval(lc_penalty)
            ave_mse_train, ave_rate_train, ave_loss = compute_mse_rate_loss(lc_evaluator, self.train_loader)
            print('\ttrain loss: {:.6f}, mse: {:.4f}, rate: {:0.4f}'.format(ave_loss, ave_mse_train, ave_rate_train))
            ave_mse_val, ave_rate_val, ave_loss = compute_mse_rate_loss(lc_evaluator, self.val_loader)
            print('\tval    loss: {:.6f}, mse: {:.4f}, rate: {:0.4f}'.format(ave_loss, ave_mse_val, ave_rate_val))
            model.train()
            epoch_time = AverageMeter()
            rec = Recorder()

            # avg_epoch_losses = []
            s = 0
            checkpoints = 0
            start_time = time.time()
            avg_loss_ = AverageMeter()

            q_bar = tqdm(total=config['steps'], position=0)

            while s < config['steps']:
                s += 1

                for x, _ in self.train_loader:
                    optimizer.zero_grad()
                    x_r, y, p_y = self.model(x)
                    loss_dict = self.criterion(x=x, y=y, x_r=x_r, p_y=p_y, net=self.model)
                    loss = torch.mean(loss_dict['dist_rate_loss'])
                    avg_loss_.update(loss.item())
                    loss.backward()
                    optimizer.step()

                    q_bar.update()

                    if (s % config['print_freq'] == 0
                      or s >= config['steps']):
                        end_time = time.time()
                        training_time = end_time - all_start_time
                        epoch_time.update(end_time - start_time)

                        checkpoints += 1
                        print("LC step {0}, Checkpoint {1} reached in {2.val:.3f}s (avg: {2.avg:.3f}s). Training for {3}"
                                    .format(s, checkpoints, epoch_time, format_time(end_time - all_start_time)))
                        print('AVG train loss {0.avg:.6f}'.format(avg_loss_))
                        rec.record('average_loss_per_epoch', avg_loss_)

                        model.eval()
                        lc_evaluator = constract_my_forward_lc_eval(lc_penalty)

                        ave_mse_train, ave_rate_train, ave_loss = compute_mse_rate_loss(lc_evaluator, self.train_loader)
                        rec.record('train', [ave_loss, ave_mse_train, ave_rate_train, training_time, step + 1, checkpoints])
                        print('\ttrain loss: {:.6f}, mse: {:.4f}, rate: {:.4f}'.format(ave_loss, ave_mse_train, ave_rate_train))

                        ave_mse_val, ave_rate_val, ave_loss = compute_mse_rate_loss(lc_evaluator, self.val_loader)
                        rec.record('val', [ave_loss, ave_mse_val, ave_rate_val, training_time, step + 1, checkpoints])
                        print('\tvalidation loss: {:.6f}, mse: {:.4f}, rate: {:.4f}'.format(ave_loss, ave_mse_val, ave_rate_val))
                        model.train()

                        if config['lr_decay_mode'] == 'restart_on_l':
                            print("\told LR: {:.4e}".format(optimizer.param_groups[0]['lr']))
                            lr_scheduler.step()
                            print("\tnew LR: {:.4e}".format(optimizer.param_groups[0]['lr']))

                        else:
                            print("\tLR: {:.4e}".format(learning_rate))
                        
                        start_time = time.time()
                        avg_loss_ = AverageMeter()
                    
                    if s >= config['steps']:
                        break

            q_bar.close()

            info = {'train': rec.train, 'val': rec.val, 'average_loss_per_train_epoch': rec.average_loss_per_epoch}
            return info

        def evaluation(model):
            def my_forward_eval(x):
                x_r, y, p_y = model(x)
                loss_dict = self.criterion(x=x, y=y, x_r=x_r, p_y=p_y, net=model)
                mse = loss_dict['dist'][0]
                rate = loss_dict['rate_loss']
                loss = torch.mean(loss_dict['dist_rate_loss'])
                return mse, rate, loss

            model.eval()

            ave_mse_train, ave_rate_train, ave_loss_train = compute_mse_rate_loss(my_forward_eval, self.train_loader)
            print('\tnested train loss: {:.6f}, mse: {:.4f}, rate: {:0.4f}'.format(ave_loss_train, ave_mse_train, ave_rate_train))
            ave_mse_val, ave_rate_val, ave_loss_val = compute_mse_rate_loss(my_forward_eval, self.val_loader)
            model.train()
            print('\tnested validation loss: {:.6f}, mse: {:.4f}, rate: {:0.4f}'.format(ave_loss_val, ave_mse_val, ave_rate_val))

            return {
                'nested_train_loss': ave_loss_train,
                'nested_train_mse': ave_mse_train,
                'nested_train_rate': ave_rate_train,
                'nested_val_loss': ave_loss_val,
                'nested_val_mse': ave_mse_val,
                'nested_val_rate': ave_rate_val,
            }

        def create_lc_compression_task(config_):

            if config_['criterion'] == "flops":
                model = add_flops_counting_methods(self.model)
                model.start_flops_count()

                for x, _ in self.val_loader:
                    _ = model(x)
                    break
                uncompressed_flops = model.compute_average_flops_cost()
                print('The number of FLOPS in model', uncompressed_flops)
                model.stop_flops_count()

            compression_tasks = {}
            for i, (w, module) in enumerate([((lambda x=x: getattr(x, 'weight')), x) for x in self.model.modules() if
                                                                             isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear)]):
                compression_tasks[LCParameterTorch(w, self.device)] \
                    = (AsIs,
                         RankSelection(conv_scheme=config_['conv_scheme'], alpha=config_["alpha"], criterion=config_["criterion"],
                                                     normalize=True, module=module), f"task_{i}")

            return compression_tasks

        return l_step_optimization, evaluation, create_lc_compression_task

    def finetune_setup(self, tag_of_lc_model, c_step_config):
        exp_run_details = torch.load(f"results/{self.name}_lc_{tag_of_lc_model}.th", map_location="cpu")
        # despite the 's' at the end, there is only one model state, the last
        model_state_to_load = exp_run_details['model_states']
        last_lc_it = exp_run_details['last_step_number']

        model = add_flops_counting_methods(self.model)
        model.start_flops_count()

        for x, _ in self.val_loader:
            _ = model(x)
            break
        all_flops = model.compute_average_flops_cost()
        model.stop_flops_count()
        self.model = self.model.cpu()
        all_params = count_params(self.model)

        compression_info = {}
        for task_name, infos in exp_run_details['compression_tasks_info'].items():
            compression_info[task_name] = infos[last_lc_it]
        # compression_infos = exp_run_details['compression_tasks_info'][last_lc_it]

        print(model_state_to_load.keys())
        del exp_run_details

        for key in list(model_state_to_load.keys()):
            if key.startswith("lc_param_list"):
                del model_state_to_load[key]

        import gc
        gc.collect()
        print(gc.garbage)

        print(model_state_to_load.keys())

        self.model.load_state_dict(model_state_to_load)
        print("model has been sucessfully loaded")

        for i, module in enumerate(
                [x for x in self.model.modules() if isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear)]):
            module.selected_rank_ = compression_info[f"task_{i}"]['selected_rank']
            print(module.selected_rank_)

        old_weight_decay = True
        if hasattr(model, 'old_weight_decay'):
            old_weight_decay = model.old_weight_decay

        if hasattr(model, 'conv_scheme'):
            conv_scheme = model.conv_scheme
        else:
            conv_scheme = 'scheme_1'

        reparametrize_low_rank(self.model, conv_scheme=conv_scheme, old_weight_decay=old_weight_decay)
        print(self.model)

        self.model = self.model.to(self.device)
        print("Low rank layers of the model has been successfully reparameterized with sequence of full-rank matrices.")

        def my_forward_eval(x):
            x_r, y, p_y = model(x)
            loss_dict = self.criterion(x=x, y=y, x_r=x_r, p_y=p_y, net=model)
            mse = loss_dict['dist'][0]
            rate = loss_dict['rate_loss']
            loss = torch.mean(loss_dict['dist_rate_loss'])
            return mse, rate, loss

        self.model.eval()
        ave_mse_train, ave_rate_train, ave_loss_train = compute_mse_rate_loss(my_forward_eval, self.train_loader)
        print('\tBefore finetuning, the train loss: {:.6f}, mse: {:.4f}, rate: {:.4f}'.format(ave_loss_train, ave_mse_train, ave_rate_train))
        # rec.record('train_nested', [ave_loss, accuracy, training_time, step + 1])
        ave_mse_val, ave_rate_val, ave_loss_val = compute_mse_rate_loss(my_forward_eval, self.val_loader)
        self.model.train()
        print('\tBefore finetuning, the val loss: {:.6f}, mse: {:.4f}, rate: {:.4f}'.format(ave_loss_val, ave_mse_val, ave_rate_val))

        model = add_flops_counting_methods(self.model)
        model.start_flops_count()

        for x, _ in self.val_loader:
            _ = model(x)
            break

        compressed_flops = model.compute_average_flops_cost()
        model.stop_flops_count()
        self.model = self.model.cpu()
        compressed_params = count_params(model)
        self.model = self.model.to(self.device)
        print('The number of FLOPS in original model', all_flops)
        print('The number of params in original model:', all_params)
        print('The number of FLOPS in this model', compressed_flops)
        print('The number of params in this model:', compressed_params)
        flops_rho = all_flops[0] / compressed_flops[0]
        storage_rho = all_params / compressed_params
        print(f'FLOPS rho={flops_rho:.3f}; STORAGE rho={storage_rho:.3f};')

        compression_stats = {
            'original_flops': all_flops,
            'compressed_flops': compressed_flops,
            'flops_rho': flops_rho,
            'storage_rho': storage_rho}

        def finetuning(config):
            all_start_time = time.time()
            my_params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = torch.optim.Adam(my_params, config['lr'])
            epoch_time = AverageMeter()
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])

            train_info = {}
            val_info = {}

            from threading import Thread
            import asyncio

            def start_loop(loop):
                asyncio.set_event_loop(loop)
                loop.run_forever()

            new_loop = asyncio.new_event_loop()
            t = Thread(target=start_loop, args=(new_loop,))
            t.start()

            start_time = time.time()
            avg_loss_ = AverageMeter()

            q_bar = tqdm(total=config['steps'], position=0)

            s = 0
            checkpoints = 0
            while s < config['steps']:
                s += 1

                for x, _ in self.train_loader:
                    optimizer.zero_grad()

                    x_r, y, p_y = self.model(x)
                    loss_dict = self.criterion(x=x, y=y, x_r=x_r, p_y=p_y, net=self.model)
                    loss = torch.mean(loss_dict['dist_rate_loss'])

                    loss.backward()
                    avg_loss_.update(loss.item())
                    optimizer.step()

                    q_bar.update()

                    if (s % config['print_freq'] == 0
                      or s >= config['steps']):
                        end_time = time.time()
                        training_time = end_time - all_start_time
                        epoch_time.update(end_time - start_time)

                        checkpoints += 1
                        print("Checkpoint {} reached in {1.val:.3f}s (avg: {1.avg:.3f}s). Training for {2}"
                                    .format(checkpoints, epoch_time, format_time(end_time - all_start_time)))
                        print('AVG train loss {0.avg:.6f}'.format(avg_loss_))

                        print("\tLR: {:.4e}".format(lr_scheduler.get_lr()[0]))
                        lr_scheduler.step()

                        self.model.eval()
                        ave_mse_train, ave_rate_train, ave_loss = compute_mse_rate_loss(my_forward_eval, self.train_loader)
                        train_info[checkpoints + 1] = [ave_loss, ave_mse_train, ave_rate_train, training_time]
                        print('\ttrain loss: {:.6f}, mse: {:.4f}, rate: {:.4f}'.format(ave_loss, ave_mse_train, ave_rate_train))

                        ave_mse_val, ave_rate_val, ave_loss = compute_mse_rate_loss(my_forward_eval, self.val_loader)
                        val_info[checkpoints + 1] = [ave_loss, ave_mse_val, ave_rate_val, training_time]
                        print('\tvalidation loss: {:.6f}, mse: {:.4f}, rate: {:.4f}'.format(ave_loss, ave_mse_val, ave_rate_val))
                        self.model.train()

                        to_save = {}
                        to_save['config'] = config
                        to_save['optimizer_state'] = optimizer.state_dict()
                        to_save['model_state'] = self.model.state_dict()
                        to_save['training_time'] = training_time
                        to_save['traing_info'] = train_info
                        to_save['val_info'] = val_info
                        to_save['current_checkpoint'] = checkpoints
                        to_save['compression_stats'] = compression_stats

                        async def actual_save():
                            # TODO: make better saves, 1) mv file as backup, 2) save new data 3) delte bk
                            torch.save(to_save, f'{config["log_dir"]}/{self.name}_ft_{config["tag"]}.th')

                        asyncio.run_coroutine_threadsafe(actual_save(), new_loop)
                    
                        start_time = time.time()
                        avg_loss_ = AverageMeter()

                    if s >= config['steps']:
                        break
            q_bar.close()

            async def last_task():
                print("Async file saving has been finished.")
                new_loop.stop()

            asyncio.run_coroutine_threadsafe(last_task(), new_loop)

        return finetuning