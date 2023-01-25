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


import os
import logging
import utils
from train_cae_ms import setup_network, setup_criteria, resume_checkpoint

import argparse
import time
from torch.backends import cudnn
cudnn.benchmark = True


def lc_exp_runner(exp_setup, lc_config, l_step_config, c_step_config,
                  resume=False):
    (l_step_optimization,
     evaluation,
     create_lc_compression_task) = exp_setup.lc_setup()
    compression_tasks = create_lc_compression_task(c_step_config)

    lc_alg = utils.RankSelectionLcAlg(model=exp_setup.model,
                                      compression_tasks=compression_tasks,
                                      lc_config=lc_config,
                                      l_step_optimization=l_step_optimization,
                                      evaluation_func=evaluation,
                                      l_step_config=l_step_config)
    if resume:
        lc_alg.run(name=exp_setup.name, tag=lc_config['tag'], restore=True)
    else:
        lc_alg.run(name=exp_setup.name, tag=lc_config['tag'])


def ft_exp_runner(exp_setup, ft_config, c_step_config, pretrained_model):
    finetuning_func = exp_setup.finetune_setup(tag_of_lc_model=ft_config['tag'],
                                               c_step_config=c_step_config,
                                               pretrained_model=pretrained_model)
    finetuning_func(ft_config)


if __name__ == "__main__":
    args = utils.get_args(task='lc-compress', mode='training')
    utils.setup_logger(args)

    # -------8<-----
    if args.lc_type == 'lc':
        cae_model = setup_network(args)
        criterion, _ = setup_criteria(args)

        train_loader, val_loader = utils.get_data(args)
        # Set the evaluation data loaders. These could contain less examples
        # for quick evaluation of the model performance.
        if args.lc_data_dir is not None:
            args.data_dir = args.lc_data_dir
        eval_trn_loader, eval_val_loader = utils.get_data(args)

        resume_checkpoint(cae_model, None, None, args.resume, gpu=args.gpu,
                          resume_optimizer=False)

        cae_model.train()

        exp_setup = utils.CompressibleCAE(
            'cnn_autoencoder', cae_model, train_loader, val_loader,
            eval_trn_loader,
            eval_val_loader,
            criterion,
            progress_bar=args.progress_bar)

        l_step_config = {
            'lr_decay_mode': args.lc_lr_decay_mode,
            'lr': args.learning_rate,
            'steps': args.steps,
            'print_freq': args.checkpoint_steps,
            'lr_decay': args.lc_lr_decay,
        }
        c_step_config = {
            'alpha': args.lc_alpha,
            'criterion': args.lc_criterion,
            'conv_scheme': args.lc_conv_scheme
        }
        lc_config = {
            'mu_init': args.lc_mu_init,
            'mu_inc': args.lc_mu_inc,
            'mu_rep': args.lc_mu_rep,
            'steps': args.lc_steps,
            'tag': args.lc_tag,
            'log_dir': args.log_dir,
        }
        exp_setup.eval_config=l_step_config
        lc_exp_runner(exp_setup, lc_config, l_step_config, c_step_config,
                      resume=args.lc_resume)

    elif args.lc_type == 'ft':
        #finetuning
        cae_model = setup_network(args)
        criterion, _ = setup_criteria(args)
        train_loader, val_loader = utils.get_data(args)

        # Set the evaluation data loaders. These could contain less examples
        # for quick evaluation of the model performance.
        if args.lc_data_dir is not None:
            args.data_dir = args.lc_data_dir
        eval_trn_loader, eval_val_loader = utils.get_data(args)

        resume_checkpoint(cae_model, None, None, args.resume, gpu=args.gpu,
                          resume_optimizer=False)

        exp_setup = utils.CompressibleCAE(
            'cnn_autoencoder', cae_model, train_loader, val_loader,
            eval_trn_loader,
            eval_val_loader,
            criterion,
            progress_bar=args.progress_bar)

        c_step_config = {
            'alpha': args.lc_alpha,
            'criterion': args.lc_criterion,
            'conv_scheme': args.lc_conv_scheme
        }

        ft_config = {
            'lr': args.learning_rate,
            'steps': args.steps,
            'print_freq': args.checkpoint_steps,
            'lr_decay': args.lc_lr_decay,
            'tag': args.lc_tag,
            'log_dir': args.log_dir,
        }
        ft_exp_runner(exp_setup, ft_config, c_step_config, args.lc_pretrained_model)

    logging.shutdown()