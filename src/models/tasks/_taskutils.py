import torch
import torch.nn as nn


class ModelEmptyTask(nn.Module):
    def __init__(self, *args, **kwagrs):
        super(ModelEmptyTask, self).__init__()

    def forward(self, *args, **kwargs):
        return None, None


def trainable_module(func):
    def trainable_module_step(*args, **kwargs):
        torch.set_grad_enabled(True)
        output = func(*args, **kwargs)
        torch.enable_grad()
        return output

    return trainable_module_step


def fixed_module(func):
    def fixed_module_step(*args, **kwargs):
        torch.set_grad_enabled(False)
        output = func(*args, **kwargs)
        torch.enable_grad()
        return output

    return fixed_module_step


def model_step(model, *args, **kwargs):
    output = model(*args, **kwargs)
    return output


def decorate_trainable_modules(trainable_modules=None):
    if trainable_modules is None:
        trainable_modules = []

    if 'encoder' in trainable_modules:
        encoder_step_fun = trainable_module(model_step)
    else:
        encoder_step_fun = fixed_module(model_step)

    if 'decoder' in trainable_modules:
        fact_ent_step_fun = trainable_module(model_step)
    else:
        fact_ent_step_fun = fixed_module(model_step)

    if 'decoder' in trainable_modules:
        decoder_step_fun = trainable_module(model_step)
    else:
        decoder_step_fun = fixed_module(model_step)

    if 'class_model' in trainable_modules:
        class_model_step_fun = trainable_module(model_step)
    else:
        class_model_step_fun = fixed_module(model_step)

    if 'seg_model' in trainable_modules:
        seg_model_step_fun = trainable_module(model_step)
    else:
        seg_model_step_fun = fixed_module(model_step)

    def forward_func(x, model):
        y = encoder_step_fun(model['encoder'], x)
        y_q, p_y = fact_ent_step_fun(model['fact_ent'], y)
        x_r, x_brg = decoder_step_fun(model['decoder'], y_q)

        t_pred, t_aux_pred = class_model_step_fun(model['class_model'], y_q)
        s_pred, s_aux_pred = seg_model_step_fun(model['seg_model'], y_q,
                                                fx_brg=x_brg)

        return dict(x_r=x_r, y=y, y_q=y_q, p_y=p_y,
                    t_pred=t_pred,
                    t_aux_pred=t_aux_pred,
                    s_pred=s_pred,
                    s_aux_pred=s_aux_pred)

    return forward_func
