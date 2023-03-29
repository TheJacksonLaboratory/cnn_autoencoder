import torch
import torch.nn as nn


def trainable_module(func):
    def trainable_module_step(*args, **kwargs):
        torch.set_grad_enabled(True)
        output = func(*args, **kwargs)
        torch.set_grad_enabled(True)
        return output

    return trainable_module_step


def fixed_module(func):
    def fixed_module_step(*args, **kwargs):
        torch.set_grad_enabled(False)
        output = func(*args, **kwargs)
        torch.set_grad_enabled(True)
        return output

    return fixed_module_step


def model_step(model, module_key, *args, **kwargs):
    output = model[module_key](*args, **kwargs)
    return output


def identity_single_output_step(model, module_key, *args, **kwargs):
    return args[0]


def identity_double_output_step(model, module_key, *args, **kwargs):
    return args[0], None


def empty_single_output_step(model, module_key, *args, **kwargs):
    return None


def empty_double_output_step(model, module_key, *args, **kwargs):
    return None, None


def decorate_trainable_modules(trainable_modules=None,
                               enabled_modules=None):
    if enabled_modules is None:
        enabled_modules = ['encoder', 'decoder', 'fact_ent', 'class_model',
                           'seg_model']

    if trainable_modules is None:
        trainable_modules = []

    if 'encoder' in enabled_modules:
        if 'encoder' in trainable_modules:
            encoder_step_fun = trainable_module(model_step)
        else:
            encoder_step_fun = fixed_module(model_step)
    else:
        encoder_step_fun = identity_single_output_step

    if 'fact_ent' in enabled_modules:
        if 'fact_ent' in trainable_modules:
            fact_ent_step_fun = trainable_module(model_step)
        else:
            fact_ent_step_fun = fixed_module(model_step)
    else:
        fact_ent_step_fun = identity_double_output_step

    if 'decoder' in enabled_modules:
        if 'decoder' in trainable_modules:
            decoder_step_fun = trainable_module(model_step)
        else:
            decoder_step_fun = fixed_module(model_step)
    else:
        decoder_step_fun = identity_double_output_step

    if 'class_model' in enabled_modules:
        if 'class_model' in trainable_modules:
            class_model_step_fun = trainable_module(model_step)
        else:
            class_model_step_fun = fixed_module(model_step)
    else:
        class_model_step_fun = empty_double_output_step

    if 'seg_model' in enabled_modules:
        if 'seg_model' in trainable_modules:
            seg_model_step_fun = trainable_module(model_step)
        else:
            seg_model_step_fun = fixed_module(model_step)
    else:
        seg_model_step_fun = empty_double_output_step

    def forward_func(x, model):
        y = encoder_step_fun(model, 'encoder', x)
        y_q, p_y = fact_ent_step_fun(model, 'fact_ent', y)
        x_r, fx_brg = decoder_step_fun(model, 'decoder', y_q)

        t_pred, t_aux_pred = class_model_step_fun(model, 'class_model', y_q)
        s_pred, s_aux_pred = seg_model_step_fun(model, 'seg_model', y_q,
                                                fx_brg=fx_brg)

        return dict(x_r=x_r, fx_brg=fx_brg, y=y, y_q=y_q, p_y=p_y, 
                    t_pred=t_pred,
                    t_aux_pred=t_aux_pred,
                    s_pred=s_pred,
                    s_aux_pred=s_aux_pred)

    return forward_func
