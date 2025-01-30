import functools
import os
from functools import wraps

import torch
from PIL import Image
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_std
from tqdm import tqdm

from util.path import mkdir_if_missing


def save_checkpoint(state_dict, directory, file_name):
    mkdir_if_missing(directory)
    fpath = os.path.join(directory, file_name)
    torch.save(state_dict, fpath)
    return fpath

def load_checkpoint(fpath):
    if not os.path.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except Exception:
        raise

    return checkpoint

def get_model_device(model):
    for param in model.parameters():
        if hasattr(param, 'device'):
            return param.device
        else:
            return torch.device('cpu')
    return torch.device('cpu')

def torch_decorator(device=None, no_grad=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_args = [arg.to(device) if isinstance(
                arg, torch.Tensor) else arg for arg in args] if device else args
            new_kwargs = {k: v.to(device) if isinstance(
                v, torch.Tensor) else v for k, v in kwargs.items()} if device else kwargs
            result = None
            try:
                if no_grad:
                    with torch.no_grad():
                        result = func(*new_args, **new_kwargs)
                else:
                    result = func(*new_args, **new_kwargs)
                return result
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return wrapper
    return decorator


def describe_model(model, display_nested=True, logging_fn=print):
    def calculate_params_percentage(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel()
                        for p in module.parameters() if p.requires_grad)
        frozen = total - trainable
        if total > 0:
            trainable_perc = (trainable / total) * 100
            frozen_perc = (frozen / total) * 100
        else:
            trainable_perc = frozen_perc = 0
        return trainable_perc, frozen_perc

    total_params = sum(p.numel() for p in model.parameters())
    trainable_perc, frozen_perc = calculate_params_percentage(model)

    logging_fn(f"Total model parameters: {total_params}")
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logging_fn(
        f"🔥 Trainable model parameters: {trainable_params} ({trainable_perc:.2f}%)")
    frozen_params = total_params - trainable_params
    logging_fn(f"🧊 Frozen model parameters: {frozen_params} ({frozen_perc:.2f}%)\n")

    for name, module in model._modules.items():
        module_total_params = sum(p.numel() for p in module.parameters())
        trainable_perc, frozen_perc = calculate_params_percentage(module)
        logging_fn(f"Attribute Name: {name}, Type: {type(module).__name__}, Param Share: {100 * module_total_params / total_params:.2f}%:")
        if display_nested:
            for sub_name, sub_module in module._modules.items():
                sub_module_total_params = sum(p.numel() for p in sub_module.parameters())

                sub_trainable_perc, sub_frozen_perc = calculate_params_percentage(
                    sub_module)
                logging_fn(
                    f"-    Attribute Name: {sub_name}, Type: {type(sub_module).__name__}, Param Share: {100 * sub_module_total_params / module_total_params:.2f}%:")
                logging_fn(
                    f"--       🔥 Trainable: {sub_trainable_perc:.2f}%, 🧊 Frozen: {sub_frozen_perc:.2f}%")

        logging_fn("\n")

    logging_fn(f'Device: {next(model.parameters()).device}')
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    logging_fn(f'Model size: {size_all_mb:.3f}MB')

def is_preprocessed_text(texts):
    if isinstance(texts, str):
        return False
    elif isinstance(texts, list) and any(isinstance(text, str) for text in texts):
        return False
    return True


def is_preprocessed_image(images):
    if isinstance(images, Image.Image):
        return False
    elif isinstance(images, list) or any(isinstance(img, Image.Image) for img in images):
        return False
    return True


def batch_process(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        batch_size = kwargs.pop('batch_size', None)
        if not batch_size:
            return func(*args, **kwargs)

        batch_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        [arg for arg in args if not isinstance(arg, torch.Tensor)]

        if not batch_args:
            return func(*args, **kwargs)

        tensor_arg = batch_args[0]
        num_batches = (tensor_arg.size(0) + batch_size - 1) // batch_size
        result_batches = []

        for i in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batched_args = [tensor_arg[start_idx:end_idx]
                            if tensor_arg is arg else arg for arg in args]

            batch_result = func(*batched_args, **kwargs)
            result_batches.append(batch_result)

        if isinstance(result_batches[0], torch.Tensor):
            return torch.cat(result_batches, dim=0)
        elif isinstance(result_batches[0], list):
            return [item for sublist in result_batches for item in sublist]
        else:
            return result_batches

    return wrapper


def group_operation(input_tensor, ids, dim=0, method='mean', **kwargs):
    ids = ids.to(torch.long)
    if method == 'mean':
        return scatter_mean(input_tensor, ids, dim=dim, **kwargs), None
    elif method == 'min':
        return scatter_min(input_tensor, ids, dim=dim, **kwargs)
    elif method == 'max':
        return scatter_max(input_tensor, ids, dim=dim, **kwargs)
    elif method == 'std':
        return scatter_std(input_tensor, ids, dim=dim, **kwargs), None
    else:
        raise ValueError("Invalid method specified. Choose from 'mean', 'std', 'min', 'max'.")

def set_module_trainable(model, module=None, set_trainable=True):
    module_nn = model if module is None else getattr(model, module)        
    for param in module_nn.parameters():
        param.requires_grad = set_trainable

    if module is None:
        if set_trainable:
            module_nn.train()
        else:
            module_nn.eval()

def get_model_dtype(model):
    for param in model.parameters():
        if isinstance(param, torch.Tensor):
            return param.dtype
    return None


class Aggregator:
    def __init__(self, agg_config):
        self.agg_strategy = agg_config['scatter_strategy']
        self.aggregate_after_config = agg_config['aggregate_after']
    
    def aggregate(self, phase, tensors, inverse_index):
        if self.aggregate_after_config[phase]['enabled']:
            aggregated_tensors, _ = group_operation(tensors, inverse_index, dim=0, method=self.agg_strategy)
            return aggregated_tensors
        return tensors
