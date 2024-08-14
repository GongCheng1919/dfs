import torch
import numpy as np
import torch.nn.utils.prune as prune

def get_specific_submodule(module,submodule_type=()):
    if submodule_type == ():
        return None
    named_modules = []
    for n,m in module.named_modules():
        if isinstance(m,submodule_type):
            named_modules.append((n,m))
    return named_modules

def get_prune_params(named_modules,prune_pos=("weight",)):
    param_list = []
    for n,m in named_modules:
        for key in prune_pos:
            param_list.append((m,key))
    return tuple(param_list)

def global_remove(parameters_to_prune):
    for m,n in parameters_to_prune:
        prune.remove(m,n)