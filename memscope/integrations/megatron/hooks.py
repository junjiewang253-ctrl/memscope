from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from memscope.runtime.hooks import register_runtime_hooks


def unwrap_for_memscope(model):
    if isinstance(model, (list, tuple)):
        target = model[0]
    else:
        target = model

    seen = set()
    while hasattr(target, "module") and id(target) not in seen:
        seen.add(id(target))
        target = target.module

    return target


def register_megatron_runtime_hooks(
    model,
    tracer,
    *,
    hook_modules=True,
    hook_output_grads=True,
    hook_param_grads=True,
    sync_on_module_hooks=False,
):
    target = unwrap_for_memscope(model)
    return register_runtime_hooks(
        target,
        tracer,
        hook_modules=hook_modules,
        hook_output_grads=hook_output_grads,
        hook_param_grads=hook_param_grads,
        sync_on_module_hooks=sync_on_module_hooks,
    )