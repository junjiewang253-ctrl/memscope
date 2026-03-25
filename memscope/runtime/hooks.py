from __future__ import annotations

from typing import List

from memscope.runtime.memory import memory_stats, synchronize_if_needed

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _module_class_name(module) -> str:
    return module.__class__.__name__.lower()


def _should_hook_module(module_name: str, module) -> bool:
    if module_name == "":
        return False

    cls_name = _module_class_name(module)

    interesting = [
        "embedding",
        "linear",
        "layernorm",
        "rmsnorm",
        "attention",
        "mlp",
        "block",
        "lmhead",
        "selfattention",
    ]
    return any(key in cls_name for key in interesting)


def _flatten_output_tensors(outputs):
    if torch is not None and isinstance(outputs, torch.Tensor):
        return [outputs]

    if isinstance(outputs, (list, tuple)):
        out = []
        for x in outputs:
            out.extend(_flatten_output_tensors(x))
        return out

    if isinstance(outputs, dict):
        out = []
        for _, v in outputs.items():
            out.extend(_flatten_output_tensors(v))
        return out

    return []


def register_runtime_hooks(
    model,
    tracer,
    *,
    hook_modules: bool = True,
    hook_output_grads: bool = True,
    hook_param_grads: bool = True,
    sync_on_module_hooks: bool = False,
):
    handles: List = []

    def _maybe_sync():
        if sync_on_module_hooks:
            synchronize_if_needed(tracer.device)

    if hook_modules:
        for module_name, module in model.named_modules():
            if not _should_hook_module(module_name, module):
                continue

            def make_pre_hook(name):
                def pre_hook(mod, inputs):
                    _maybe_sync()
                    before = memory_stats(tracer.device)
                    tracer.record_module_forward_pre(
                        module=name,
                        inputs=inputs,
                        before=before,
                        notes=f"{name} forward_pre",
                    )
                return pre_hook

            def make_fwd_hook(name):
                def fwd_hook(mod, inputs, outputs):
                    _maybe_sync()
                    after = memory_stats(tracer.device)

                    tracer.record_module_forward(
                        module=name,
                        inputs=inputs,
                        outputs=outputs,
                        after=after,
                        notes=f"{name} forward",
                    )

                    if not hook_output_grads:
                        return

                    tensors = _flatten_output_tensors(outputs)
                    for idx, t in enumerate(tensors):
                        if not isinstance(t, torch.Tensor):
                            continue
                        if not t.requires_grad:
                            continue

                        def make_output_grad_hook(hook_name, tensor_idx):
                            def grad_hook(grad):
                                _maybe_sync()
                                stats = memory_stats(tracer.device)
                                tracer.record_tensor_grad(
                                    module=f"{hook_name}.output[{tensor_idx}]",
                                    grad=grad,
                                    before=stats,
                                    after=stats,
                                    notes="output tensor grad",
                                )
                                return grad
                            return grad_hook

                        t.register_hook(make_output_grad_hook(name, idx))

                return fwd_hook

            handles.append(module.register_forward_pre_hook(make_pre_hook(module_name)))
            handles.append(module.register_forward_hook(make_fwd_hook(module_name)))

    if hook_param_grads:
        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            def make_param_grad_hook(name):
                def grad_hook(grad):
                    _maybe_sync()
                    stats = memory_stats(tracer.device)
                    tracer.record_tensor_grad(
                        module=f"param::{name}",
                        grad=grad,
                        before=stats,
                        after=stats,
                        notes="parameter grad",
                    )
                    return grad
                return grad_hook

            handles.append(param.register_hook(make_param_grad_hook(param_name)))

    return handles