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
    """
    模块过滤策略：
    - 不 hook 根模块
    - 保留关键层
    """
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
    """
    递归提取嵌套结构中的所有 Tensor。
    模型的输出可能是复杂的：(Tensor, Dict[str, Tensor], List[Tensor])。
    我们需要把它们全部“拍扁”成一个列表，以便逐个注册梯度 Hook。
    """
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
):
    """
    注册 runtime hooks:
    1. module forward_pre_hook
    2. module forward_hook
    3. output tensor backward hook
    4. parameter grad hook
    """
    handles: List = []
    # A. 注册模块级 Forward Hooks (hook_modules)
    if hook_modules:
        for module_name, module in model.named_modules():
            if not _should_hook_module(module_name, module):
                continue
            
            # --- 构造 Pre-Hook (Forward 之前执行) ---
            def make_pre_hook(name):
                def pre_hook(mod, inputs):
                    synchronize_if_needed(tracer.device)
                    before = memory_stats(tracer.device)
                    tracer.record_module_forward_pre(
                        module=name,
                        inputs=inputs,
                        before=before,
                        notes=f"{name} forward_pre",
                    )
                return pre_hook

            # --- 构造 Forward Hook (Forward 之后执行) ---
            def make_fwd_hook(name):
                def fwd_hook(mod, inputs, outputs):
                    synchronize_if_needed(tracer.device)
                    after = memory_stats(tracer.device)

                    tracer.record_module_forward(
                        module=name,
                        inputs=inputs,
                        outputs=outputs,
                        after=after,
                        notes=f"{name} forward",
                    )

                    # --- 嵌套逻辑：注册输出梯度的 Hook ---
                    if not hook_output_grads:
                        return

                    # 展平输出，找到所有需要梯度的 Tensor
                    tensors = _flatten_output_tensors(outputs)
                    for idx, t in enumerate(tensors):
                        if not isinstance(t, torch.Tensor):
                            continue
                        if not t.requires_grad:
                            continue
                        
                        # 【二次闭包】为每个 Output Tensor 单独注册梯度 Hook
                        def make_output_grad_hook(hook_name, tensor_idx):
                            def grad_hook(grad):
                                synchronize_if_needed(tracer.device)
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

                        # 注册 Hook，并将句柄保存起来以便后续清理
                        t.register_hook(make_output_grad_hook(name, idx))

                return fwd_hook

            handles.append(module.register_forward_pre_hook(make_pre_hook(module_name)))
            handles.append(module.register_forward_hook(make_fwd_hook(module_name)))

    # B. 注册参数梯度 Hooks (hook_param_grads)
    if hook_param_grads:
        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            def make_param_grad_hook(name):
                def grad_hook(grad):
                    synchronize_if_needed(tracer.device)
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