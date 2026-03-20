from __future__ import annotations

from typing import List

from memscope.runtime.memory import memory_stats, synchronize_if_needed

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

def _should_hook_module(module_name: str, module) -> bool:
    """
    判断当前模块是否需要被 Hook。
    策略：白名单机制。只关注对显存影响大的核心组件
    过滤掉最外层空名模块，同时避免 hook 过多无意义节点。
    这里先保留：
    - Embedding
    - Linear
    - LayerNorm / RMSNorm-like
    - Attention / MLP / Block
    - 最终 lm_head
    """

    # 1. 过滤掉根节点 (通常 name 为空字符串 "")
    if module_name == "":
        return False
    
    # 2. 获取类名并转为小写，方便模糊匹配
    cls_name = module.__class__.__name__.lower()

    # 3. 定义感兴趣的关键字列表
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

    # 4. 核心逻辑：只要类名中包含列表中任意一个关键字，就返回 True
    return any(key in cls_name for key in interesting)

def register_runtime_hooks(model, tracer):
    """
    注册：
    1. forward_pre_hook
    2. forward_hook
    3. output tensor backward hook
    为模型的指定模块注册 Forward 和 Backward 钩子。
    :param model: PyTorch 模型对象 (nn.Module)
    :param tracer: RuntimeTracer 实例，用于记录数据
    :return: 返回所有 hook handle 的列表，用于后续移除 hook
    """
    handles: List = [] # 用于存储注册的句柄，方便以后取消注册
    
    # 遍历模型的所有命名子模块
    # named_modules() 返回 (name, module_object) 的迭代器
    # 例如: ("layers.0.self_attn", SelfAttention(...))
    for module_name, module in model.named_modules():

        # 第一步：过滤，只处理感兴趣的模块
        if not _should_hook_module(module_name, module):
            continue
        
        # ---------------------------------------------------------
        # 第二步：动态创建 Pre-Hook (前向传播开始前)
        # ---------------------------------------------------------
        def make_pre_hook(name):
            """
            工厂函数：返回一个具体的 pre_hook 函数。
            为什么要用工厂函数？为了利用闭包锁定当前的 name 变量！
            如果直接定义 pre_hook，循环结束时 name 会变成最后一个模块的名字。
            """
            def pre_hook(mod, inputs):
                # 1. 同步 GPU：确保之前的操作都执行完了，读数才准
                synchronize_if_needed(tracer.device)

                # 2. 读取当前显存状态 (Before)
                before = memory_stats(tracer.device)

                # 3. 记录事件
                # 注意：这里 before 和 after 暂时一样，因为还没开始算
                # 主要目的是记录 inputs 和进入时刻的状态
                tracer.log_event(
                    event_type="module",
                    module=name,
                    phase="forward_pre", 
                    before=before,
                    after=before,
                    inputs=inputs,
                    notes="before forward",
                )
            return pre_hook
        
        # ---------------------------------------------------------
        # 第三步：动态创建 Forward-Hook (前向传播结束后)
        # ---------------------------------------------------------
        def make_fwd_hook(name):
            """
            工厂函数：返回一个具体的 fwd_hook 函数。
            这里逻辑更复杂，因为它还要注册梯度 Hook。
            """
            def fwd_hook(mod, inputs, outputs):
                # 1. 同步 GPU
                synchronize_if_needed(tracer.device)

                # 2. 读取结束时的显存状态 (After)
                after = memory_stats(tracer.device)

                # 3. 记录前向传播完成的事件
                tracer.log_event(
                    event_type="module", 
                    module=name, 
                    phase="forward",
                    # 优化建议：可以通过 tracer 缓存 pre_hook 的 before 值传过来
                    before=memory_stats(tracer.device), 
                    after=after, 
                    inputs=inputs, 
                    outputs=outputs, 
                    notes="after forward", 
                )

                # -----------------------------------------------------
                # 第四步：注册梯度 Hook (Backward Hook)
                # -----------------------------------------------------
                # 只有当输出需要计算梯度时，才注册 backward hook
                tensors = []
                if torch is not None:
                    if isinstance(outputs, torch.Tensor):
                        tensors = [outputs]
                    elif isinstance(outputs, (list, tuple)):
                        tensor = [x for x in outputs if isinstance(x, torch.Tensor)]

                for idx, t in enumerate(tensors):
                    if t.requires_grad:

                        # 再次使用工厂函数制造 grad_hook，锁定 name 和 idx
                        def make_grad_hook(hook_name, tensor_idx):
                            def grad_hook(grad):
                                # 1. 同步 GPU (反向传播时也要同步！)
                                synchronize_if_needed(tracer.device)

                                # 2. 读取反向传播时的显存
                                after_bwd = memory_stats(tracer.device)

                                # 3. 记录梯度事件
                                tracer.log_event(
                                    event_type="tensor_grad", 
                                    module=f"{hook_name}.output[{tensor_idx}]", 
                                    phase="backward", 
                                    before=after_bwd, 
                                    grads=grad,       # 捕获梯度张量 
                                    notes="output grad capture by tensor.register_hook", 
                                )
                                return grad
                            return grad_hook
                        
                        t.register_hook(make_grad_hook(name, idx))
            return fwd_hook
        
        # ---------------------------------------------------------
        # 第五步：正式注册 Hook 到 PyTorch 模块上
        # ---------------------------------------------------------
        # register_forward_pre_hook: 在 forward() 执行前调用
        handles.append(module.register_forward_pre_hook(make_pre_hook(module_name)))
        # register_forward_hook: 在 forward() 执行后调用
        handles.append(module.register_forward_hook(make_fwd_hook(module_name)))
    
    # 返回所有句柄，调用者可以用这些句柄来 remove_hook (清理内存)
    return handles