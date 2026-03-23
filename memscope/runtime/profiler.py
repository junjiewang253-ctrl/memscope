from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

try:
    import torch
    from torch.profiler import ProfilerActivity
except ImportError:
    torch = None
    ProfilerActivity = None

def profiler_supported() -> bool:
    return torch is not None and hasattr(torch, "profiler")

def build_profiler(
        *, 
        enabled: bool, 
        device: str, 
        record_shapes: bool = True, 
        profile_memory: bool = True, 
        with_stack: bool = False, 
        with_flops: bool = False, 
):
    """
    返回一个 profiler context:
    - enabled=False 时返回 nullcontext()
    - 不支持 torch.profiler 时返回 nullcontext()
    """
    if not enabled:
        return nullcontext()
    
    if not profiler_supported():
        return nullcontext()
    
    # 构建活动列表 (Activities)
    # ProfilerActivity.CPU: 始终记录 CPU 端的操作（如数据加载、Python 开销）
    activities = [ProfilerActivity.CPU]

    # 动态判断：如果设备是 cuda 且 CUDA 可用，则添加 CUDA 活动记录
    if str(device).startswith("cuda") and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # [核心接口] 返回真正的 PyTorch Profiler 上下文对象
    # 这是一个生成器风格的上下文管理器，进入时开始记录，退出时停止并整理数据。
    return torch.profiler.profile(
        activities=activities, 
        record_shapes=record_shapes, 
        profile_memory=profile_memory, 
        with_stack=with_stack, 
        with_flops=with_flops, 
    )

def export_chrome_trace(prof, outpath: str | Path) -> bool:
    """
    导出 chrome trace。
    成功返回 True，失败返回 False。
    """
    if prof is None:
        return False
    
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    try: 
        prof.export_chrome_trace(str(outpath))
        return True
    except Exception:
        return False