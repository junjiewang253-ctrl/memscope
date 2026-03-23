from __future__ import annotations

from typing import Dict

try:
    import torch
except ImportError:
    torch = None

def cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()

def get_device_name(device: str) -> str:
    if not cuda_available():
        return "cpu"
    if device.startswith("cuda"):
        idx = torch.device(device).index or 0
        return torch.cuda.get_device_name(idx)
    return str(device)

def memory_stats(device: str) -> Dict[str, int]:
    """
    返回当前设备显存统计。
    CPU 模式下统一返回 0，保证 runtime 脚本在无 GPU 环境也能跑通。
    """
    if not cuda_available():
        return {
            "allocated": 0,
            "reserved": 0,
            "max_allocated": 0,
            "max_reserved": 0,
        }
    
    dev = torch.device(device)
    return {
        "allocated": int(torch.cuda.memory_allocated(dev)), 
        "reserved": int(torch.cuda.memory_reserved(dev)), 
        "max_allocated": int(torch.cuda.max_memory_allocated(dev)), 
        "max_reserved": int(torch.cuda.max_memory_reserved(dev)), 
    }

def reset_peak_memory_stats(device: str) -> None:
    """
    清零“历史峰值”记录。
    用途：在开始一个新的测试阶段（如 Forward 或 Backward）前调用，
    这样 max_allocated 就只记录当前阶段的峰值，而不是全局的。
    """
    if not cuda_available():
        return
    dev = torch.device(device)
    torch.cuda.reset_peak_memory_stats(dev)

def synchronize_if_needed(device: str) -> None:
    """
    确保 GPU 上的所有操作都执行完毕。
    用途：GPU 是异步执行的（CPU 发令后就不管了）。
    如果不加这个，你测到的显存可能是“还没发生”的状态，导致数据不准。
    """
    if not cuda_available():
        return 
    # 只有 CUDA 设备才需要同步
    if str(device).startswith("cuda"):
        # 阻塞 CPU，直到 GPU 上所有排队的工作都做完
        torch.cuda.synchronize(torch.device(device))