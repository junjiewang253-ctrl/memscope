from __future__ import annotations

from pathlib import Path
from typing import Optional

try: 
    import torch 
except ImportError:
    torch = None

def snapshot_supported(device: str) -> bool:
    """
    只有 CUDA 且 torch.cuda.memory 私有接口存在时才支持。
    """
    if torch is None:
        return False
    if not str(device).startswith("cuda"):
        return False
    if not torch.cuda.is_available():
        return False
    if not hasattr(torch.cuda, "memory"):
        return False
    mem_mod = torch.cuda.memory
    return hasattr(mem_mod, "_record_memory_history") and hasattr(mem_mod, "_dump_snapshot")

def start_memory_history(
        *, 
        device: str, 
        enabled: bool, 
        max_entries: int = 100000, 
):
    """
    开启 CUDA allocator memory history 记录。
    """
    if not enabled:
        return False
    if not snapshot_supported(device):
        return False
    
    try:
        # [核心接口] 新版 PyTorch 调用方式
        # enabled="all": 记录所有分配器事件 (包括 mmap, virtual alloc 等)
        # max_entries: 限制记录的最大事件数，防止内存爆掉 (默认 10 万条)
        torch.cuda.memory._record_memory_history(
            enabled="all", 
            max_entries=max_entries, 
        )
        return True
    except TypeError:
        # 有些 torch 版本签名不同，只接受布尔值
        try:
            torch.cuda.memory._record_memory_history(True)
            return True
        except Exception:
            return False
    except Exception:
        return False
    
def stop_memory_history(*, device:str) -> bool:
    """
    关闭 history 记录
    """
    if not snapshot_supported(device):
        return False
    
    try:
        # [核心接口] 新版调用：传入 None 表示停止
        torch.cuda.memory._record_memory_history(enabled=None)
        return True
    except TypeError:
        # [兼容性处理] 旧版调用：传入 False 表示停止
        try:
            torch.cuda.memory._record_memory_history(False)
            return True
        except Exception:
            return False
    except Exception:
        return False
    
def dump_memory_snapshot(
        *, 
        device: str, 
        outpath: str | Path, 
        enabled: bool, 
) -> bool:
    """
    导出 snapshot.pickle
    生成的文件包含从 start 到 stop 期间所有的显存分配/释放事件及堆栈信息
    """
    if not enabled:
        return False
    if not snapshot_supported(device):
        return False
    
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.cuda.memory._dump_snapshot(str(outpath))
        return True
    except Exception:
        return False