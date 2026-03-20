from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, List, Optional

from memscope.schemas.report import (
    RuntimeEvent,
    RuntimePeak, 
    RuntimeReport, 
    RuntimeTensorInfo, 
)
from memscope.runtime.memory import memory_stats

try: 
    import torch
except ImportError:
    torch = None

def _dtype_str(t) -> str:
    """
    将 PyTorch 的 dtype (如 torch.float32) 转换为纯字符串 ("float32")。
    """
    if torch is None:
        return "unknown"
    return str(t.dtype).replace("torch.", "")

def _num_bytes(t) -> int:
    """
    计算一个 Tensor 占用的显存字节数。
    公式：元素个数 (numel) × 每个元素的字节大小 (element_size)
    例如：形状 [2, 3] 的 float32 张量 = 6 个元素 × 4 字节 = 24 字节
    """
    if torch is None:
        return 0
    return int(t.numel() * t.element_size())

def _flatten_tensors(obj: Any) -> List[Any]:
    """
    递归遍历任意嵌套结构 (list/tuple/dict)，提取出所有的 torch.Tensor 对象。
    """
    # 情况 1: 如果本身就是 Tensor，直接放入列表
    if torch is not None and isinstance(obj, torch.Tensor):
        return [obj]
    # 情况 2: 如果是列表或元组，遍历每个元素并递归调用
    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            out.extend(_flatten_tensors(x))
        return out
    # 情况 3: 如果是字典，只遍历值 (values)，忽略键 (keys)
    if isinstance(obj, dict):
        out = []
        for _, v in obj.items():
            out.extend(_flatten_tensors(v))
        return out
    return []

def tensor_to_info(t, name: str = "") -> RuntimeTensorInfo:
    """
    将一个 Tensor 对象转换为 RuntimeTensorInfo 数据类实例。
    """
    return RuntimeTensorInfo(
        name=name, 
        shape=list(t.shape), 
        dtype=_dtype_str(t), 
        device=str(t.device), 
        # getattr(obj, attr, default): 安全获取属性，如果没有 requires_grad 则默认为 False
        requires_grad=bool(getattr(t, "requires_grad", False)), 
        bytes=_num_bytes(t), 
    )

class RuntimeTracer:
    def __init__(self, device: str, step: int = 0):
        self.device = device
        self.step = step
        self.events: List[RuntimeEvent] = []

        self._peak_allocated = 0
        self._peak_reserved = 0
        self._peak_phase = ""
        self._peak_module = ""
        self._peak_step = 0

    def set_step(self, step: int) -> None:
        self.step = step

    def _update_peak(self, module: str, phase: str, stats_after: dict) -> None:
        """
        检查当前显存是否打破了历史纪录。如果是，更新峰值信息。
        """
        allocated = stats_after["allocated"]
        reserved = stats_after["reserved"]

        if allocated >= self._peak_allocated:
            self._peak_allocated = allocated
            self._peak_reserved = reserved
            self._peak_phase = phase
            self._peak_module = module
            self._peak_step = self.step

    def make_tensor_infos(self, obj: Any) -> List[RuntimeTensorInfo]:
        """
        利用前面的 _flatten_tensors 和 tensor_to_info，一键转换复杂对象。
        """
        tensors = _flatten_tensors(obj)
        return [tensor_to_info(t) for t in tensors]
    
    def log_event(
            self, 
            *,  # 强制后面的参数必须使用关键字传递 (如 log_event(event_type="...", ...))，提高可读性
            event_type: str, 
            module: str, 
            phase: str, 
            before: Optional[dict] = None, 
            after: Optional[dict] = None, 
            inputs: Optional[Any] = None, 
            outputs: Optional[Any] = None, 
            grads: Optional[Any] = None, 
            notes: str = "", 
    ) -> None:
        # 1. 获取显存快照
        # 如果调用者没传 before/after，就自动调用 memory_stats 获取当前状态
        before = before or memory_stats(self.device)
        after = after or memory_stats(self.device)

        # 2. 构建事件对象
        event = RuntimeEvent(
            event_type=event_type, 
            module=module, 
            phase=phase, 
            step=self.step, 
            mem_allocated_before=before["allocated"], 
            mem_allocated_after=after["allocated"], 
            mem_reserved_before=before["reserved"], 
            mem_reserved_after=after["reserved"], 
            max_mem_allocated=after["max_allocated"], 
            delta_allocated=after["allocated"] - before["allocated"], 

            # 3. 转换输入输出张量
            inputs=self.make_tensor_infos(inputs) if inputs is not None else [], 
            outputs=self.make_tensor_infos(outputs) if outputs is not None else [], 
            grads=self.make_tensor_infos(grads) if grads is not None else [],
            notes=notes, 
        )
        
        # 4. 存入列表
        self.events.append(event)

        # 5. 更新峰值记录
        self._update_peak(module, phase, after)

    def build_report(self, metadata: Optional[dict] = None, comparisons: Optional[dict] = None) -> RuntimeReport:
        """
        将收集到的所有事件和峰值信息打包成最终的 RuntimeReport 对象。
        """
        peak = RuntimePeak(
            phase=self._peak_phase, 
            module=self._peak_module, 
            step=self._peak_step, 
            memory_bytes=self._peak_allocated, 
            reserved_bytes=self._peak_reserved,
        )
        return RuntimeReport(
            runtime_trace=self.events, 
            peak=peak,
            metadata=metadata or {}, 
            comparisons=comparisons or {}
        )
    
    def to_dict(self, report: RuntimeReport) -> dict:
        """
        利用 dataclasses.asdict 将报告对象转换为普通字典，方便保存为 JSON。
        """
        return asdict(report)