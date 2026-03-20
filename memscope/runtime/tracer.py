from __future__ import annotations

from typing import Any, Dict, List, Optional

from memscope.schemas.report import (
    RuntimeEvent,      # 单个事件的 schema
    RuntimePeak,       # 峰值记录的 schema
    RuntimeReport,     # 最终报告的 schema
    RuntimeTensorInfo, # Tensor 信息的 schema
)
from memscope.runtime.memory import memory_stats

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _dtype_str(t) -> str:
    if torch is None:
        return "unknown"
    return str(t.dtype).replace("torch.", "")


def _tensor_bytes(t) -> int:
    if torch is None:
        return 0
    return int(t.numel() * t.element_size())


def _flatten_tensors(obj: Any) -> List[Any]:
    """
    递归遍历任意嵌套结构（list/tuple/dict），提取出所有的 torch.Tensor。
    因为模型的输入/输出/梯度往往是复杂的嵌套字典或列表。
    """
    if torch is not None and isinstance(obj, torch.Tensor):
        return [obj]

    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            out.extend(_flatten_tensors(x))
        return out

    if isinstance(obj, dict):
        out = []
        for _, v in obj.items():
            out.extend(_flatten_tensors(v))
        return out

    return []


def tensor_to_info(t, name: str = "") -> RuntimeTensorInfo:
    """
    将一个 Tensor 对象转换为轻量级的 RuntimeTensorInfo 数据类。
    这样做的目的是避免在报告中保存巨大的 Tensor 数据本身，只保存元数据（形状、类型等）。
    """
    return RuntimeTensorInfo(
        name=name,
        shape=list(t.shape),
        dtype=_dtype_str(t),
        device=str(t.device),
        requires_grad=bool(getattr(t, "requires_grad", False)),
        bytes=_tensor_bytes(t),
    )


class RuntimeTracer:
    """
    运行时追踪器：
    - 保存事件序列
    - 保存模块 forward_pre 时的显存快照
    - 维护全局 peak
    """

    def __init__(self, device: str, step: int = 0):
        self.device = device
        self.step = step
        self.events: List[RuntimeEvent] = []

        # module_name -> memory_stats snapshot
        # 【关键逻辑】用于匹配 forward_pre 和 forward 的临时缓存
        # Key: 模块名称, Value: forward_pre 时的显存快照
        self.module_pre_stats: Dict[str, dict] = {}

        self._peak_allocated = 0
        self._peak_reserved = 0
        self._peak_phase = ""
        self._peak_module = ""
        self._peak_step = 0

    def set_step(self, step: int) -> None:
        """更新当前的训练步数 (step)，用于标记事件发生的时间点。"""
        self.step = step

    def capture_stats(self) -> dict:
        """调用底层工具获取当前时刻的显存快照。"""
        return memory_stats(self.device)

    def remember_module_pre(self, module_name: str, stats: dict) -> None:
        """
        在模块 forward 执行前，暂存显存状态。
        这是为了稍后在 forward 结束后，计算该模块单独消耗了多少显存。
        """
        self.module_pre_stats[module_name] = stats

    def pop_module_pre(self, module_name: str) -> Optional[dict]:
        """
        取出并删除之前暂存的显存状态。
        如果没找到（可能出错或未注册 pre-hook），返回 None。
        """
        return self.module_pre_stats.pop(module_name, None)

    def make_tensor_infos(self, obj: Any) -> List[RuntimeTensorInfo]:
        """组合工具：先展平对象，再批量转换为 Info 对象。"""
        tensors = _flatten_tensors(obj)
        return [tensor_to_info(t) for t in tensors]

    def _update_peak(self, module: str, phase: str, stats_after: dict) -> None:
        """
        检查当前显存是否打破了历史纪录。
        如果是，则更新内部的峰值变量。
        """
        allocated = int(stats_after.get("allocated", 0))
        reserved = int(stats_after.get("reserved", 0))

        if allocated >= self._peak_allocated:
            self._peak_allocated = allocated
            self._peak_reserved = reserved
            self._peak_phase = phase
            self._peak_module = module
            self._peak_step = self.step

    def record_event(
        self,
        *,                # 强制使用关键字参数调用，提高可读性
        event_type: str,  # 事件类型：'step', 'module', 'tensor_grad'
        module: str,      # 涉及的模块名
        phase: str,       # 阶段：'forward_pre', 'forward', 'backward'
        before: Optional[dict] = None, # 之前的显存快照
        after: Optional[dict] = None,  # 之后的显存快照
        inputs: Optional[Any] = None,  # 输入数据
        outputs: Optional[Any] = None, # 输出数据
        grads: Optional[Any] = None,   # 梯度数据
        notes: str = "",
    ) -> RuntimeEvent:
        
        # 如果没有传入快照，就现场抓一个（作为兜底策略）
        before = before or self.capture_stats()
        after = after or self.capture_stats()

        event = RuntimeEvent(
            event_type=event_type,
            module=module,
            phase=phase,
            step=self.step,
            mem_allocated_before=int(before.get("allocated", 0)),
            mem_allocated_after=int(after.get("allocated", 0)),
            mem_reserved_before=int(before.get("reserved", 0)),
            mem_reserved_after=int(after.get("reserved", 0)),
            max_mem_allocated=int(after.get("max_allocated", 0)),
            max_mem_reserved=int(after.get("max_reserved", 0)),
            # 计算差值：这是分析显存泄漏或突增的关键指标
            delta_allocated=int(after.get("allocated", 0) - before.get("allocated", 0)),
            delta_reserved=int(after.get("reserved", 0) - before.get("reserved", 0)),
            # 转换 Tensor 数据为元数据
            inputs=self.make_tensor_infos(inputs) if inputs is not None else [],
            outputs=self.make_tensor_infos(outputs) if outputs is not None else [],
            grads=self.make_tensor_infos(grads) if grads is not None else [],
            notes=notes,
        )

        # 1. 存入历史列表
        self.events.append(event)
        # 2. 检查是否更新全局峰值
        self._update_peak(module, phase, after)
        return event

    def record_step_boundary(
        self,
        *,
        phase: str,
        module: str = "train_step",
        before: Optional[dict] = None,
        after: Optional[dict] = None,
        inputs: Optional[Any] = None,
        outputs: Optional[Any] = None,
        notes: str = "",
    ) -> RuntimeEvent:
        """记录一个完整训练步骤（Step）的开始或结束。"""
        return self.record_event(
            event_type="step",
            module=module,
            phase=phase,
            before=before,
            after=after,
            inputs=inputs,
            outputs=outputs,
            notes=notes,
        )

    def record_module_forward_pre(
        self,
        *,
        module: str,
        inputs: Any,
        before: dict,
        notes: str = "",
    ) -> RuntimeEvent:
        """
        在模块 forward 执行前调用。
        关键动作：调用 remember_module_pre 保存状态，以便后续计算增量。
        """
        self.remember_module_pre(module, before) # 保存现场
        return self.record_event(
            event_type="module",
            module=module,
            phase="forward_pre",
            before=before,
            after=before,
            inputs=inputs,
            notes=notes or "before forward",
        )

    def record_module_forward(
        self,
        *,
        module: str,
        inputs: Any,
        outputs: Any,
        after: dict,
        notes: str = "",
    ) -> RuntimeEvent:
        """
        在模块 forward 执行后调用。
        关键动作：弹出之前保存的 pre 状态作为 'before'，从而精确计算该模块的显存增量。
        """

        # 尝试获取 pre 状态；如果丢失（异常），则用当前状态作为 before（差值为 0，保稳）
        before = self.pop_module_pre(module) or after

        return self.record_event(
            event_type="module",
            module=module,
            phase="forward",
            before=before,
            after=after,
            inputs=inputs,
            outputs=outputs,
            notes=notes or "after forward",
        )

    def record_tensor_grad(
        self,
        *,
        module: str,
        grad: Any,
        before: Optional[dict] = None,
        after: Optional[dict] = None,
        notes: str = "",
    ) -> RuntimeEvent:
        """
        在反向传播时，通过 Tensor Hook 捕获梯度。
        """
        return self.record_event(
            event_type="tensor_grad",
            module=module,
            phase="backward",
            before=before,
            after=after,
            grads=grad,
            notes=notes or "tensor grad captured",
        )

    def top_events_by_allocated_after(self, top_k: int = 10) -> List[RuntimeEvent]:
        """
        找出显存占用绝对值最大的事件。
        排序优先级：当前分配量 > 历史最大分配量 > 增量
        """
        return sorted(
            self.events,
            key=lambda e: (e.mem_allocated_after, e.max_mem_allocated, e.delta_allocated),
            reverse=True,
        )[:top_k]

    def top_events_by_delta_allocated(self, top_k: int = 10) -> List[RuntimeEvent]:
        """
        找出显存增量（跳变）最大的事件。
        这通常对应于大型算子（如 Attention 矩阵乘法）的执行瞬间。
        """
        return sorted(
            self.events,
            key=lambda e: e.delta_allocated,
            reverse=True,
        )[:top_k]

    def build_report(
        self,
        metadata: Optional[dict] = None,
        comparisons: Optional[dict] = None,
        top_k: int = 10,
    ) -> RuntimeReport:
        """
        打包所有数据，生成最终的 RuntimeReport 对象。
        """
        # 1. 封装峰值信息
        peak = RuntimePeak(
            phase=self._peak_phase,
            module=self._peak_module,
            step=self._peak_step,
            memory_bytes=self._peak_allocated,
            reserved_bytes=self._peak_reserved,
        )

        # 2. 获取 Top 事件（按显存占用排序）
        top_events = self.top_events_by_allocated_after(top_k=top_k)

        # 3. 组装报告
        return RuntimeReport(
            runtime_trace=self.events,   # 完整的时间线
            peak=peak,                   # 峰值摘要
            top_events=top_events,       # 关键事件摘要
            metadata=metadata or {},     # 额外元数据（如模型名、GPU 型号）
            comparisons=comparisons or {}, # 与静态分析的对比数据
        )