from dataclasses import dataclass, field
from typing import Any, Dict, List

# 计算显存占用 = 元素个数 × 单个元素字节数
@dataclass
class TensorMeta:
    """
    张量元数据类。
    用于在不持有实际数据的情况下，描述一个 Tensor 的内存占用情况。
    """
    shape: List[int]
    dtype: str
    bytes: int
    name: str = ""

# 算子记录：描述一次完整的计算操作，它是显存分析的基本单元：输入 -> [计算] -> 输出
@dataclass
class OpRecord:
    """
    算子执行记录类。
    用于追踪模型前向/反向传播过程中，每一个算子的显存申请与释放情况。
    """
    name: str
    category: str
    phase: str
    inputs: List[TensorMeta] = field(default_factory=list)
    outputs: List[TensorMeta] = field(default_factory=list)
    memory_bytes: int = 0
    persistent: bool = False
    # formula: 显存计算公式的文本描述
    formula: str = ""
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)