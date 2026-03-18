from dataclasses import dataclass, field
from typing import Dict, List
from .op import OpRecord

@dataclass
class Summary:
    """
    显存占用统计摘要。
    将所有算子的内存消耗按类别归类，并计算峰值。
    """
    param_count: int = 0

    # 权重显存占用, 存储模型参数本身所需的显存, 这是静态占用的大头
    weight_memory_bytes: int = 0

    # 梯度显存占用, 反向传播时，为每个参数存储梯度所需的显存, 仅在训练阶段存在
    grad_memory_bytes: int = 0

    # 优化器状态显存占用, AdamW 等优化器需要存储动量 (m) 和方差 (v), 通常是权重显存的 2倍 (fp16混合精度下) 或更多, 这是训练显存的大头
    optimizer_memory_bytes: int = 0

    # 激活值显存占用, 前向传播产生的中间结果，需保留用于反向传播, 随 Batch Size 和 Sequence Length 线性增长。开启重计算可大幅降低此项
    activation_memory_bytes: int = 0

    # 临时缓冲区显存占用, 算子执行过程中产生的临时变量 (如 MatMul 的 workspace, Dropout 的 mask), 生命周期极短，算子结束即释放
    temporary_memory_bytes: int = 0

    # 持久化显存占用, 当前阶段必须保留的所有显存总和 (权重 + 梯度 + 激活值 + 优化器状态)
    persistent_memory_bytes: int = 0

    # 显存峰值, 整个训练/推理过程中，显存占用的最高水位线, max(persistent + temporary) across all steps
    # 这是决定 "OOM (Out Of Memory)" 的关键指标
    peak_memory_bytes: int = 0

    # 达到峰值的阶段, 记录峰值发生在什么时候，方便定位优化点
    peak_stage: str = ""

@dataclass
class StaticReport:
    """
    静态显存估算报告。
    基于配置和公式推导得出，无需 GPU 运行。
    """
    summary: Summary
    operators: List[OpRecord] = field(default_factory=list)

    # 报告的元数据: 记录生成报告的环境信息，便于复现
    # 示例：{"model_name": "Llama-7B", "batch_size": "4", "timestamp": "2026-03-17"}
    metadata: Dict[str, str] = field(default_factory=dict)

@dataclass
class RuntimeEvent:
    """
    运行时显存事件。
    对应 PyTorch 等框架中某个 Module 执行前后的显存状态。
    """
    module: str
    phase: str
    mem_allocated_before: int
    mem_allocated_after: int
    delta: int
    inputs: List[dict] = field(default_factory=list)
    outputs: List[dict] = field(default_factory=list)

@dataclass
class RuntimeReport:
    """
    运行时显存分析报告。
    基于真实 GPU 运行时的 Hook 数据生成。
    """

    # 时间序列的事件列表: 按时间顺序记录所有 RuntimeEvent, 可以画出 "显存随时间变化曲线"，直观看到哪里出现了尖峰
    runtime_trace: List[RuntimeEvent] = field(default_factory=list)

    peak_memory_bytes: int = 0
    peak_stage: str = ""

    # 运行时环境元数据, 示例：{"gpu_type": "A100", "cuda_version": "12.1", "pytorch_version": "2.5"}
    meta_data: Dict[str, str] = field(default_factory=dict)