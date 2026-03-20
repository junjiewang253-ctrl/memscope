from dataclasses import dataclass, field
from typing import Dict, List, Optional
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
class RuntimeTensorInfo:
    """
    运行时张量快照：用于描述在某个时间点，Hook捕获到的具体Tensor信息
    """
    name: str = ""                                  # 张量名称 (如 "hidden_states", "grad_weight")
    shape: List[int] = field(default_factory=list)  # 真实 Shape (如 [4, 2048, 4096])
    dtype: str = ""                                 # 真实数据类型 (如 "torch.bfloat16")
    device: str = ""                                # 设备 (如 "cuda:0")
    requires_grad: bool = False                     # 是否需要梯度 (决定反向传播时是否占显存)
    bytes: int = 0                                  # 占用字节数 (shape 元素个数 * dtype 字节大小)

@dataclass
class RuntimeEvent:
    """
    运行时事件： 每一次 Hook 触发（如进入一个 Module 的前向传播），就会生成一个这样的事件
    记录训练过程中每一个关键节点的显存变化快照
    - module forward_pre
    - module forward
    - tensor backward grad
    - step boundary
    """
    event_type: str       # 事件类型：如 "forward_pre", "forward_post", "backward", "optimizer_step"
    module: str           # 触发事件的模块名：如 "model.layers.0.self_attn"
    phase: str            # 阶段：如 "forward", "backward"
    step: int = 0         # 当前的训练步数 (Global Step)

    # --- 显存水位的“前后对比” (核心逻辑) ---
    # 通过对比 Before 和 After，我们可以精确计算出“这个算子到底吃掉了多少显存”
    mem_allocated_before: int = 0  # 事件发生前，PyTorch 已分配的显存
    mem_allocated_after: int = 0   # 事件发生后，PyTorch 已分配的显存
    mem_reserved_before: int = 0   # 事件发生前，PyTorch 向驱动申请的保留显存 (包含碎片)
    mem_reserved_after: int = 0    # 事件发生后，PyTorch 向驱动申请的保留显存
    
    # --- 衍生指标 (自动计算) ---
    max_mem_allocated: int = 0     # 历史最大分配显存 (可用于追踪全局峰值)
    max_mem_reserved: int = 0
    delta_allocated: int = 0       # 净增量 = allocated_after - allocated_before
    delta_reserved: int = 0

    # 记录该事件涉及的具体 Tensor 信息，用于深度分析“是谁占用了显存”
    inputs: List[RuntimeTensorInfo] = field(default_factory=list)  # 输入张量列表
    outputs: List[RuntimeTensorInfo] = field(default_factory=list) # 输出张量列表
    grads: List[RuntimeTensorInfo] = field(default_factory=list)   # 涉及的梯度张量列表

    notes: str = "" # 备注：可记录特殊情况，如 "Triggered Activation Checkpointing"

@dataclass
class RuntimePeak:
    """
    峰值快照
    """
    phase: str = ""
    module: str = ""
    step: int = 0
    memory_bytes: int = 0
    reserved_bytes: int = 0

@dataclass
class RuntimeReport:
    """
    运行时显存分析报告。
    基于真实 GPU 运行时的 Hook 数据生成。
    """

    # 时间序列的事件列表: 按时间顺序记录所有 RuntimeEvent, 可以画出 "显存随时间变化曲线"，直观看到哪里出现了尖峰
    runtime_trace: List[RuntimeEvent] = field(default_factory=list)

    peak: RuntimePeak = field(default_factory=RuntimePeak)
    top_events: List[RuntimeEvent] = field(default_factory=list)

    # 运行时环境元数据, 示例：{"gpu_type": "A100", "cuda_version": "12.1", "pytorch_version": "2.5"}
    metadata: Dict[str, str] = field(default_factory=dict)
    comparisons: Dict[str, float] = field(default_factory=dict)