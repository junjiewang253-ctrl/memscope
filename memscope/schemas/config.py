from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_type: str = "llama"
    num_layers: int = 1
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_kv_heads: int = 32
    ffn_hidden_size: int = 11008
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None
    seq_len: int = 2048
    query_seq_len: Optional[int] = None
    key_seq_len: Optional[int] = None

@dataclass
class TrainConfig:
    micro_batch_size: int = 1
    grad_accum_steps: int = 1
    # --- 分布式并行策略 ---
    # 张量并行（TP）: 将单层模型切分到多个GPU。显存需求近似除以 TP 数
    tensor_parallel: int = 1
    # 流水线并行（PP）：将不同层切分到多个GPU
    pipeline_parallel: int = 1
    # 数据并行（DP）：复制模型到多个GPU，显存需求不变，但总批次增大
    data_parallel: int = 1
    # ZeRO 优化阶段 (DeepSpeed/FSDP)
    # 0: 无优化; 1: 优化器分片; 2: 优化器+梯度分片; 3: 优化器+梯度+参数分片。
    # 逻辑：Stage 越高，单卡显存占用越低，但通信开销越大。
    zero_stage: int = 0
    dtype: str = "bf16"
    optimizer: str = "adamw"
    # 激活重计算：True 时，不保存所有中间激活值，而是反向传播时重新计算。
    activation_checkpointing: bool = False
    # 梯度桶大小: 在 DDP/ZeRO 中，梯度会被打包成桶进行通信。影响临时显存峰值
    reduce_bucket_size: int = 500_000_000

@dataclass
class FullConfig:
    model: ModelConfig
    train: TrainConfig