from __future__ import annotations

from memscope.schemas.config import FullConfig, ModelConfig, TrainConfig


def megatron_args_to_full_config(args) -> FullConfig:
    model = ModelConfig(
        model_type="megatron_gpt",
        num_layers=getattr(args, "num_layers", 1),
        hidden_size=getattr(args, "hidden_size", 1),
        num_attention_heads=getattr(args, "num_attention_heads", 1),
        num_kv_heads=getattr(
            args, "num_query_groups", getattr(args, "num_attention_heads", 1)
        ),
        ffn_hidden_size=getattr(
            args, "ffn_hidden_size", 4 * getattr(args, "hidden_size", 1)
        ),
        vocab_size=getattr(args, "vocab_size", None)
        or getattr(args, "padded_vocab_size", 0),
        padded_vocab_size=getattr(args, "padded_vocab_size", None),
        seq_len=getattr(args, "seq_length", 1),
    )

    dtype = "fp32"
    if getattr(args, "bf16", False):
        dtype = "bf16"
    elif getattr(args, "fp16", False):
        dtype = "fp16"

    zero_stage = 0
    if getattr(args, "use_distributed_optimizer", False):
        zero_stage = 1

    train = TrainConfig(
        micro_batch_size=getattr(args, "micro_batch_size", 1),
        grad_accum_steps=1,
        tensor_parallel=getattr(args, "tensor_model_parallel_size", 1),
        pipeline_parallel=getattr(args, "pipeline_model_parallel_size", 1),
        data_parallel=getattr(args, "data_parallel_size", 1),
        zero_stage=zero_stage,
        dtype=dtype,
        optimizer=getattr(args, "optimizer", "adam"),
        activation_checkpointing=(
            getattr(args, "recompute_activations", False)
            or getattr(args, "recompute_granularity", None) is not None
        ),
        reduce_bucket_size=getattr(args, "ddp_bucket_size", None) or 500_000_000,
    )

    return FullConfig(model=model, train=train)