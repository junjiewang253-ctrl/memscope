from memscope.constants import bytes_per_dtype
from memscope.schemas.config import FullConfig
from memscope.schemas.report import Summary
from memscope.static.formulas import count_non_norm_params, estimate_param_count_llama
from memscope.static.shape_infer import infer_llama_ops

def estimate_static_summary(cfg: FullConfig):
    phi = estimate_param_count_llama(cfg)
    non_norm_phi = count_non_norm_params(cfg, phi)
    dtype_bytes = bytes_per_dtype(cfg.train.dtype)

    # MVP先按文档中 bf16 + AdamW 风格近似
    weight_memory = dtype_bytes * non_norm_phi
    grad_memory = dtype_bytes * phi

    optimizer_memory = 0
    if cfg.train.optimizer.lower() in ["adam", "adamw"]:
        # master weights + fp32 grad + exp_avg + exp_avg_sq
        optimizer_memory = 4 * non_norm_phi * 4

    ops = infer_llama_ops(cfg)
    activation_memory = sum(op.memory_bytes for op in ops if op.category not in ["communication", "backward"])
    temporary_memory = sum(op.memory_bytes for op in ops if not op.persistent)
    persistent_memory = weight_memory + grad_memory + optimizer_memory + sum(
        op.memory_bytes for op in ops if op.persistent
    )

    # MVP 简化：peak 取 persistent + top temporary contributors 的保守估计
    peak_memory = persistent_memory + temporary_memory

    if cfg.train.micro_batch_size <= 1:
        peak_stage = "optimizer_step"
    else:
        peak_stage = "backward.lm_head"

    summary = Summary(
        param_count=phi, 
        weight_memory_bytes=weight_memory, 
        grad_memory_bytes=grad_memory, 
        optimizer_memory_bytes=optimizer_memory, 
        activation_memory_bytes=activation_memory, 
        temporary_memory_bytes=temporary_memory, 
        persistent_memory_bytes=persistent_memory, 
        peak_memory_bytes=peak_memory, 
        peak_stage=peak_stage, 
    )

    return summary, ops