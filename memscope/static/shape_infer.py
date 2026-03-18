from memscope.constants import bytes_per_dtype
from memscope.schemas.config import FullConfig
from memscope.schemas.op import OpRecord, TensorMeta
from memscope.static.formulas import derived_vars

def tensor_meta(name, shape, dtype):
    n = 1
    for x in shape:
        n *= x
    return TensorMeta(name=name,
                      shape=list(shape), 
                      dtype=dtype, 
                      bytes = n * bytes_per_dtype(dtype)
                      )

def infer_llama_ops(cfg: FullConfig):
    v = derived_vars(cfg)
    B, S, Sq, Sk = v["B"], v["S"], v["Sq"], v["Sk"]
    H, A, T, Dh, H_ffn, Vpad, L = (
        v["H"], v["A"], v["T"], v["Dh"], v["H_ffn"], v["Vpad"], v["L"]
    )
    dtype = cfg.train.dtype

    ops = []

    ops.append(OpRecord(
        name="embedding",
        category="embedding", 
        phase="forward", 
        inputs=[tensor_meta("token_ids", [B, S], "int8")], 
        outputs=[tensor_meta("embeddings", [S, B, H], dtype)], 
        memory_bytes=2 * B * S * H,
        formula="2BSH", 
        notes="BF16 embeddings",
    ))

    ops.append(OpRecord(
        name="rmsnorm_pre_attn", 
        category="transformer_block", 
        phase="forward", 
        outputs=[tensor_meta("rmsnorm_pre_attn", [S, B, H], dtype)], 
        memory_bytes=2 * L * B * S * H, 
        formula="2LBSH",
    ))

    head_per_tp = A // T
    kv_dim_per_tp = head_per_tp * Dh

    ops.append(OpRecord(
        name="repeated_value", 
        category="attention", 
        phase="forward", 
        outputs=[tensor_meta("repeated_value", [S, B, head_per_tp, Dh], dtype)], 
        memory_bytes=2 * L * B * S * kv_dim_per_tp, 
        formula="2LBS(A/T)Dh",
    ))

    ops.append(OpRecord(
        name="rope_query", 
        category="attention", 
        phase="forward", 
        outputs=[tensor_meta("rope_query", [S, B, head_per_tp, Dh], dtype)], 
        memory_bytes=2 * L * B * S * kv_dim_per_tp, 
        formula="2LBS(A/T)Dh",
    ))

    ops.append(OpRecord(
        name="rope_key", 
        category="attention", 
        phase="forward", 
        outputs=[tensor_meta("rope_key", [S, B, head_per_tp, Dh], dtype)], 
        memory_bytes=2 * L * B * S * kv_dim_per_tp, 
        formula="2LBS(A/T)Dh",
    ))

    ops.append(OpRecord(
        name="qk", 
        category="attention", 
        phase="forward", 
        outputs=[tensor_meta("qk", [B * head_per_tp, Sq, Sk], dtype)], 
        memory_bytes=2 * B * head_per_tp * Sq * Sk, 
        formula="2B(A/T)SqSk", 
        notes="Only created once in block1 in the referenced doc",
    ))

    ops.append(OpRecord(
        name="softmax", 
        category="attention", 
        phase="forward", 
        outputs=[tensor_meta("softmax", [B * head_per_tp, Sq, Sk], dtype)], 
        memory_bytes=2 * L * B * head_per_tp * Sq * Sk, 
        formula="2B(A/T)SqSk",
    ))

    ops.append(OpRecord(
        name="context", 
        category="attention", 
        phase="forward", 
        outputs=[tensor_meta("context", [Sq, B, head_per_tp, Dh], dtype)], 
        memory_bytes=2 * L * B * Sq * kv_dim_per_tp, 
        formula="2LBSq(A/T)Dh", 
    ))

    ops.append(OpRecord(
        name="residual_add_attn_qint8", 
        category="transformer_block", 
        phase="forward", 
        outputs=[tensor_meta("residual_add_attn_qint8", [S, B, H], "qint8")], 
        memory_bytes=L * B * S * H, 
        formula="LBSH",
    ))

    ops.append(OpRecord(
        name="residual_add_attn_bf16", 
        category="transformer_block", 
        phase="forward", 
        outputs=[tensor_meta("residual_add_attn_bf16", [S, B, H], dtype)], 
        memory_bytes=2*L*B*S*H, 
        formula="2LBSH", 
    ))

    ops.append(OpRecord(
        name="rmsnorm_post_attn", 
        category="transformer_block", 
        phase="forward", 
        outputs=[tensor_meta("rmsnorm_post_attn", [S, B, H], dtype)], 
        memory_bytes=2 * L * B * S * H, 
        formula="2LBSH", 
    ))

    ops.append(OpRecord(
        name="ffn1", 
        category="mlp", 
        phase="forward", 
        outputs=[tensor_meta("ffn1", [S, B, 2 * (H_ffn // T)], dtype)], 
        memory_bytes=4*L*B*S*(H_ffn//T), 
        formula="4LBS(H'/T)", 
    ))

    ops.append(OpRecord(
        name="silu", 
        category="mlp", 
        phase="forward", 
        outputs=[tensor_meta("silu", [S, B, (H_ffn // T)], dtype)], 
        memory_bytes=2 * L * B * S * (H_ffn // T), 
        formula="2LBS(H'/T)", 
    ))

    ops.append(OpRecord(
        name="swiglu", 
        category="mlp", 
        phase="forward", 
        outputs=[tensor_meta("swiglu", [S, B, (H_ffn // T)], dtype)], 
        memory_bytes=2 * L * B * S * (H_ffn // T), 
        formula="2LBS(H'/T)", 
    ))

    ops.append(OpRecord(
        name="residual_add_mlp_qint8", 
        category="transformer_block", 
        phase="forward", 
        outputs=[tensor_meta("residual_add_mlp_qint8", [S, B, H], "qint8")], 
        memory_bytes=L * B * S * H, 
        formula="LBSH", 
    ))

    ops.append(OpRecord(
        name="residual_add_mlp_bf16",
        category="transformer_block",
        phase="forward",
        outputs=[tensor_meta("residual_add_mlp_bf16", [S, B, H], dtype)],
        memory_bytes=2 * L * B * S * H,
        formula="2LBSH",
    ))

    ops.append(OpRecord(
        name="rmsnorm_pre_lmhead",
        category="lm_head",
        phase="forward",
        outputs=[tensor_meta("rmsnorm_pre_lmhead", [S, B, H], dtype)],
        memory_bytes=2 * B * S * H,
        formula="2BSH",
    ))

    ops.append(OpRecord(
        name="logits",
        category="lm_head",
        phase="forward",
        outputs=[tensor_meta("logits", [S, B, Vpad // T], "fp32")],
        memory_bytes=4 * B * S * (Vpad // T),
        formula="4BS(Vpad/T)",
    ))

    ops.append(OpRecord(
        name="logits_max",
        category="cross_entropy",
        phase="forward",
        outputs=[tensor_meta("logits_max", [S, B, Vpad // T], "fp32")],
        memory_bytes=4 * B * S * (Vpad // T),
        formula="4BS(Vpad/T)",
    ))

    ops.append(OpRecord(
        name="reduce_bucket",
        category="communication",
        phase="backward",
        outputs=[tensor_meta("reduce_bucket", [cfg.train.reduce_bucket_size], dtype)],
        memory_bytes=2 * cfg.train.reduce_bucket_size,
        formula="2λ",
        persistent=True,
    ))

    ops.append(OpRecord(
        name="logits_grad",
        category="backward",
        phase="backward",
        outputs=[tensor_meta("logits_grad", [S, B, Vpad // T], dtype)],
        memory_bytes=2 * B * S * (Vpad // T),
        formula="2BS(Vpad/T)",
    ))

    return ops