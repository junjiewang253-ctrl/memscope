import argparse
from pathlib import Path
from dataclasses import asdict

import yaml

from memscope.parsers.config_loader import load_config
from memscope.report.json_reporter import write_json_report
from memscope.runtime.hooks import register_runtime_hooks
from memscope.runtime.memory import (
    cuda_available, 
    get_device_name, 
    memory_stats, 
    reset_peak_memory_stats, 
    synchronize_if_needed, 
)
from memscope.runtime.tracer import RuntimeTracer
from memscope.static.analyzer import analyze_static
from memscope.models.toy_transformer import ToyTransformerLM

import torch
import torch.nn as nn

def load_runtime_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def pick_torch_dtype(dtype: str):
    key = dtype.lower()
    if key in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if key in ["fp16", "float16"]:
        return torch.float16
    if key in ["fp32", "float32", "float"]:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype}")

def move_model_dtype(model, device: str, dtype_str: str):
    model = model.to(device)
    # embedding / layernorm / linear 是否统一 cast，MVP 先整体 cast
    target_dtype = pick_torch_dtype(dtype_str)

    # CPU 下部分 op 对 bf16/fp16 支持有限，保守处理
    if device == "cpu" and target_dtype in [torch.float16, torch.bfloat16]:
        model = model.to(torch.float32)
    else:
        model = model.to(target_dtype)
    return model

def main():
    parser = argparse.ArgumentParser()
    # 必需参数：模型结构和训练配置的 JSON/YAML
    parser.add_argument("--config", required=True, help="Static model/train config json/yaml")
    # 必需参数：运行时环境的 YAML (决定跑几步、用什么卡)
    parser.add_argument("--runtime-config", required=True, help="Runtime yaml config")
    # 可选参数：输出目录
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    args = parser.parse_args()

    # 加载两份配置
    full_cfg = load_config(args.config)       # 包含 model 结构定义 + train 超参
    runtime_cfg = load_runtime_config(args.runtime_config) # 包含 device, steps 等

    # 确定运行设备：优先使用配置指定的，如果没有则自动检测 CUDA
    device = runtime_cfg.get("device", "cuda" if cuda_available() else "cpu")
    steps = int(runtime_cfg.get("steps", 1))
    lr = float(runtime_cfg.get("lr", 1e-4))
    seed = int(runtime_cfg.get("seed", 42))

    # 设置随机种子，保证实验可复现
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = ToyTransformerLM(
        vocab_size=full_cfg.model.vocab_size, 
        hidden_size=full_cfg.model.hidden_size, 
        num_layers=full_cfg.model.num_layers, 
        num_heads=full_cfg.model.num_attention_heads, 
        ffn_hidden_size=full_cfg.model.ffn_hidden_size, 
        max_seq_len=full_cfg.model.seq_len, 
    )

    model = move_model_dtype(model, device=device, dtype_str=full_cfg.train.dtype)
    model.train()

    optimizer_name = full_cfg.train.optimizer.lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    # 初始化追踪器
    tracer = RuntimeTracer(device=device)

    # 【关键】注册 Hooks
    # 这行代码执行后，模型的所有 forward/backward 过程都会被 tracer 记录
    handles = register_runtime_hooks(model, tracer)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 如果是 GPU，重置显存峰值统计
    # 目的：确保我们记录的 Peak Memory 仅包含本次脚本运行产生的显存，排除之前残留的
    if device.startswith("cuda"):
        reset_peak_memory_stats(device)

    for step in range(steps):
        tracer.set_step(step)

        B = full_cfg.train.micro_batch_size
        S = full_cfg.model.seq_len
        V = full_cfg.model.vocab_size

        input_ids = torch.randint(0, V, (B, S), device=device, dtype=torch.long)
        labels = torch.randint(0, V, (B, S), device=device, dtype=torch.long)

        # --- 1. Step 开始 ---
        synchronize_if_needed(device) # 同步 GPU，确保读数准确
        before_step = memory_stats(device) # 记录步前显存
        tracer.log_event(
            event_type="step", 
            module="train_step", 
            phase="step_start", 
            before=before_step, 
            after=before_step, 
            inputs=input_ids, 
            notes="training step start", 
        )

        optimizer.zero_grad(set_to_none=True)

        # --- 2. 前向传播 (Forward) ---
        synchronize_if_needed(device)
        before_fwd = memory_stats(device)
        logits = model(input_ids) # <--- 这里会触发 register_forward_hook
        synchronize_if_needed(device)
        after_fwd = memory_stats(device)
        tracer.log_event(
            event_type="step", 
            module="train_step", 
            phase="forward_end", 
            before=before_fwd, 
            after=after_fwd, 
            outputs=logits, 
            notes="full forward done", 
        )

        # --- 3. 计算 Loss ---
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # --- 4. 反向传播 (Backward) ---
        synchronize_if_needed(device)
        before_bwd = memory_stats(device)
        loss.backward()
        synchronize_if_needed(device)
        after_bwd = memory_stats(device)
        tracer.log_event(
            event_type="step", 
            module="train_step", 
            phase="backward_end", 
            before=before_bwd, 
            after=after_bwd, 
            outputs=loss, 
            notes="full backward done", 
        )

        # --- 5. 优化器步进 (Optimizer Step) ---
        synchronize_if_needed(device)
        before_optim = memory_stats(device)
        optimizer.step()
        synchronize_if_needed(device)
        after_optim = memory_stats(device)
        tracer.log_event(
            event_type="step", 
            module="train_step", 
            phase="optimizer_step_end", 
            before=before_optim, 
            after=after_optim, 
            notes="optimizer step done"
        )

        # --- 6. Step 结束总结 ---
        synchronize_if_needed(device)
        after_step = memory_stats(device)
        tracer.log_event(
            event_type="step", 
            module="train_step", 
            phase="step_end", 
            before=before_step, 
            after=after_step, 
            outputs=loss.detach(), 
            notes=f"loss={loss.item():.6f}", 
        )

    for h in handles:
        h.remove()

    static_report = analyze_static(full_cfg)
    static_peak = static_report.summary.peak_memory_bytes

    runtime_report = tracer.build_report(
        metadata={
            "mode": "runtime", 
            "device": device, 
            "device_name": get_device_name(device), 
            "dtype": full_cfg.train.dtype, 
            "steps": str(steps), 
            "optimizer": full_cfg.train.optimizer, 
            "model_type": full_cfg.model.model_type, 
        }, 
        comparisons={
            "static_peak_memory_bytes": float(static_peak), 
            "runtime_peak_memory_bytes": float(tracer._peak_allocated), 
            "peak_diff_bytes": float(tracer._peak_allocated - static_peak), 
            "peak_diff_ratio": float((tracer._peak_allocated - static_peak) / static_peak) if static_peak > 0 else 0.0,
        },
    )

    runtime_json_path = outdir / "runtime_report.json"
    write_json_report(runtime_report, runtime_json_path)

    print("MemScope Runtime Report")
    print(f"Device: {device}")
    print(f"Device name: {get_device_name(device)}")
    print(f"Peak allocated: {runtime_report.peak.memory_bytes} bytes")
    print(f"Peak reserved:  {runtime_report.peak.reserved_bytes} bytes")
    print(f"Peak phase:     {runtime_report.peak.phase}")
    print(f"Peak module:    {runtime_report.peak.module}")
    print(f"Saved runtime report to: {runtime_json_path}")


if __name__ == "__main__":
    main()