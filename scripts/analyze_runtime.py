import argparse
import json
from pathlib import Path

import yaml
import torch
import torch.nn as nn

from memscope.models.toy_transformer import ToyTransformerLM
from memscope.parsers.config_loader import load_config
from memscope.report.json_reporter import write_json_report
from memscope.report.runtime_markdown_reporter import render_runtime_report
from memscope.runtime.hooks import register_runtime_hooks
from memscope.runtime.memory import (
    cuda_available,
    get_device_name,
    memory_stats,
    reset_peak_memory_stats,
    synchronize_if_needed,
)
from memscope.runtime.profiler import build_profiler, export_chrome_trace, profiler_supported
from memscope.runtime.snapshot import (
    dump_memory_snapshot,
    snapshot_supported,
    start_memory_history,
    stop_memory_history,
)
from memscope.runtime.tracer import RuntimeTracer
from memscope.static.analyzer import analyze_static
from memscope.utils import format_bytes


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
    target_dtype = pick_torch_dtype(dtype_str)

    if device == "cpu" and target_dtype in [torch.float16, torch.bfloat16]:
        model = model.to(torch.float32)
    else:
        model = model.to(target_dtype)
    return model

def main():
    parser = argparse.ArgumentParser()
    # 必需参数：静态模型配置文件 (JSON/YAML)
    parser.add_argument("--config", required=True, help="Path to static model/train config json/yaml")
    # 必需参数：运行时配置文件 (YAML)，控制步数、是否开启 Profiler 等
    parser.add_argument("--runtime-config", required=True, help="Path to runtime yaml config")
    # 输出目录
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    runtime_cfg = load_runtime_config(args.runtime_config)

    device = runtime_cfg.get("device", "cuda" if cuda_available() else "cpu")
    steps = int(runtime_cfg.get("steps", 1))
    lr = float(runtime_cfg.get("lr", 1e-4))
    seed = int(runtime_cfg.get("seed", 42))
    top_k = int(runtime_cfg.get("top_k", 10))

    enable_profiler = bool(runtime_cfg.get("enable_profiler", False))
    enable_memory_snapshot = bool(runtime_cfg.get("enable_memory_snapshot", False))

    profiler_record_shapes = bool(runtime_cfg.get("profiler_record_shapes", True))
    profiler_profile_memory = bool(runtime_cfg.get("profiler_profile_memory", True))
    profiler_with_stack = bool(runtime_cfg.get("profiler_with_stack", False))
    profiler_with_flops = bool(runtime_cfg.get("profiler_with_flops", False))

    trace_filename = runtime_cfg.get("trace_filename", "trace.json")
    snapshot_filename = runtime_cfg.get("snapshot_filename", "memory_snapshot.pickle")
    compare_filename = runtime_cfg.get("compare_filename", "compare_report.md")

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
    tracer = RuntimeTracer(device=device)

    handles = register_runtime_hooks(
        model,
        tracer,
        hook_modules=True,
        hook_output_grads=True,
        hook_param_grads=True,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runtime_json_path = outdir / "runtime_report.json"
    runtime_md_path = outdir / "runtime_report.md"
    trace_path = outdir / trace_filename
    snapshot_path = outdir / snapshot_filename
    compare_path = outdir / compare_filename

    if device.startswith("cuda"):
        reset_peak_memory_stats(device)

    snapshot_started = start_memory_history(
        device=device,
        enabled=enable_memory_snapshot,
        max_entries=int(runtime_cfg.get("snapshot_max_entries", 100000)),
    )

    prof = None
    trace_exported = False
    snapshot_dumped = False

    profiler_ctx = build_profiler(
        enabled=enable_profiler,
        device=device,
        record_shapes=profiler_record_shapes,
        profile_memory=profiler_profile_memory,
        with_stack=profiler_with_stack,
        with_flops=profiler_with_flops,
    )

    try:
        with profiler_ctx as prof:
            for step in range(steps):
                tracer.set_step(step)

                B = full_cfg.train.micro_batch_size
                S = full_cfg.model.seq_len
                V = full_cfg.model.vocab_size

                input_ids = torch.randint(0, V, (B, S), device=device, dtype=torch.long)
                labels = torch.randint(0, V, (B, S), device=device, dtype=torch.long)

                synchronize_if_needed(device)
                before_step = memory_stats(device)
                tracer.record_step_boundary(
                    phase="step_start",
                    before=before_step,
                    after=before_step,
                    inputs=input_ids,
                    notes="training step start",
                )

                optimizer.zero_grad(set_to_none=True)

                synchronize_if_needed(device)
                before_forward = memory_stats(device)
                logits = model(input_ids)
                synchronize_if_needed(device)
                after_forward = memory_stats(device)
                tracer.record_step_boundary(
                    phase="forward_end",
                    before=before_forward,
                    after=after_forward,
                    outputs=logits,
                    notes="full forward done",
                )

                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                synchronize_if_needed(device)
                before_backward = memory_stats(device)
                loss.backward()
                synchronize_if_needed(device)
                after_backward = memory_stats(device)
                tracer.record_step_boundary(
                    phase="backward_end",
                    before=before_backward,
                    after=after_backward,
                    outputs=loss,
                    notes=f"full backward done; loss={loss.item():.6f}",
                )

                synchronize_if_needed(device)
                before_optim = memory_stats(device)
                optimizer.step()
                synchronize_if_needed(device)
                after_optim = memory_stats(device)
                tracer.record_step_boundary(
                    phase="optimizer_step_end",
                    before=before_optim,
                    after=after_optim,
                    notes="optimizer step done",
                )

                synchronize_if_needed(device)
                end_step = memory_stats(device)
                tracer.record_step_boundary(
                    phase="step_end",
                    before=before_step,
                    after=end_step,
                    outputs=loss.detach(),
                    notes="training step end",
                )

                if prof is not None:
                    prof.step()

        if enable_profiler and prof is not None:
            trace_exported = export_chrome_trace(prof, trace_path)

        if snapshot_started:
            synchronize_if_needed(device)
            snapshot_dumped = dump_memory_snapshot(
                device=device,
                outpath=snapshot_path,
                enabled=True,
            )

    finally:
        for h in handles:
            h.remove()

        if snapshot_started:
            stop_memory_history(device=device)

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
            "torch_version": getattr(torch, "__version__", "unknown"),
            "profiler_enabled": str(enable_profiler),
            "profiler_supported": str(profiler_supported()),
            "trace_exported": str(trace_exported),
            "trace_path": str(trace_path) if trace_exported else "",
            "memory_snapshot_enabled": str(enable_memory_snapshot),
            "memory_snapshot_supported": str(snapshot_supported(device)),
            "memory_snapshot_started": str(snapshot_started),
            "memory_snapshot_dumped": str(snapshot_dumped),
            "memory_snapshot_path": str(snapshot_path) if snapshot_dumped else "",
        },
        comparisons={
            "static_peak_memory_bytes": float(static_peak),
            "runtime_peak_memory_bytes": float(tracer._peak_allocated),
            "peak_diff_bytes": float(tracer._peak_allocated - static_peak),
            "peak_diff_ratio": float((tracer._peak_allocated - static_peak) / static_peak) if static_peak > 0 else 0.0,
        },
        top_k=top_k,
    )

    write_json_report(runtime_report, runtime_json_path)
    runtime_md_path.write_text(render_runtime_report(runtime_report), encoding="utf-8")

    print("MemScope Runtime Report")
    print(f"Device:             {device}")
    print(f"Device name:        {get_device_name(device)}")
    print(f"Peak allocated:     {format_bytes(runtime_report.peak.memory_bytes)}")
    print(f"Peak reserved:      {format_bytes(runtime_report.peak.reserved_bytes)}")
    print(f"Peak phase:         {runtime_report.peak.phase}")
    print(f"Peak module:        {runtime_report.peak.module}")
    print("")

    print(f"Top {len(runtime_report.top_events)} events by allocated-after:")
    for i, ev in enumerate(runtime_report.top_events, 1):
        print(
            f"{i:>2}. step={ev.step} "
            f"type={ev.event_type} "
            f"phase={ev.phase} "
            f"module={ev.module} "
            f"after={format_bytes(ev.mem_allocated_after)} "
            f"delta={format_bytes(ev.delta_allocated)}"
        )

    print("")
    print(f"Saved runtime report to:   {runtime_json_path}")
    print(f"Saved runtime markdown to: {runtime_md_path}")
    print(f"Saved compare report to:   {compare_path}")

    if enable_profiler:
        if trace_exported:
            print(f"Saved profiler trace to:   {trace_path}")
        else:
            print("Profiler enabled, but trace export failed or profiler unsupported.")

    if enable_memory_snapshot:
        if snapshot_dumped:
            print(f"Saved memory snapshot to:  {snapshot_path}")
        else:
            print("Memory snapshot enabled, but dump failed or snapshot unsupported.")


if __name__ == "__main__":
    main()