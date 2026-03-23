```markdown
# MemScope — Phase 2 (Runtime Tracing MVP)

Phase 2 focuses on **measuring real GPU memory during a training step**.  
Instead of estimating from formulas, it runs a small model, hooks into forward/backward, and writes a `runtime_report.json` you can use to answer:

- peak memory happens **when** (forward / backward / optimizer step)
- memory jumps happen **where** (which module / boundary event)
- the biggest tensors are **what** (outputs / grads by shape + dtype + bytes)
- how `allocated` compares to `reserved` (allocator cache / fragmentation signals)

This phase is intentionally simple: module-level sampling + tensor grad hooks. It’s not a full CUDA allocator trace.

## Quick start

### 1) Pick a config
Model + train config (JSON), e.g.:

- `configs/llama_runtime_toy.json`

Runtime environment config (YAML), e.g.:

- `configs/runtime_toy.yaml`

Example:

```bash
python scripts/analyze_runtime.py \
  --config configs/llama_runtime_toy.json \
  --runtime-config configs/runtime_toy.yaml \
  --outdir outputs
```

### 2) Outputs

- `outputs/runtime_report.json` — source of truth for analysis/visualization
- `outputs/runtime_report.md` — quick human-readable snapshot

## What gets traced

During each step the script records boundary events:

- `step_start`
- `forward_end`
- `backward_end`
- `optimizer_step_end`
- `step_end`

It also records:

- `module` events via `forward_pre_hook` and `forward_hook` for “interesting” modules (Embedding / Linear / RMSNorm / Attention / MLP / Block / LM head, etc.)
- `tensor_grad` events by registering hooks on module outputs (and on parameters) to capture gradient tensor metadata

For every event we save CUDA memory stats:

- `mem_allocated_before/after`
- `mem_reserved_before/after`
- `delta_allocated/delta_reserved`
- `max_mem_allocated/max_mem_reserved`

Plus lightweight tensor metadata:

- `shape`, `dtype`, `device`, `requires_grad`, `bytes`

## How to read `runtime_report.json`

Top-level fields:

- `runtime_trace`: ordered event list for the whole step
- `peak`: peak `allocated` and `reserved` observed at event boundaries
- `top_events`: events with the highest `mem_allocated_after` (useful for “what was happening near the peak”)
- `metadata`: device / dtype / optimizer / etc.
- `comparisons`: optional (only used if you also run static; safe to ignore in Phase 2)

Important notes:

- Peak is computed from the events we sample. If a big temporary allocation happens *inside* a module and is freed before the module hook returns, we may miss that spike.
- `reserved` can stay high even if `allocated` drops; that’s expected with PyTorch’s caching allocator.

## Limitations (by design)

- Not operator-level: we trace module boundaries, not every matmul/softmax kernel.
- Not a full “peak composition”: we can tell you where the peak occurred and what large tensors exist, but we don’t reconstruct exact live tensor sets at the peak.
- Numbers are allocator-level (`torch.cuda.memory_allocated/reserved`), not raw CUDA allocations.

## Tips

- If you want cleaner measurements, run with a fresh process and keep `steps: 1` while iterating.
- When comparing runs, watch both `allocated` and `reserved`. A change that reduces `allocated` but increases `reserved` can still matter for long runs.