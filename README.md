```markdown
# MemScope (Phase 1) — Static GPU Memory Estimator

MemScope is a **static (rule-based) GPU memory estimator** for LLaMA-like transformer training.  
Given a model/training config, it produces a **memory breakdown** (weights/grad/optimizer/activations/temp/persistent) and an **operator-level report** with shapes, dtypes, byte sizes, and formulas.

> Status: **Phase 1 (Static Analysis MVP)** — produces `static_report.json` and `static_report.md`.

---

## Features (Phase 1)

- **One-command static analysis** from a JSON config
- **Summary memory breakdown**
  - `param_count`
  - `weight_memory_bytes`
  - `grad_memory_bytes`
  - `optimizer_memory_bytes`
  - `activation_memory_bytes`
  - `temporary_memory_bytes`
  - `persistent_memory_bytes`
  - `peak_memory_bytes` and `peak_stage`
- **Operator-level records**
  - `name`, `category`, `phase`
  - `inputs[]` / `outputs[]`: `shape`, `dtype`, `bytes`, `name`
  - `memory_bytes`, `persistent`, `formula`, `notes`, `extra`
- **Report formats**
  - Console
  - JSON (`static_report.json`) — source of truth for downstream tooling
  - Markdown (`static_report.md`) — human-readable snapshot

---

## Quick Start

### 1) Create/choose a config
Put a config JSON under `configs/`, for example:

- `configs/llama_toy.json`

### 2) Run static analysis
```bash
python scripts/analyze_static.py --config configs/llama_toy.json
```

### 3) Outputs
By default the script writes reports to `outputs/`:

- `outputs/static_report.json`
- `outputs/static_report.md`

---

## Output Schema (StaticReport)

### Top-level
- `summary`: global memory totals and peak info
- `operators`: list of operator/module records (static estimates)
- `metadata`: run metadata (e.g., mode/model_type/dtype)

### Example (abridged)
```json
{
  "summary": {
    "param_count": 375414784,
    "weight_memory_bytes": 750780416,
    "grad_memory_bytes": 750829568,
    "optimizer_memory_bytes": 6006243328,
    "activation_memory_bytes": 786432000,
    "temporary_memory_bytes": 819462144,
    "persistent_memory_bytes": 8507853312,
    "peak_memory_bytes": 9327315456,
    "peak_stage": "optimizer_step"
  },
  "operators": [
    {
      "name": "embedding",
      "category": "embedding",
      "phase": "forward",
      "inputs": [
        {"name":"token_ids","shape":[1,2048],"dtype":"int8","bytes":2048}
      ],
      "outputs": [
        {"name":"embeddings","shape":[2048,1,8192],"dtype":"bf16","bytes":33554432}
      ],
      "memory_bytes": 33554432,
      "persistent": false,
      "formula": "2BSH",
      "notes": "BF16 embeddings",
      "extra": {}
    }
  ],
  "metadata": {
    "mode": "static",
    "model_type": "llama",
    "dtype": "bf16"
  }
}
```

---

## Project Structure (Phase 1)

Typical layout:

- `scripts/`
  - `analyze_static.py` — CLI entrypoint for static estimation
- `memscope/`
  - `parsers/`
    - `config_loader.py` — load/validate config JSON
  - `schemas/`
    - `config.py` — config dataclasses
    - `op.py` — `OpRecord` and tensor record schema
    - `report.py` — `StaticReport`, `Summary`, metadata schema
  - `static/`
    - `shape_infer.py` — static shape inference + op generation
    - `formulas.py` — byte formulas and helpers
  - `report/`
    - `console_reporter.py`
    - `json_reporter.py`
    - `markdown_reporter.py`

> The JSON report is intended to be stable and machine-consumable; Markdown/console are views.

---

## Assumptions & Limitations (Phase 1)

This version is **static** and **rule-based**. It does **not** measure real runtime allocations.

Common limitations:
- Memory is estimated from shapes/dtypes and simplified lifecycle assumptions.
- Peak stage is identified at a coarse granularity (e.g., `"optimizer_step"`), and may not match exact runtime peaks.
- Operator list may represent **aggregated** operations (some formulas may include an `L` multiplier for number of layers) rather than a fully unrolled per-layer trace.

---

## Roadmap

### Phase 2 — Runtime Tracing (Validation)
- Hook into a real training step (or a forward/backward pass)
- Record runtime memory (`torch.cuda.memory_allocated()` / peak stats)
- Produce `runtime_report.json`
- Compare static vs runtime errors and calibrate rules

### Phase 3 — Peak Composition / Memory Snapshots
- Identify **what tensors contribute to peak**
- Optional integration with CUDA memory snapshots / profiler traces
- Better lifecycle modeling and per-stage/per-op peak breakdown