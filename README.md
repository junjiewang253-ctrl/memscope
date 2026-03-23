```markdown
# MemScope — Phase 2 (Runtime Tracing + Profiler/Snapshot)

这一版的重点是把 **runtime tracing** 做成一个更完整的“采集套件”：  
除了原来的 hooks + `memory_allocated/reserved` 时间线，又加了两条可选能力：

- **PyTorch Profiler**：导出 Chrome Trace（`trace.json`）
- **CUDA allocator memory snapshot**：导出 `memory_snapshot.pickle`（用于事后做 allocator 级分析）

整体目标很明确：先把“训练一步里显存怎么涨、峰值在哪、跟哪些模块/阶段相关”抓出来；必要时再用 profiler/allocator snapshot 深挖。

---

## Quick start

### 1) 配置文件

**模型/训练配置（JSON）**：定义 toy 模型结构与训练超参  
例：`configs/llama_runtime_toy.json`

**运行时配置（YAML）**：控制设备、步数、以及是否开启 profiler / snapshot  
例：`configs/runtime_toy.yaml`

`runtime_toy.yaml` 里新增的关键开关：

- `enable_profiler: true|false`
- `enable_memory_snapshot: true|false`

以及 profiler/snapshot 的参数（record_shapes、profile_memory、max_entries、输出文件名等）。

---

### 2) 运行

```bash
python scripts/analyze_runtime.py \
  --config configs/llama_runtime_toy.json \
  --runtime-config configs/runtime_toy.yaml \
  --outdir outputs
```

---

## Outputs

默认输出到 `--outdir`（默认 `outputs/`）：

- `runtime_report.json`：主要产物，包含事件时间线、peak、top events、metadata
- `runtime_report.md`：便于快速浏览的 Markdown 摘要

如果开启 profiler：

- `trace.json`（文件名可在 YAML 里通过 `trace_filename` 改）  
  用 Chrome 打开：`chrome://tracing` 或 Perfetto 导入查看

如果开启 memory snapshot 且 torch/环境支持：

- `memory_snapshot.pickle`（文件名可通过 `snapshot_filename` 改）

> 注意：memory snapshot 依赖 PyTorch 的私有接口 `torch.cuda.memory._record_memory_history/_dump_snapshot`，不同 torch 版本/平台不保证可用。脚本会在 metadata 里写明是否 supported/started/dumped。

---

## What gets traced (runtime_report.json)

这一版 runtime tracing 仍然以“训练一步”为单位，记录几类事件：

### Step boundary 事件（粗粒度阶段）
每一步会记录：

- `step_start`
- `forward_end`
- `backward_end`
- `optimizer_step_end`
- `step_end`

这些事件非常适合做阶段对比：forward/backward/optim 各自让 `allocated` 和 `reserved` 变化多少。

### Module 事件（模块级采样）
通过 `forward_pre_hook` / `forward_hook` 对“interesting modules”采样，默认按 class name 过滤：

- Embedding / Linear / (RMS)Norm / Attention / MLP / Block / LM head 等

每个 module 事件会带：

- `mem_allocated_before/after`, `mem_reserved_before/after`
- `delta_allocated/delta_reserved`
- 输入/输出张量的元信息（shape/dtype/device/bytes）

### Grad 事件（反向传播时的 tensor hook）
在 module forward hook 中，对输出 tensor 注册 `register_hook`，以及对参数注册 grad hook：

- 记录梯度 tensor 的 shape/dtype/bytes
- 方便回答“最大的梯度/输出在哪里”

---

## Read the numbers correctly

### allocated vs reserved
- `allocated`：PyTorch 当前实际分配并在用的显存
- `reserved`：PyTorch caching allocator 向驱动申请并保留的显存（可能包含碎片/缓存）

很多时候 `reserved` 不会随着 `allocated` 下降而立即下降，这是正常现象。

### Peak 的含义
`runtime_report.peak` 是在**事件采样点**上观测到的最大 `allocated/reserved`。  
如果某个瞬时峰值发生在模块内部、且在 hook 返回前已经释放，有可能不会被 module 边界采到。

遇到这种情况：
- 用 profiler trace 看 kernel 级别的 memory timeline（如果 profile_memory 开了）
- 或用 allocator snapshot 做更底层的定位

---

## Runtime config reference (configs/runtime_toy.yaml)

常用字段：

- `device`: `cuda` / `cuda:0` / `cpu`
- `steps`: 运行多少个训练 step（建议从 1 开始）
- `lr`, `seed`, `top_k`

Profiler 相关：

- `enable_profiler`
- `profiler_record_shapes`
- `profiler_profile_memory`
- `profiler_with_stack`
- `profiler_with_flops`
- `trace_filename`

Memory snapshot 相关：

- `enable_memory_snapshot`
- `snapshot_max_entries`
- `snapshot_filename`

---

## Limitations

- 这是 module 级采样，不是逐算子/逐 kernel 的精确峰值重建。
- memory snapshot 使用 PyTorch 私有 API，可能在不同版本不兼容。
- 当前 runtime 脚本仍然会生成 static report 并写入 comparisons（为了对比），但如果你只关心 runtime，可以忽略 comparisons 字段。

---

## Repo layout (relevant files)

- `scripts/analyze_runtime.py`：runtime 主入口（hooks + profiler + snapshot + 输出）
- `memscope/runtime/`
  - `hooks.py`：注册 module hooks / grad hooks
  - `memory.py`：`memory_stats()`、同步、peak reset
  - `tracer.py`：事件记录与报告打包
  - `profiler.py`：torch.profiler 封装与 trace 导出
  - `snapshot.py`：CUDA allocator history + snapshot 导出
- `memscope/report/runtime_markdown_reporter.py`：runtime markdown 渲染
- `configs/`：示例配置

---

## Tips

- 做对比实验时尽量每次用新进程跑（PyTorch allocator 会缓存 reserved，影响复现实验）。
- 如果你关心瞬时尖峰，优先打开 profiler 的 `profile_memory`；如果还不够，再考虑 snapshot。