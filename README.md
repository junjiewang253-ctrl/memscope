# 项目简介
MemScope 是一个面向大模型训练场景的显存分析工具，目标不是只给出一个“峰值显存”数字，而是尽可能把训练过程里的显存变化变成可追踪、可解释、可复现的事件序列。它既支持基于配置的静态显存估算，也支持在真实训练过程中做 runtime tracing，记录 step 边界、模块前向、反向梯度以及关键显存峰值。

这个仓库最初是围绕 Transformer/LLaMA 类模型的显存分析做原型实现，后来我把它进一步集成进了 FlagScale + Megatron-LM 的训练流程里。这样它就不再只是一个独立的小工具，而是可以直接插入真实分布式训练任务，在多卡环境下输出每个 rank 的显存报告。

目前已经支持在 Megatron/FlagScale 训练中记录：
训练 step 的显存边界、模块级 forward 事件、output tensor grad、parameter grad、runtime peak，以及 static 与 runtime 的对比结果。输出格式为 JSON 和 Markdown，后续也会补充更直观的 HTML 可视化。

## 目前已经验证的环境
我目前主要在下面这个环境里验证：

Ubuntu 远程服务器
8 × NVIDIA A100-SXM4-80GB
容器镜像：nvcr.io/nvidia/pytorch:25.12-py3
训练框架：FlagScale + Megatron-LM
模型配置：Qwen 风格 GPT 训练配置，支持 TP / PP / DP 组合并行

## 这个项目能做什么
和传统只看 nvidia-smi 或只抓一个 profiler trace 不一样，MemScope 更关注“训练中的显存事件到底是谁触发的”。

它可以回答这样一些问题：
一个 step 里显存是在 forward 还是 backward 到达峰值
哪些模块在前向过程中产生了明显的显存增量
哪些参数梯度在 backward 里是真正的大头
当前 rank 上的局部模型在 TP / PP 配置下呈现出什么 shape
静态估算和真实 runtime 之间差了多少
如果你在调大模型训练里的 OOM、显存不均衡、局部 rank 异常高峰，或者只是想更清楚地理解 Megatron 下每个 rank 实际发生了什么，这类信息会比单纯的峰值数字更有帮助。

## 如何在 FlagScale + Megatron-LM 中复现
我这版集成方式是基于 FlagScale 的 runner 机制完成的。FlagScale 会把 train.system / train.model / train.data 里的配置 flatten 成命令行参数，然后交给 flagscale/train/train_gpt.py。因此，MemScope 参数是通过 extra_args_provider 注册进 Megatron argparse 的，而不是去直接修改 Megatron 原生 arguments 入口。(在..<你自己的FlagScale根目录>/FlagScale/flagscale/train/train_gpt.py)
### 在 flagscale/train/train_gpt.py 注册 memscope 参数
我是这么做的：
**在 train_gpt.py 顶部增加 import**
加上：
```python
import argparse
```

**新增参数注册函数**
放在 if __name__ == "__main__": 之前：
```python
def add_memscope_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="memscope")

    group.add_argument(
        "--memscope",
        action="store_true",
        help="Enable MemScope runtime tracing during training.",
    )
    group.add_argument(
        "--memscope-outdir",
        type=str,
        default="memscope_outputs",
        help="Output directory for MemScope reports.",
    )
    group.add_argument(
        "--memscope-top-k",
        type=int,
        default=20,
        help="Top-K events kept in MemScope runtime report.",
    )

    group.add_argument(
        "--memscope-hook-modules",
        action="store_true",
        help="Enable module forward hooks in MemScope.",
    )
    group.add_argument(
        "--memscope-hook-output-grads",
        action="store_true",
        help="Enable output tensor grad hooks in MemScope.",
    )
    group.add_argument(
        "--memscope-hook-param-grads",
        action="store_true",
        help="Enable parameter grad hooks in MemScope.",
    )

    group.add_argument(
        "--memscope-enable-profiler",
        action="store_true",
        help="Enable torch profiler in MemScope.",
    )
    group.add_argument(
        "--memscope-enable-memory-snapshot",
        action="store_true",
        help="Enable CUDA allocator memory snapshot in MemScope.",
    )
    group.add_argument(
        "--memscope-profiler-ranks",
        nargs="+",
        type=int,
        default=[0],
        help="Global ranks on which MemScope profiler is enabled.",
    )
    group.add_argument(
        "--memscope-snapshot-ranks",
        nargs="+",
        type=int,
        default=[0],
        help="Global ranks on which MemScope snapshot is enabled.",
    )

    group.add_argument(
        "--memscope-sync-on-step-boundaries",
        action="store_true",
        help="Synchronize CUDA at MemScope step boundaries.",
    )
    group.add_argument(
        "--memscope-sync-on-module-hooks",
        action="store_true",
        help="Synchronize CUDA at module hooks. Slower but more accurate.",
    )

    return parser
```

**合并已有的 add_modelopt_args**
继续加一个组合函数：
```python
def extra_args_provider_with_memscope(parser: argparse.ArgumentParser):
    parser = add_memscope_args(parser)

    # 让 hook 系列默认开启：由于 FlagScale bool flatten 只会在 true 时传参，
    # 所以这里通过 set_defaults 提供默认值最稳妥
    parser.set_defaults(
        memscope_hook_modules=True,
        memscope_hook_output_grads=True,
        memscope_hook_param_grads=True,
        memscope_sync_on_step_boundaries=True,
        memscope_sync_on_module_hooks=False,
    )

    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)
    return parser
```

**修改 pretrain(...) 调用**
原来是：
```python
extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
```
改成：
```python
extra_args_provider=extra_args_provider_with_memscope,
```

使用时主要需要做三件事：

第一，把 memscope 仓库放进训练环境里，保证训练进程可以 import 到 memscope.integrations.megatron.runtime。最简单的办法就是把仓库放在 FlagScale workspace 下，或者用 editable install。

第二，在 flagscale/train/train_gpt.py 中通过 extra_args_provider 注册 --memscope 及相关参数，例如输出目录、top-k、是否开启 profiler、是否开启 memory snapshot 等。

第三，在 flagscale/train/train.py 的训练主循环里，在 train() 和 train_step() 中插入 runtime 生命周期与 step 边界回调。这样训练开始时会 attach hooks，训练过程中会记录事件，结束时会生成每个 rank 的报告。
### 在 flagscale/train/train.py 接入 MemScope runtime
这是运行时采集核心。
**顶部 import**
在 flagscale/train/train.py import 区加：
```python
try:
    from memscope.integrations.megatron.runtime import MemScopeMegatronRuntime
    HAVE_MEMSCOPE = True
except ImportError:
    MemScopeMegatronRuntime = None
    HAVE_MEMSCOPE = False
```

**修改 train_step(...) 签名**
原来：
```python
def train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
```
改成：
```python
def train_step(
    forward_step_func,
    data_iterator,
    model,
    optimizer,
    opt_param_scheduler,
    config,
    memscope_runtime=None,
    iteration=None,
):
```

#### 在 train_step(...) 中插埋点
**在 forward_backward_func 调用前**
找到：
```python
# Forward pass.
forward_backward_func = get_forward_backward_func()
```
下面插入：
```python
if memscope_runtime is not None:
    memscope_runtime.on_train_step_start(
        step=iteration if iteration is not None else 0,
        batch=None,
        notes="flagscale train_step start",
    )
```

**在 forward_backward_func(...) 返回后插入**
在：
```python
losses_reduced = forward_backward_func(
    ...
)
```
后面加：
```python
if memscope_runtime is not None:
    memscope_runtime.on_forward_backward_end(
        outputs=losses_reduced,
        notes="flagscale forward_backward_func complete",
    )
```

**在 optimizer.step() 前后加**
找到：
```python
timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
timers('optimizer').stop()
```
改成：
```python
if memscope_runtime is not None:
        memscope_runtime.on_optimizer_step_start(
            notes="optimizer.step about to run"
        )

timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
timers('optimizer').stop()

if memscope_runtime is not None:
    memscope_runtime.on_optimizer_step_end(
        notes="optimizer.step complete"
    )
```

**在 early exit 前加**
找到：
```python
if should_exit:
    return {}, True, should_checkpoint, should_exit, exit_code, None, None
```
改成：
```python
if should_exit:
    if memscope_runtime is not None:
        memscope_runtime.on_train_step_end(
            outputs=None,
            notes="train_step early exit",
        )
    return {}, True, should_checkpoint, should_exit, exit_code, None, None
```

**在 pipeline last stage return 前加**
找到：
```python
return (
    loss_reduced,
    skipped_iter,
    should_checkpoint,
    should_exit,
    exit_code,
    grad_norm,
    num_zeros_in_grad,
)
```
改成：
```python
if memscope_runtime is not None:
    memscope_runtime.on_train_step_end(
        outputs=loss_reduced,
        notes="train_step end (last pipeline stage)",
    )

return (
    loss_reduced,
    skipped_iter,
    should_checkpoint,
    should_exit,
    exit_code,
    grad_norm,
    num_zeros_in_grad,
)
```

**在非 last stage return 前加**
找到最后：
```python
return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad
```
改成：
```python
if memscope_runtime is not None:
    memscope_runtime.on_train_step_end(
        outputs=None,
        notes="train_step end",
    )

return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad
```

### 在 train(...) 中初始化与结束 runtime
**训练开始时初始化**
在 train() 里，找到：
```python
# Turn on training mode which enables dropout.
for model_module in model:
    model_module.train()
```
下面加：
```python
memscope_runtime = None
if getattr(args, "memscope", False):
    if HAVE_MEMSCOPE:
        try:
            memscope_runtime = MemScopeMegatronRuntime.from_megatron_args(args)
            memscope_runtime.attach_model(model)
            memscope_runtime.start()
            print_rank_0(
                f"> MemScope enabled. Reports will be written to {args.memscope_outdir}"
            )
        except Exception as e:
            print_rank_0(f"> WARNING: failed to initialize MemScope: {e}")
            memscope_runtime = None
    else:
        print_rank_0(
            "> WARNING: args.memscope is enabled but memscope package is not importable."
        )
```

**调用 train_step 时传入 runtime**
找到：
```python
        ) = train_step(
            forward_step_func, train_data_iterator, model, optimizer, opt_param_scheduler, config
        )
```
改成：
```python
        ) = train_step(
            forward_step_func,
            train_data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            memscope_runtime=memscope_runtime,
            iteration=iteration,
        )
```

**在 train() 正常结束前 finalize**
在最后：
```python
    return iteration, num_floating_point_operations_so_far
```
前面加：
```python
    if memscope_runtime is not None:
        try:
            memscope_runtime.finalize()
            print_rank_0("> MemScope finalize complete.")
        except Exception as e:
            print_rank_0(f"> WARNING: MemScope finalize failed: {e}")
```

**在 should_exit 分支也 finalize**
找到：
```python
    if should_exit:
        wandb_writer = get_wandb_writer()
        if wandb_writer:
            wandb_writer.finish()
        ft_integration.shutdown()
        one_logger_utils.finish()
        sys.exit(exit_code)
```
改成：
```python
    if should_exit:
        if memscope_runtime is not None:
            try:
                memscope_runtime.finalize()
            except Exception:
                pass

        wandb_writer = get_wandb_writer()
        if wandb_writer:
            wandb_writer.finish()
        ft_integration.shutdown()
        one_logger_utils.finish()
        sys.exit(exit_code)
```

配置文件里只需要在 train.system 下(也就是例如14b.yaml的system: 下)增加类似下面的字段：
memscope: true
memscope_outdir: ${experiment.exp_dir}/memscope
memscope_top_k: 20
memscope_hook_modules: true
memscope_hook_output_grads: true
memscope_hook_param_grads: true
memscope_enable_profiler: false
memscope_enable_memory_snapshot: false
memscope_profiler_ranks: [0]
memscope_snapshot_ranks: [0]
memscope_sync_on_step_boundaries: true
memscope_sync_on_module_hooks: false

训练完成后，报告会输出到类似下面的目录：
output_xxx/memscope/rank00000/runtime_report.json
output_xxx/memscope/rank00000/runtime_report.md
如果是多卡训练，每个 rank 都会各自输出一个目录。

## 输出报告怎么看
runtime report 里最重要的部分有四个：runtime_trace、peak、top_events、comparisons。

runtime_trace 是完整的事件时间线，包含 step、module forward、tensor grad 等事件。
peak 是当前 rank 的显存峰值快照。
top_events 是按显存占用排序的关键事件摘要，通常很适合直接定位“谁最重”。
comparisons 会给出 static estimate 和 runtime peak 的差异，方便判断静态模型是否过于保守，或者 runtime 是否出现了额外开销。

## 后续计划
接下来我会继续把这个项目往两个方向补强。一个方向是可视化，把 runtime report 直接渲染成更容易阅读的 HTML 页面；另一个方向是现在的静态显存估算公式偏保守估计，所以输出的结果会比动态分析的结果相差较多，建议先以动态报告为准，我后续会把 static estimator 做得更贴近真实的 Megatron 配置，尤其是对并行切分、激活生命周期和优化器状态的估算。

如果你对大模型训练显存分析、Megatron 训练调试或者 profiling 工具感兴趣，欢迎交流或提 issue。