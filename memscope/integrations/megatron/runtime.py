from __future__ import annotations

from pathlib import Path

from memscope.integrations.megatron.args_loader import megatron_args_to_full_config
from memscope.integrations.megatron.hooks import register_megatron_runtime_hooks
from memscope.report.json_reporter import write_json_report
from memscope.report.runtime_markdown_reporter import render_runtime_report
from memscope.runtime.memory import (
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

class MemScopeMegatronRuntime:
    """
    [核心类] Megatron 运行时监控器
    作为 Facade 模式，统一管理 Hook、Profiler、Snapshot 和报告生成。
    """
    def __init__(self, args, outdir: str):
        """
        初始化监控器
        :param args: Megatron 的 argparse.Namespace 对象
        :param outdir: 输出目录根路径
        """
        self.args = args

        # [分布式信息] 获取当前 Rank ID 和总卡数
        self.rank = int(getattr(args, "rank", 0))
        self.world_size = int(getattr(args, "world_size", 1))

        # [设备定位] 确定当前进程绑定的 CUDA 设备
        # Megatron 通常使用 local_rank 来绑定具体显卡
        self.device = f"cuda:{getattr(args, 'local_rank', 0)}"

        # [开关控制] 检查是否开启了 memscope 功能 (--memscope)
        self.enabled = bool(getattr(args, "memscope", False))

        # [路径设置] 设置输出目录，并为当前 Rank 创建独立子目录
        self.outdir = Path(outdir)
        self.rank_outdir = self.outdir / f"rank{self.rank:05d}"
        self.rank_outdir.mkdir(parents=True, exist_ok=True)

        self.top_k = int(getattr(args, "memscope_top_k", 20))
        self.hook_modules = bool(getattr(args, "memscope_hook_modules", True))
        self.hook_output_grads = bool(getattr(args, "memscope_hook_output_grads", True))
        self.hook_param_grads = bool(getattr(args, "memscope_hook_param_grads", True))

        # [同步策略] 是否在关键节点强制 GPU 同步
        self.sync_on_step_boundaries = bool(
            getattr(args, "memscope_sync_on_step_boundaries", True)
        )
        self.sync_on_module_hooks = bool(
            getattr(args, "memscope_sync_on_module_hooks", False)
        )

        # [Profiler 控制] 仅在特定 Rank 上开启 Profiler (避免所有卡都开导致性能剧降)
        # 逻辑：全局开关为真 AND 当前 rank 在允许的列表中 (默认只开 rank 0)
        self.enable_profiler = (
            bool(getattr(args, "memscope_enable_profiler", False)) 
            and self.rank in getattr(args, "memscope_profiler_ranks", [0])
        )

        # [Snapshot 控制] 同理，仅在特定 Rank 开启显存快照
        self.enable_memory_snapshot = (
            bool(getattr(args, "memscope_enable_memory_snapshot", False)) 
            and self.rank in getattr(args, "memscope_snapshot_ranks", [0])
        )

        # [组件初始化]
        self.tracer = RuntimeTracer(device=self.device) # 核心数据记录器
        self.handles = []       # 存储 Hook 句柄，用于后续移除
        self.prof = None        # PyTorch Profiler 实例
        self.prof_ctx = None    # Profiler 上下文管理器
        self.prof_started = False
        self.snapshot_started = False

    @classmethod
    def from_megatron_args(cls, args):
        """
        [工厂方法] 便捷构造函数
        自动从 args 中提取输出目录，简化调用
        """
        outdir = getattr(args, "memscope_outdir", "memscope_outputs")
        return cls(args=args, outdir=outdir)
    
    def attach_model(self, model):
        """
        [阶段 1] 模型挂载
        在模型构建完成后，立即注入 Hook
        """
        if not self.enabled:
            return 
        self.handles = register_megatron_runtime_hooks(
            model, 
            self.tracer, 
            hook_modules=self.hook_modules, 
            hook_output_grads=self.hook_output_grads, 
            hook_param_grads=self.hook_param_grads, 
        )

    def start(self):
        """
        [阶段 2] 训练前启动
        在训练循环开始前调用，重置统计并启动 Profiler/Snapshot
        """
        if not self.enabled:
            return 
        
        reset_peak_memory_stats(self.device)

        self.snapshot_started = start_memory_history(
            device=self.device, 
            enabled=self.enable_memory_snapshot, 
            max_entries=100000, 
        )

        self.prof_ctx = build_profiler(
            enabled=self.enable_profiler, 
            device=self.device,
            record_shapes=True, 
            profile_memory=True, 
            with_stack=False, 
            with_flops=False, 
        )
        self.prof = self.prof_ctx.__enter__()
        self.prof_started = True

    def _maybe_sync(self):
        """[工具] 根据配置决定是否同步 GPU"""
        if self.sync_on_step_boundaries:
            synchronize_if_needed(self.device)

    def on_train_step_start(self, step: int, batch=None, notes: str = ""):
        """
        [回调] 训练步开始
        对应 Megatron 训练循环的最外层
        """
        if not self.enabled:
            return
        self.tracer.set_step(step)
        self._maybe_sync()
        stats = memory_stats(self.device)
        self.tracer.record_step_boundary(
            phase="step_start", 
            before=stats, 
            after=stats, 
            inputs=batch, 
            notes=notes or "megatron training step start", 
        )

    def on_forward_backward_end(self, outputs=None, notes: str=""):
        """
        [回调] 前向+反向传播结束
        此时激活值梯度已计算完成，显存可能处于高位
        """
        if not self.enabled:
            return 
        self._maybe_sync()
        stats = memory_stats(self.device)
        self.tracer.record_step_boundary(
            phase="forward_backward_end", 
            before=stats, 
            after=stats, 
            outputs=outputs, 
            notes=notes or "magatron forward_backward_func done", 
        )

    def on_optimizer_step_start(self, notes: str=""):
        """[回调] 优化器更新开始"""
        if not self.enabled:
            return 
        self._maybe_sync()
        stats = memory_stats(self.device)
        self.tracer.record_step_boundary(
            phase="optimizer_step_start", 
            before=stats, 
            after=stats, 
            notes=notes or "optimizer step start", 
        )

    def on_optimizer_step_end(self, notes: str = ""):
        """[回调] 优化器更新结束"""
        if not self.enabled:
            return
        self._maybe_sync()
        stats = memory_stats(self.device)
        self.tracer.record_step_boundary(
            phase="optimizer_step_end", 
            before=stats, 
            after=stats, 
            notes=notes or "optimizer step end",
        )

    def on_train_step_end(self, outputs=None, notes: str = ""):
        """
        [回调] 训练步完全结束
        通知 Profiler 一个 Step 结束，以便进行窗口分析
        """
        if not self.enabled:
            return
        self._maybe_sync()
        stats = memory_stats(self.device)
        self.tracer.record_step_boundary(
            phase="step_end", 
            before=stats, 
            after=stats, 
            outputs=outputs, 
            notes=notes or "training step end",
        )
        # 通知 Profiler 步进
        if self.prof is not None:
            self.prof.step()

    def finalize(self):
        """
        [阶段 3] 收尾与报告生成
        训练结束后调用，停止监控，导出数据，生成报告
        """
        if not self.enabled:
            return 
        trace_exported = False
        snapshot_dumped = False

        try:
            if self.prof_started and self.prof_ctx is not None:
                self.prof_ctx.__exit__(None, None, None)
                if self.prof is not None and self.enable_profiler:
                    trace_exported = export_chrome_trace(
                        self.prof, self.rank_outdir / "trace.json"
                    )
        finally:
            for h in self.handles:
                try: 
                    h.remove()
                except Exception:
                    pass
        
        if self.snapshot_started:
            synchronize_if_needed(self.device)
            snapshot_dumped = dump_memory_snapshot(
                device=self.device, 
                outpath=self.rank_outdir / "memory_snapshot.pickle", 
                enabled=True, 
            )
            stop_memory_history(device=self.device)

        full_cfg = megatron_args_to_full_config(self.args)
        static_report = analyze_static(full_cfg)
        static_peak = static_report.summary.peak_memory_bytes

        runtime_report = self.tracer.build_report(
            metadata={
                "mode": "runtime",
                "rank": str(self.rank),
                "world_size": str(self.world_size),
                "device": self.device,
                "device_name": get_device_name(self.device),
                "dtype": "bf16" if getattr(self.args, "bf16", False) else (
                    "fp16" if getattr(self.args, "fp16", False) else "fp32"
                ),
                "tensor_model_parallel_size": str(getattr(self.args, "tensor_model_parallel_size", 1)),
                "pipeline_model_parallel_size": str(getattr(self.args, "pipeline_model_parallel_size", 1)),
                "data_parallel_size": str(getattr(self.args, "data_parallel_size", 1)),
                "use_distributed_optimizer": str(getattr(self.args, "use_distributed_optimizer", False)),
                "optimizer": str(getattr(self.args, "optimizer", "adam")),
                "profiler_enabled": str(self.enable_profiler),
                "profiler_supported": str(profiler_supported()),
                "trace_exported": str(trace_exported),
                "memory_snapshot_enabled": str(self.enable_memory_snapshot),
                "memory_snapshot_supported": str(snapshot_supported(self.device)),
                "memory_snapshot_started": str(self.snapshot_started),
                "memory_snapshot_dumped": str(snapshot_dumped),
            },
            comparisons={
                "static_peak_memory_bytes": float(static_peak),
                "runtime_peak_memory_bytes": float(self.tracer._peak_allocated),
                "peak_diff_bytes": float(self.tracer._peak_allocated - static_peak),
                "peak_diff_ratio": float(
                    (self.tracer._peak_allocated - static_peak) / static_peak
                ) if static_peak > 0 else 0.0,
            },
            top_k=self.top_k,
        )

        write_json_report(runtime_report, self.rank_outdir / "runtime_report.json")
        (self.rank_outdir / "runtime_report.md").write_text(
            render_runtime_report(runtime_report), 
            encoding="utf-8", 
        )