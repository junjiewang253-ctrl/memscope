# Markdown: 文档可读
from memscope.utils import format_bytes

def render_markdown_report(report) -> str:
    """
    将分析报告渲染为 Markdown 格式的字符串。
    
    Returns:
        str: 完整的 Markdown 文本，可直接写入 .md 文件或打印。
    """
    s = report.summary

    lines = []
    lines.append("# MemScope Static Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Param count: `{s.param_count}`")
    lines.append(f"- Weight memory: `{format_bytes(s.weight_memory_bytes)}`")
    lines.append(f"- Grad memory: `{format_bytes(s.grad_memory_bytes)}`")
    lines.append(f"- Optimizer memory: `{format_bytes(s.optimizer_memory_bytes)}`")
    lines.append(f"- Activation memory: `{format_bytes(s.activation_memory_bytes)}`")
    lines.append(f"- Temporary memory: `{format_bytes(s.temporary_memory_bytes)}`")
    lines.append(f"- Persistent memory: `{format_bytes(s.persistent_memory_bytes)}`")
    lines.append(f"- Peak memory: `{format_bytes(s.peak_memory_bytes)}`")
    lines.append(f"- Peak stage: `{s.peak_stage}`")
    lines.append("")
    lines.append("## Operators")
    lines.append("")
    lines.append("| Name | Category | Phase | Memory | Formula |")
    lines.append("|---|---|---|---:|---|")
    for op in report.operators:
        lines.append(
            f"| {op.name} | {op.category} | {op.phase} | {format_bytes(op.memory_bytes)} | {op.formula} |"
        )
    lines.append("")
    return "\n".join(lines)