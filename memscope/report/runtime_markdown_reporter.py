from memscope.utils import format_bytes


def render_runtime_report(report) -> str:
    lines = []
    lines.append("# MemScope Runtime Report")
    lines.append("")
    lines.append("## Peak")
    lines.append("")
    lines.append(f"- Peak allocated: `{format_bytes(report.peak.memory_bytes)}`")
    lines.append(f"- Peak reserved: `{format_bytes(report.peak.reserved_bytes)}`")
    lines.append(f"- Peak phase: `{report.peak.phase}`")
    lines.append(f"- Peak module: `{report.peak.module}`")
    lines.append(f"- Peak step: `{report.peak.step}`")
    lines.append("")

    if report.comparisons:
        lines.append("## Static vs Runtime")
        lines.append("")
        for k, v in report.comparisons.items():
            if "bytes" in k:
                lines.append(f"- {k}: `{int(v)}`")
            else:
                lines.append(f"- {k}: `{v}`")
        lines.append("")

    lines.append("## Top Events")
    lines.append("")
    lines.append("| Step | Type | Module | Phase | After Alloc | Delta Alloc | Notes |")
    lines.append("|---:|---|---|---|---:|---:|---|")

    for ev in report.top_events:
        lines.append(
            f"| {ev.step} | {ev.event_type} | {ev.module} | {ev.phase} | "
            f"{format_bytes(ev.mem_allocated_after)} | {format_bytes(ev.delta_allocated)} | {ev.notes} |"
        )

    lines.append("")
    return "\n".join(lines)