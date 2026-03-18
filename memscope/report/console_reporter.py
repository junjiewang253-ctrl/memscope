# Console: 终端交互式可读
# 引入 Rich 库的核心组件
# Console: 终端输出控制器
# Table: 漂亮的表格组件
from rich.console import Console
from rich.table import Table
from memscope.utils import format_bytes

def print_static_report(report):
    console = Console()
    s = report.summary

    console.print("[bold cyan]MemScope Static Report[/bold cyan]")
    console.print(f"Param count: {s.param_count}")
    console.print(f"Weight memory: {format_bytes(s.weight_memory_bytes)}")
    console.print(f"Grad memory: {format_bytes(s.grad_memory_bytes)}")
    console.print(f"Optimizer memory: {format_bytes(s.optimizer_memory_bytes)}")
    console.print(f"Activation memory: {format_bytes(s.activation_memory_bytes)}")
    console.print(f"Temporary memory: {format_bytes(s.temporary_memory_bytes)}")
    console.print(f"Persistent memory: {format_bytes(s.persistent_memory_bytes)}")
    console.print(f"Peak memory: {format_bytes(s.peak_memory_bytes)}")
    console.print(f"Peak stage: {s.peak_stage}")

    table = Table(title="Operators")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Phase")
    table.add_column("Memory")
    table.add_column("Formula")

    for op in report.operators:
        table.add_row(
            op.name, 
            op.category, 
            op.phase, 
            format_bytes(op.memory_bytes), 
            op.formula, 
        )

    console.print(table)