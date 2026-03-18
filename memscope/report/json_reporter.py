# JSON: 机器可读
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

def write_json_report(report, path: str | Path) -> None:
    """
    将分析报告序列化为 JSON 文件保存。
    支持直接传入 Dataclass 对象，自动处理转换。
    
    Args:
        report: 包含分析结果的对象 (通常有一个 to_dict() 方法)
        path: 输出文件的路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # dataclass -> dict（递归展开）
    payload = asdict(report) if is_dataclass(report) else report

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)