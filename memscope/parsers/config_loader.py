import json
from pathlib import Path

import yaml

from memscope.schemas.config import FullConfig, ModelConfig, TrainConfig

def load_config(path: str) -> FullConfig:
    """
    加载配置文件并返回强类型的 FullConfig 对象。
    
    Args:
        path (str): 配置文件的路径 (如 "configs/llama.json" 或 "configs/llama.yaml")
        
    Returns:
        FullConfig: 包含 model 和 train 子对象的完整配置实例
        
    Raises:
        ValueError: 当文件格式不支持或结构错误时抛出
    """

    # 路径标准化: 将字符串路径转换为 Path 对象，方便跨平台处理 (Windows/Linux/Mac)
    # 例如：自动处理斜杠 '/' 和反斜杠 '\' 的区别
    path_obj = Path(path)

    # 获取文件后缀并转为小写，确保 ".YAML" 和 ".yaml" 被视为相同
    suffix = path_obj.suffix.lower()

    with open(path_obj, "r", encoding="utf-8") as f:
        if suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        elif suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file: {path}")
    
    # 使用 **data["model"] 将字典解包为关键字参数，传递给 ModelConfig 的构造函数
    model = ModelConfig(**data["model"])
    train = TrainConfig(**data["train"])
    return FullConfig(model=model, train=train)