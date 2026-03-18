import math

def ceil_to_multiple(x: int, multiple: int) -> int:
    """
    将数字 x 向上取整到最接近的 multiple 的倍数。
    
    逻辑：
    1. x / multiple: 计算包含多少个倍数单元。
    2. math.ceil(...): 向上取整 (例如 3.1 -> 4)。
    3. * multiple: 还原为实际数值。
    
    示例：ceil_to_multiple(10, 8) -> 16 (因为 10 不是 8 的倍数，下一个是 16)
    """
    return int(math.ceil(x / multiple) * multiple)

def prod(shape):
    """
    计算列表所有元素的乘积。
    用于计算 Tensor 的总元素个数 (Total Elements)。
    
    示例：shape=[2, 3, 4] -> 2*3*4 = 24 个元素
    """
    out = 1
    for x in shape:
        out *= x
    return out

def format_bytes(num_bytes: int) -> str:
    """
    将字节数转换为可读的字符串 (如 "1.50 GiB")。
    
    逻辑：
    1. 定义单位列表 ["B", "KiB", "MiB", "GiB", "TiB"]。注意这里是 KiB (1024)，不是 KB (1000)。
    2. 循环除以 1024，直到数值小于 1024 或到达最大单位。
    3. 返回格式化字符串，保留两位小数。
    """
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        # 如果值已经很小 (<1024) 或者已经是最大单位 (TiB)，停止转换
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"