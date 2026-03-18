DTYPE_BYTES = {
    "fp32": 4, 
    "float": 4, 
    "bf16": 2,
    "bfloat16": 2, 
    "fp16": 2, 
    "float16": 2, 
    "int8": 1,
    "qint8": 1,
}

def bytes_per_dtype(dtype: str) -> int:
    key = dtype.lower()
    if key not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return DTYPE_BYTES[key]