from memscope.schemas.config import FullConfig
from memscope.utils import ceil_to_multiple

# 从配置对象中提取基础参数，并计算推导变量（Derived Variables）
def derived_vars(cfg: FullConfig) -> dict:
    m = cfg.model
    t = cfg.train

    H = m.hidden_size
    A = m.num_attention_heads
    G = m.num_kv_heads
    T = t.tensor_parallel
    B = t.micro_batch_size
    S = m.seq_len

    # 处理 Q/K/V 序列长度不一致的情况
    Sq = m.query_seq_len or S
    Sk = m.key_seq_len or S

    H_ffn = m.ffn_hidden_size
    L = m.num_layers

    # 合法性检查：隐藏层维度必须能被注意力头数整除，否则无法平均分配到头
    if H % A != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    
    Dh = H // A
    vpad = m.padded_vocab_size
    if vpad is None:
        # 【核心逻辑】参考 PDF 文档中的“显存对齐”原则
        # 在张量并行 (TP) 下，词表被切分到 T 个 GPU 上。
        # 为了保证每个 GPU 上的词表切片大小一致且满足硬件对齐要求（通常是 128 的倍数），
        # 需要将原始词表大小向上取整到 128 * T 的倍数。
        # 例如：Llama-7B vocab=32000, TP=1 -> ceil(32000, 128) = 32000
        #       Llama-7B vocab=32000, TP=2 -> ceil(32000, 256) = 32000
        #       若 vocab=32001, TP=1 -> ceil(32001, 128) = 32128 (多了 127 个填充 token)
        vpad = ceil_to_multiple(m.vocab_size, 128 * T)

    # 返回推导后的字典，供公式直接使用
    return {
        "B": B, 
        "S": S, 
        "Sq": Sq, 
        "Sk": Sk, 
        "H": H, 
        "A": A,
        "G": G, 
        "T": T, 
        "Dh": Dh,            # 每个头的维度 (Head Dimension)
        "H_ffn": H_ffn, 
        "L": L, 
        "V": m.vocab_size,   # 原始词表大小
        "Vpad": vpad,        # 对齐后的词表大小 (用于计算 Embedding 和 LM Head)
        "lambda_bucket": t.reduce_bucket_size,  # ZeRO 优化中的桶大小
    }

# 计算 Llama 架构模型的总参数量（考虑张量并行切分后的逻辑参数量，通常指单卡持有的参数量或总参数量，需看公式细节）
def estimate_param_count_llama(cfg: FullConfig) -> int:
    v = derived_vars(cfg)
    H = v["H"]
    T = v["T"]
    Vpad = v["Vpad"]
    L = v["L"]
    A = v["A"]
    G = v["G"]
    H_ffn = v["H_ffn"]

    phi = (
        2 * Vpad * H // T 
        + L * (
            2 * H * H // T 
            + 2 * H * H * G // (A * T) 
            + 3 * H * H_ffn // T 
            + 2 * H
        )
        + H
    )
    return int(phi)

def count_non_norm_params(cfg: FullConfig, phi: int) -> int:
    m = cfg.model
    return phi - 2 * m.num_layers * m.hidden_size - m.hidden_size