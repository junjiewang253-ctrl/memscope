from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # 提升精度到 Float32 进行统计计算，防止溢出
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(rms + self.eps)
        return (x_norm.to(x.dtype)) * self.weight
    
class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        # x: [B, S, H]
        B, S, H = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, S, dh]
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)   # [B, nh, S, S]
        attn_probs = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)

        context = context.transpose(1, 2).contiguous().view(B, S, H)
        out = self.o_proj(context)
        return out
    
class MLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()

        # SwiGLU 结构的三个投影层
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False) # 门控
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)   # 值
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False) # 输出
        self.act = nn.SiLU() # Swish 激活函数

    def forward(self, x):
        # 1. 门控路径：Gate(x) = SiLU(W_g * x)
        gate = self.act(self.gate_proj(x))

        # 2. 值路径：Up(x) = W_u * x
        up = self.up_proj(x)

        # 3. 元素级相乘 (Gating Mechanism)
        x = gate * up

        # 4. 输出投影
        x = self.down_proj(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = SelfAttention(hidden_size, num_heads)
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, ffn_hidden_size)

    def forward(self, x):
        # Pre-Norm 结构：先 Norm，再计算，最后残差相加
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x
    
class ToyTransformerLM(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            hidden_size: int, 
            num_layers: int, 
            num_heads: int, 
            ffn_hidden_size: int, 
            max_seq_len: int, # 虽然定义了，但在当前 forward 中未使用（因为缺位置编码）
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_hidden_size)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        # input_ids: [B, S]
        x = self.embed_tokens(input_ids) # [B, S, H]
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x) # [B, S, V]
        return logits