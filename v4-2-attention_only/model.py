import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        variance = x.float().square().mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps).to(x.dtype)) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BidirectionalAttention(nn.Module):
    def __init__(self, d_model, n_heads, rope_base=10000.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_base = rope_base
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _rope(self, x):
        half = self.head_dim // 2
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        frequencies = self.rope_base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
        angles = positions[:, None] * frequencies[None, :]
        cosine = angles.cos().to(x.dtype)[None, None, :, :]
        sine = angles.sin().to(x.dtype)[None, None, :, :]
        left, right = x[..., :half], x[..., half:]
        return torch.cat((left * cosine - right * sine, right * cosine + left * sine), dim=-1)

    def forward(self, x):
        batch, length, _ = x.shape
        shape = (batch, length, self.n_heads, self.head_dim)
        q = self._rope(self.q_proj(x).reshape(shape).transpose(1, 2))
        k = self._rope(self.k_proj(x).reshape(shape).transpose(1, 2))
        v = self.v_proj(x).reshape(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out_proj(y.transpose(1, 2).reshape(batch, length, self.d_model))


class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = BidirectionalAttention(d_model, n_heads)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PhobetorModel(nn.Module):
    def __init__(self, vocab_size, d_model=1024, n_heads=16, d_ff=3072, n_layers=16,
                 recurrent_passes=1, gradient_checkpointing=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.recurrent_passes = recurrent_passes
        self.gradient_checkpointing = gradient_checkpointing
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([AttentionBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)

    def forward(self, tokens, output_start=None, output_end=None):
        h = self.embed(tokens)
        for _ in range(self.recurrent_passes):
            for layer in self.layers:
                h = checkpoint(layer, h, use_reentrant=False) if self.training and self.gradient_checkpointing else layer(h)
        h = self.norm_f(h)
        if output_start is not None or output_end is not None:
            h = h[:, output_start:output_end]
        return F.linear(h, self.embed.weight)
