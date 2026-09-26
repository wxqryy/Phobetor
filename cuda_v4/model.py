import math
import os

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


def reference_scan(u, delta, b, c, a):
    batch, length, heads, width = u.shape
    state = torch.zeros(batch, heads, width, b.shape[-1], dtype=torch.float32, device=u.device)
    outputs = []
    for t in range(length):
        dt = delta[:, t, :, None, None].float()
        decay = torch.exp(dt * a[None, :, None, None].float())
        drive = dt * b[:, t, 0, None, None, :].float() * u[:, t, :, :, None].float()
        state = decay * state + drive
        outputs.append((state * c[:, t, 0, None, None, :].float()).sum(dim=-1).to(u.dtype))
    return torch.stack(outputs, dim=1)


class Mamba2Core(nn.Module):
    def __init__(self, d_model, expand=2.0, d_state=64, head_dim=64, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand)
        self.d_state = d_state
        self.head_dim = head_dim
        self.d_conv = d_conv
        if self.d_inner % head_dim:
            raise ValueError('d_model * expand must be divisible by head_dim')
        self.n_heads = self.d_inner // head_dim
        self.conv_dim = self.d_inner + 2 * d_state
        self.proj_dim = 2 * self.d_inner + 2 * d_state + self.n_heads
        self.in_proj = nn.Linear(d_model, self.proj_dim, bias=False)
        self.conv = nn.Conv1d(self.conv_dim, self.conv_dim, d_conv, padding=d_conv - 1, groups=self.conv_dim)
        self.A_log = nn.Parameter(torch.zeros(self.n_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads))
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        batch, original_length, _ = x.shape
        pad = (-original_length) % 32
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        length = x.shape[1]
        projected = self.in_proj(x)
        z, xbc, dt_raw = torch.split(projected, (self.d_inner, self.conv_dim, self.n_heads), dim=-1)
        xbc = F.silu(self.conv(xbc.transpose(1, 2))[:, :, :length].transpose(1, 2))
        u, b, c = torch.split(xbc, (self.d_inner, self.d_state, self.d_state), dim=-1)
        u = u.reshape(batch, length, self.n_heads, self.head_dim)
        b = b.unsqueeze(2).contiguous()
        c = c.unsqueeze(2).contiguous()
        delta = F.softplus(dt_raw.float() + self.dt_bias.float())
        a = -torch.exp(self.A_log.float())
        if x.device.type == 'cuda' and os.getenv('PHOBETOR_REFERENCE_SSD') != '1':
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
            y = mamba_chunk_scan_combined(u.contiguous(), delta.contiguous(), a.contiguous(), b, c,
                                          chunk_size=32, state_dtype=torch.float32)
        else:
            y = reference_scan(u, delta, b, c, a)
        y = y + u * self.D[None, None, :, None]
        y = self.norm(y.reshape(batch, length, self.d_inner) * F.silu(z))
        return self.out_proj(y)[:, :original_length]


class BidirectionalMamba2(nn.Module):
    def __init__(self, d_model, expand=2.0, d_state=64, head_dim=64, d_conv=4):
        super().__init__()
        self.forward_core = Mamba2Core(d_model, expand, d_state, head_dim, d_conv)
        self.backward_core = Mamba2Core(d_model, expand, d_state, head_dim, d_conv)
        self.fuse = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x):
        forward = self.forward_core(x)
        backward = self.backward_core(x.flip(1)).flip(1)
        return self.fuse(torch.cat((forward, backward), dim=-1))


class HybridPhobetorBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, mamba_expand, d_state, mamba_head_dim, d_conv):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.mamba = BidirectionalMamba2(d_model, mamba_expand, d_state, mamba_head_dim, d_conv)
        self.attn = BidirectionalAttention(d_model, n_heads)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        return x + self.mlp(self.norm3(x))


class PhobetorModel(nn.Module):
    def __init__(self, vocab_size, d_model=1024, n_heads=16, d_ff=3072, n_layers=8,
                 recurrent_passes=2, gradient_checkpointing=True, mamba_expand=2.0,
                 d_state=64, mamba_head_dim=64, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.recurrent_passes = recurrent_passes
        self.gradient_checkpointing = gradient_checkpointing
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            HybridPhobetorBlock(d_model, n_heads, d_ff, mamba_expand, d_state, mamba_head_dim, d_conv)
            for _ in range(n_layers)
        ])
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
