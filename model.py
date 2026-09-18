import math
import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + self.eps) * self.weight


class EulerRoPE(nn.Module):
    def __init__(self, dims: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dims = dims
        freqs = 1.0 / (theta ** (mx.arange(0, dims, 2).astype(mx.float32) / dims))
        t = mx.arange(max_seq_len).astype(mx.float32)
        angles = mx.outer(t, freqs)
        self.cos = mx.cos(angles)
        self.sin = mx.sin(angles)

    def __call__(self, x, offset: int = 0):
        seq_len = x.shape[1]
        cos = self.cos[offset : offset + seq_len][:, None, :]
        sin = self.sin[offset : offset + seq_len][:, None, :]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated_real = x1 * cos - x2 * sin
        rotated_imag = x1 * sin + x2 * cos
        out = mx.stack([rotated_real, rotated_imag], axis=-1)
        return mx.reshape(out, x.shape)


class BidirectionalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = EulerRoPE(self.head_dim)

    def __call__(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        probs = mx.softmax(scores, axis=-1)
        output = probs @ v
        output = output.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.out_proj(output)


class BiMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, max_seq_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.A_log = mx.log(mx.arange(1, d_state + 1, dtype=mx.float32))
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    def __call__(self, x):
        B, L, D = x.shape
        proj = self.in_proj(x)
        u, gate = proj[..., :D], proj[..., D:]
        gate = nn.silu(gate)
        u_conv = nn.silu(self.conv(u))

        u_s = u_conv[..., :self.d_state]

        decay = mx.exp(-mx.exp(self.A_log))

        idx = mx.arange(L)
        diff = idx[:, None] - idx[None, :]
        mask = diff >= 0
        diff_clamped = mx.maximum(diff, 0)

        decay_exp = decay[:, None, None] ** diff_clamped[None, :, :]
        M_fwd = mx.where(mask[None, :, :], decay_exp, 0.0)
        M_bwd = M_fwd.transpose(0, 2, 1)

        u_vec = u_s.transpose(0, 2, 1)[..., None]

        out_fwd = (M_fwd[None, ...] @ u_vec).squeeze(-1).transpose(0, 2, 1)
        out_bwd = (M_bwd[None, ...] @ u_vec).squeeze(-1).transpose(0, 2, 1)

        fused = (out_fwd + out_bwd) * gate[..., :self.d_state]
        expanded = mx.concatenate([fused, u_conv[..., self.d_state:]], axis=-1)
        return self.out_proj(expanded)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class HybridPhobetorBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mamba = BiMambaBlock(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = BidirectionalAttention(d_model, n_heads)
        self.norm3 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def __call__(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        x = x + self.mlp(self.norm3(x))
        return x


class PhobetorModel(nn.Module):
    def __init__(self, vocab_size: int = 152064, d_model: int = 1024, n_layers: int = 8, n_heads: int = 16, d_ff: int = 2816):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = [HybridPhobetorBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.norm_f = RMSNorm(d_model)

    def __call__(self, x):
        h = self.embed(x)

        for layer in self.layers:
            h = layer(h)
        for layer in self.layers:
            h = layer(h)

        h = self.norm_f(h)
        logits = h @ self.embed.weight.T
        return logits