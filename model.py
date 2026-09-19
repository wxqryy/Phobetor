import math

import mlx.core as mx
import mlx.nn as nn
from mlx_recurrence import ssd_scan


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class BidirectionalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope_base: float = 10000.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)

    def __call__(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        y = y.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.out_proj(y)


class Mamba2Core(nn.Module):
    def __init__(
        self,
        d_model: int,
        expand: float = 2.0,
        d_state: int = 64,
        head_dim: int = 64,
        d_conv: int = 4,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        a_min: float = 1.0,
        a_max: float = 16.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand)
        self.d_state = d_state
        self.head_dim = head_dim
        self.d_conv = d_conv

        if self.d_inner % head_dim != 0:
            raise ValueError("d_model * expand must be divisible by head_dim")
        if head_dim % 32 != 0:
            raise ValueError("mlx-recurrence requires head_dim to be divisible by 32")

        self.n_heads = self.d_inner // head_dim
        self.bc_dim = d_state
        self.conv_dim = self.d_inner + 2 * self.bc_dim
        self.proj_dim = 2 * self.d_inner + 2 * self.bc_dim + self.n_heads

        self.in_proj = nn.Linear(d_model, self.proj_dim, bias=False)
        self.conv = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.conv_dim,
            bias=True,
        )

        a = mx.random.uniform(low=a_min, high=a_max, shape=(self.n_heads,)).astype(mx.float32)
        self.A_log = mx.log(a)
        log_dt = mx.random.uniform(low=math.log(dt_min), high=math.log(dt_max), shape=(self.n_heads,))
        dt = mx.exp(log_dt)
        self.dt_bias = dt + mx.log(-mx.expm1(-dt))
        self.D = mx.ones((self.n_heads,), dtype=mx.float32)

        self.norm = nn.RMSNorm(self.d_inner, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def __call__(self, x):
        B, L, _ = x.shape
        original_L = L

        pad_len = (-L) % 32
        if pad_len:
            x = mx.concatenate(
                [x, mx.zeros((B, pad_len, self.d_model), dtype=x.dtype)],
                axis=1,
            )
            L = x.shape[1]

        projected = self.in_proj(x)
        z = projected[..., :self.d_inner]
        xbc_start = self.d_inner
        xbc_end = xbc_start + self.conv_dim
        xbc = projected[..., xbc_start:xbc_end]
        dt_raw = projected[..., xbc_end:]

        xbc = nn.silu(self.conv(xbc)[:, :L, :])
        u = xbc[..., :self.d_inner]
        B_shared = xbc[..., self.d_inner:self.d_inner + self.d_state]
        C_shared = xbc[..., self.d_inner + self.d_state:]

        u = u.reshape(B, L, self.n_heads, self.head_dim)
        B_in = mx.contiguous(mx.broadcast_to(B_shared[:, :, None, :], (B, L, self.n_heads, self.d_state)))
        C_in = mx.contiguous(mx.broadcast_to(C_shared[:, :, None, :], (B, L, self.n_heads, self.d_state)))

        delta = nn.softplus(dt_raw + self.dt_bias)
        A_neg = mx.contiguous(mx.broadcast_to(-mx.exp(self.A_log)[:, None], (self.n_heads, self.d_state)))

        y = ssd_scan(u, delta, B_in, C_in, A_neg)
        y = y + u * self.D[None, None, :, None]
        y = y.reshape(B, L, self.d_inner)
        y = self.norm(y * nn.silu(z))
        y = self.out_proj(y)
        return y[:, :original_L, :]


class BidirectionalMamba2(nn.Module):
    def __init__(self, d_model: int, expand: float = 2.0, d_state: int = 64, head_dim: int = 64, d_conv: int = 4):
        super().__init__()
        self.forward_core = Mamba2Core(d_model, expand, d_state, head_dim, d_conv)
        self.backward_core = Mamba2Core(d_model, expand, d_state, head_dim, d_conv)
        self.fuse = nn.Linear(d_model * 2, d_model, bias=False)

    def __call__(self, x):
        forward = self.forward_core(x)
        reversed_x = mx.flip(x, axis=1)
        backward = mx.flip(self.backward_core(reversed_x), axis=1)
        return self.fuse(mx.concatenate([forward, backward], axis=-1))


class PhobetorLayer(nn.Module):
    def __init__(self, mixer: str, d_model: int, n_heads: int, d_ff: int, mamba_expand: float, d_state: int, mamba_head_dim: int, d_conv: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)

        if mixer == "mamba":
            self.mixer = BidirectionalMamba2(d_model, mamba_expand, d_state, mamba_head_dim, d_conv)
        elif mixer == "attention":
            self.mixer = BidirectionalAttention(d_model, n_heads)
        else:
            raise ValueError(f"Unknown mixer type: {mixer}")

        self.mlp = SwiGLU(d_model, d_ff)

    def __call__(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PhobetorModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 1024,
        n_heads: int = 16,
        d_ff: int = 3072,
        layer_pattern=("mamba", "mamba", "mamba", "attention") * 3,
        mamba_expand: float = 2.0,
        d_state: int = 64,
        mamba_head_dim: int = 64,
        d_conv: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.layer_pattern = tuple(layer_pattern)

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = [
            PhobetorLayer(
                mixer=mixer,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                mamba_expand=mamba_expand,
                d_state=d_state,
                mamba_head_dim=mamba_head_dim,
                d_conv=d_conv,
            )
            for mixer in self.layer_pattern
        ]
        self.norm_f = nn.RMSNorm(d_model, eps=1e-5)

    def __call__(self, tokens, output_start=None, output_end=None):
        h = self.embed(tokens)
        for layer in self.layers:
            h = layer(h)
        h = self.norm_f(h)

        if output_start is not None or output_end is not None:
            h = h[:, output_start:output_end, :]
        return h @ self.embed.weight.T