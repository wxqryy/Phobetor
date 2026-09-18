import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from transformers import AutoTokenizer

from model import PhobetorModel

TOKENIZER_PATH = "./tokenizer"
SEQ_LEN = 64
SEED = 1337

mx.random.seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

if tokenizer.mask_token_id is None or tokenizer.eos_token_id is None:
    raise RuntimeError("Tokenizer must have MASK and EOS tokens")
if len(tokenizer) > 65536:
    raise RuntimeError("Tokenizer no longer fits uint16 token storage")

model = PhobetorModel(
    vocab_size=len(tokenizer),
    d_model=128,
    n_heads=4,
    d_ff=384,
    layer_pattern=("mamba", "attention"),
    mamba_expand=2.0,
    d_state=16,
    mamba_head_dim=64,
    d_conv=4,
)

mask_id = tokenizer.mask_token_id
x = mx.random.randint(0, len(tokenizer), shape=(1, SEQ_LEN), dtype=mx.int32)
x = mx.where(mx.arange(SEQ_LEN)[None, :] % 3 == 0, mx.array(mask_id, dtype=mx.int32), x)
target = mx.random.randint(0, len(tokenizer), shape=(1, SEQ_LEN), dtype=mx.int32)


def loss_fn(model, tokens, targets):
    logits = model(tokens)
    return mx.mean(nn.losses.cross_entropy(logits, targets))

loss_and_grad = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad(model, x, target)
mx.eval(loss, grads)

loss_value = float(loss.item())
if not math.isfinite(loss_value):
    raise RuntimeError(f"Non-finite smoke-test loss: {loss_value}")

param_count = sum(v.size for _, v in tree_flatten(model.parameters()))
grad_count = sum(v.size for _, v in tree_flatten(grads))

print(f"Tokenizer: {len(tokenizer):,} tokens")
print(f"Tiny model: {param_count / 1e6:.2f}M params")
print(f"Gradient values: {grad_count:,}")
print(f"Loss: {loss_value:.4f}")
print("Mamba SSD forward + backward + attention: OK")
