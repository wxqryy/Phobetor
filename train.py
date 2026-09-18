import gc
import glob
import json
import math
import os
import re
import shutil
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import psutil
from aim import Run, Text
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from transformers import AutoTokenizer

from dataset import TextDataset
from model import PhobetorModel

SEQ_LEN = 1024
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM_STEPS

TOTAL_STEPS = 250_000
WARMUP_STEPS = 2_000
PEAK_LR = 2.5e-4
MIN_LR = 2.5e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
MASK_EPS = 1e-3

LOG_INTERVAL = 25
SAMPLE_INTERVAL = 500
VAL_INTERVAL = 2_000
VAL_BATCHES = 64
SAVE_INTERVAL = 10_000
CLEANUP_INTERVAL = 500
MAX_CHECKPOINTS = 10
MAX_SWAP_ALLOWED_GB = 1.0
SEED = 1337

TOKENIZER_PATH = "./tokenizer"
TRAIN_DATA_PATH = "./data/train_tokens.bin"
VAL_DATA_PATH = "./data/val_tokens.bin"
CHECKPOINT_DIR = "./checkpoints"

MODEL_CONFIG = {
    "d_model": 1024,
    "n_heads": 16,
    "d_ff": 3072,
    "layer_pattern": ("mamba", "mamba", "mamba", "attention") * 3,
    "mamba_expand": 2.0,
    "d_state": 64,
    "mamba_head_dim": 64,
    "d_conv": 4,
}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
run = Run(repo="./.aim", experiment="phobetor_v2_300m")
run["hparams"] = {
    "seq_len": SEQ_LEN,
    "physical_batch": BATCH_SIZE,
    "grad_accum_steps": GRAD_ACCUM_STEPS,
    "effective_batch": EFFECTIVE_BATCH,
    "total_steps": TOTAL_STEPS,
    "warmup_steps": WARMUP_STEPS,
    "peak_lr": PEAK_LR,
    "min_lr": MIN_LR,
    "weight_decay": WEIGHT_DECAY,
    "grad_clip": GRAD_CLIP,
    "architecture": "9xBiMamba2+3xBiAttention",
    "loss": "masked_diffusion_1_over_p",
}

print("Loading tokenizer and datasets...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

train_dataset = TextDataset(
    TRAIN_DATA_PATH,
    seq_len=SEQ_LEN,
    shuffle_seed=SEED
)

val_dataset = TextDataset(
    VAL_DATA_PATH,
    seq_len=SEQ_LEN,
    shuffle_seed=SEED + 1
)

MASK_TOKEN_ID = tokenizer.mask_token_id
VOCAB_SIZE = len(tokenizer)
FORBIDDEN_SAMPLE_IDS = [
    token_id
    for token_id in {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.mask_token_id,
    }
    if token_id is not None
]

if MASK_TOKEN_ID is None:
    raise RuntimeError("Tokenizer does not define mask_token_id")
if SEQ_LEN % 32 != 0:
    raise RuntimeError("SEQ_LEN must be divisible by 32 for mlx-recurrence ssd_scan")

print("Initializing Phobetor v2...")
mx.random.seed(SEED)
model = PhobetorModel(vocab_size=VOCAB_SIZE, **MODEL_CONFIG)
mx.eval(model.parameters())

warmup = opt.linear_schedule(0.0, PEAK_LR, WARMUP_STEPS)
decay = opt.cosine_decay(PEAK_LR, TOTAL_STEPS - WARMUP_STEPS, end=MIN_LR)
lr_schedule = opt.join_schedules([warmup, decay], [WARMUP_STEPS])


def no_weight_decay(path, weight):
    name = path.rsplit(".", 1)[-1]
    return weight.ndim < 2 or name in {"A_log", "D", "dt_bias"}


optimizer = opt.MultiOptimizer(
    [
        opt.AdamW(
            learning_rate=lr_schedule,
            betas=[0.9, 0.95],
            eps=1e-8,
            weight_decay=0.0,
            bias_correction=True,
        ),
        opt.AdamW(
            learning_rate=lr_schedule,
            betas=[0.9, 0.95],
            eps=1e-8,
            weight_decay=WEIGHT_DECAY,
            bias_correction=True,
        ),
    ],
    filters=[no_weight_decay],
)


def checkpoint_step(path):
    match = re.search(r"ckpt_step_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1


def latest_checkpoint():
    paths = glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_step_*"))
    return max(paths, key=checkpoint_step) if paths else None


def checkpoint_path(step):
    return os.path.join(CHECKPOINT_DIR, f"ckpt_step_{step:07d}")


def save_optimizer(path):
    state = tree_flatten(optimizer.state, destination={})
    mx.save_safetensors(path, state)


def prune_checkpoints():
    paths = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_step_*")),
        key=checkpoint_step,
    )
    for path in paths[:-MAX_CHECKPOINTS]:
        shutil.rmtree(path, ignore_errors=True)


def save_checkpoint(step, train_loss, val_loss, tokens_seen, sample_text=""):
    path = checkpoint_path(step)
    temp_path = os.path.join(CHECKPOINT_DIR, f".tmp_ckpt_{step:07d}")
    shutil.rmtree(temp_path, ignore_errors=True)
    os.makedirs(temp_path)

    model.save_weights(os.path.join(temp_path, "model.safetensors"))
    save_optimizer(os.path.join(temp_path, "optimizer.safetensors"))
    with open(os.path.join(temp_path, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "tokens_seen": tokens_seen,
                "sample": sample_text,
                "vocab_size": VOCAB_SIZE,
                "model_config": {**MODEL_CONFIG, "layer_pattern": list(MODEL_CONFIG["layer_pattern"])},
            },
            f,
            indent=2,
        )

    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(temp_path, path)
    prune_checkpoints()
    print(f"Checkpoint saved: {path}")


def save_best(step, val_loss, tokens_seen):
    path = os.path.join(CHECKPOINT_DIR, "best_model")
    temp_path = os.path.join(CHECKPOINT_DIR, ".tmp_best_model")
    shutil.rmtree(temp_path, ignore_errors=True)
    os.makedirs(temp_path)

    model.save_weights(os.path.join(temp_path, "model.safetensors"))
    with open(os.path.join(temp_path, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"step": step, "val_loss": val_loss, "tokens_seen": tokens_seen},
            f,
            indent=2,
        )

    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(temp_path, path)
    print(f"New best validation loss: {val_loss:.4f} at step {step:,}")


start_step = 1
tokens_seen = 0
best_val_loss = float("inf")
latest = latest_checkpoint()

best_metrics_path = os.path.join(CHECKPOINT_DIR, "best_model", "metrics.json")
if os.path.exists(best_metrics_path):
    with open(best_metrics_path, "r", encoding="utf-8") as f:
        best_val_loss = float(json.load(f).get("val_loss", float("inf")))

if latest:
    weights_path = os.path.join(latest, "model.safetensors")
    optimizer_path = os.path.join(latest, "optimizer.safetensors")
    metrics_path = os.path.join(latest, "metrics.json")

    print(f"Resuming from {latest}")
    model.load_weights(weights_path)

    if os.path.exists(optimizer_path):
        optimizer.state = tree_unflatten(mx.load(optimizer_path))
        print("Optimizer state restored")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        start_step = int(metrics.get("step", 0)) + 1
        tokens_seen = int(metrics.get("tokens_seen", 0))

mx.eval(model.parameters(), optimizer.state)

total_params = sum(value.size for _, value in tree_flatten(model.parameters()))
trainable_params = sum(value.size for _, value in tree_flatten(model.trainable_parameters()))
print(f"Parameters: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable")
print(f"Vocab: {VOCAB_SIZE:,} | Effective batch: {EFFECTIVE_BATCH} | Tokens/update: {EFFECTIVE_BATCH * SEQ_LEN:,}")


def loss_fn(model, x_clean, key):
    B, L = x_clean.shape
    key_p, key_mask = mx.random.split(key, num=2)
    p_mask = mx.random.uniform(low=MASK_EPS, high=1.0, shape=(B, 1), key=key_p)
    mask = mx.random.uniform(shape=(B, L), key=key_mask) < p_mask
    x_noisy = mx.where(mask, mx.array(MASK_TOKEN_ID, dtype=x_clean.dtype), x_clean)
    logits = model(x_noisy)
    token_ce = nn.losses.cross_entropy(logits, x_clean)
    weighted = token_ce * mask.astype(token_ce.dtype) / p_mask
    return mx.sum(weighted) / (B * L)


loss_and_grad = nn.value_and_grad(model, loss_fn)


def eager_grad(batch, key):
    return loss_and_grad(model, batch, key)


try:
    compiled_grad = mx.compile(eager_grad, inputs=model.state, outputs=model.state)
    test_batch = train_dataset.get_batch(BATCH_SIZE, batch_index=0)
    test_loss, test_grads = compiled_grad(test_batch, mx.random.key(0))
    mx.eval(test_loss, test_grads)
    del test_batch, test_loss, test_grads
    mx.clear_cache()
    grad_runner = compiled_grad
    print("MLX compile: enabled")
except Exception as e:
    grad_runner = eager_grad
    print(f"MLX compile unavailable, using eager mode: {e}")


def accumulate(accum, grads, scale):
    if accum is None:
        return tree_map(lambda g: g * scale, grads)
    return tree_map(lambda a, g: a + g * scale, accum, grads)


def evaluate(step):
    model.eval()
    total = 0.0
    try:
        for i in range(VAL_BATCHES):
            batch = val_dataset.get_batch(BATCH_SIZE, batch_index=i)
            key = mx.random.key(20_000_000 + i)
            loss = loss_fn(model, batch, key)
            mx.eval(loss)
            total += float(loss.item())
    finally:
        model.train()
    return total / VAL_BATCHES


def suppress_special_logits(logits):
    for token_id in FORBIDDEN_SAMPLE_IDS:
        logits = mx.where(
            mx.arange(logits.shape[-1]) == token_id,
            mx.array(float("-inf"), dtype=logits.dtype),
            logits,
        )
    return logits


def generate_sample(prompt_text, target_len=32, steps=16):
    model.eval()

    try:
        prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt)

        real_len = prompt_len + target_len
        padded_len = ((real_len + 31) // 32) * 32
        padding_len = padded_len - real_len

        tokens = (
            prompt
            + [MASK_TOKEN_ID] * target_len
            + [MASK_TOKEN_ID] * padding_len
        )

        x = mx.array([tokens], dtype=mx.int32)

        positions = mx.arange(padded_len)[None, :]

        generation_positions = (
            (positions >= prompt_len)
            & (positions < prompt_len + target_len)
        )

        for s in range(steps):
            logits = model(x)
            logits = suppress_special_logits(logits)

            preds = mx.argmax(logits, axis=-1)

            confidence = (
                mx.max(logits, axis=-1)
                - mx.logsumexp(logits, axis=-1)
            )

            candidates = (
                (x == MASK_TOKEN_ID)
                & generation_positions
            )

            remaining = int(mx.sum(candidates).item())

            if remaining == 0:
                break

            target_revealed = math.ceil(
                (s + 1) / steps * target_len
            )

            current_revealed = target_len - remaining

            num_to_open = min(
                max(target_revealed - current_revealed, 0),
                remaining
            )

            if num_to_open == 0:
                continue

            ranked = mx.where(
                candidates,
                confidence,
                mx.array(float("-inf"), dtype=confidence.dtype)
            )

            selected = mx.argsort(-ranked[0])[:num_to_open]

            reveal = mx.zeros_like(x, dtype=mx.bool_)

            reveal = mx.put_along_axis(
                reveal,
                selected[None, :],
                mx.ones((1, num_to_open), dtype=mx.bool_),
                axis=1
            )

            x = mx.where(reveal, preds, x)

        remaining = (
            (x == MASK_TOKEN_ID)
            & generation_positions
        )

        if int(mx.sum(remaining).item()) > 0:
            logits = suppress_special_logits(model(x))
            preds = mx.argmax(logits, axis=-1)
            x = mx.where(remaining, preds, x)

        mx.eval(x)

        result = x[0, :prompt_len + target_len].tolist()

        return tokenizer.decode(
            result,
            skip_special_tokens=True
        )

    finally:
        model.train()

MAX_SWAP_ALLOWED_GB = 1.0
MAX_SWAP_WRITES_GB = 4.0

initial_swap = psutil.swap_memory()
initial_swap_gb = initial_swap.used / 1024**3
initial_swap_out = initial_swap.sout

def check_swap():
    swap = psutil.swap_memory()

    growth_gb = swap.used / 1024**3 - initial_swap_gb
    writes_gb = (swap.sout - initial_swap_out) / 1024**3

    if growth_gb > MAX_SWAP_ALLOWED_GB:
        raise RuntimeError(f"Swap grew by {growth_gb:.2f} GB > {MAX_SWAP_ALLOWED_GB:.2f} GB")

    if writes_gb > MAX_SWAP_WRITES_GB:
        raise RuntimeError(f"System wrote {writes_gb:.2f} GB to swap since training started")

model.train()
last_completed_step = start_step - 1
last_train_loss = None
last_val_loss = None
last_sample = ""
process_start = time.time()

print(f"Training from step {start_step:,} to {TOTAL_STEPS:,}")

try:
    for step in range(start_step, TOTAL_STEPS + 1):
        step_start = time.time()
        accumulated_grads = None
        train_loss = 0.0

        for micro in range(GRAD_ACCUM_STEPS):
            micro_id = (step - 1) * GRAD_ACCUM_STEPS + micro
            batch = train_dataset.get_batch(BATCH_SIZE, batch_index=micro_id)
            key = mx.random.key(2_000_000 + micro_id)
            loss, grads = grad_runner(batch, key)
            accumulated_grads = accumulate(accumulated_grads, grads, 1.0 / GRAD_ACCUM_STEPS)
            mx.eval(loss, accumulated_grads)
            train_loss += float(loss.item()) / GRAD_ACCUM_STEPS

        accumulated_grads, grad_norm = opt.clip_grad_norm(accumulated_grads, GRAD_CLIP)
        optimizer.update(model, accumulated_grads)
        mx.eval(model.parameters(), optimizer.state, grad_norm)

        if not math.isfinite(train_loss):
            raise RuntimeError(f"Non-finite loss: {train_loss}")

        tokens_this_step = EFFECTIVE_BATCH * SEQ_LEN
        tokens_seen += tokens_this_step
        last_completed_step = step
        last_train_loss = train_loss
        elapsed = time.time() - step_start
        tok_s = tokens_this_step / max(elapsed, 1e-8)
        current_lr = float(lr_schedule(mx.array(step - 1)).item())

        if step % LOG_INTERVAL == 0:
            check_swap()
            grad_norm_value = float(grad_norm.item())
            ram = psutil.virtual_memory().percent
            mlx_gb = mx.get_active_memory() / 1024**3
            print(
                f"Step {step:06d} | Loss {train_loss:.4f} | LR {current_lr:.2e} | "
                f"Grad {grad_norm_value:.2f} | {tok_s:.0f} tok/s | MLX {mlx_gb:.1f} GB | RAM {ram:.1f}%"
            )
            run.track(train_loss, name="train_loss", step=step)
            run.track(current_lr, name="learning_rate", step=step)
            run.track(tok_s, name="tokens_per_sec", step=step)
            run.track(grad_norm_value, name="grad_norm", step=step)

        if step % SAMPLE_INTERVAL == 0:
            prompt = "The basic law of"

            last_sample = generate_sample(
                prompt,
                target_len=32,
                steps=16
            )

            print(f"\nSAMPLE {step}: {last_sample}\n")

            run.track(
                Text(last_sample),
                name="sample",
                step=step
            )

            with open("./latest_samples.log", "a", encoding="utf-8") as f:
                f.write(
                    f"Step {step:06d}\n"
                    f"Prompt:   {prompt}\n"
                    f"Phobetor: {last_sample}\n"
                    f"{'=' * 80}\n"
                )

            gc.collect()
            mx.clear_cache()

        if step % VAL_INTERVAL == 0:
            last_val_loss = evaluate(step)
            print(f"VAL {step}: {last_val_loss:.4f}")
            run.track(last_val_loss, name="val_loss", step=step)
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                save_best(step, last_val_loss, tokens_seen)
            gc.collect()
            mx.clear_cache()

        if step % SAVE_INTERVAL == 0:
            save_checkpoint(step, train_loss, last_val_loss, tokens_seen, last_sample)

        if step % CLEANUP_INTERVAL == 0:
            gc.collect()
            mx.clear_cache()

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    if last_completed_step >= start_step and last_train_loss is not None:
        save_checkpoint(last_completed_step, last_train_loss, last_val_loss, tokens_seen, last_sample)
    print("Emergency checkpoint saved")

except Exception as e:
    print(f"\nTraining crashed: {type(e).__name__}: {e}")
    if last_completed_step >= start_step and last_train_loss is not None:
        try:
            save_checkpoint(last_completed_step, last_train_loss, last_val_loss, tokens_seen, last_sample)
            print("Emergency checkpoint saved")
        except Exception as save_error:
            print(f"Emergency save failed: {save_error}")
    raise

finally:
    elapsed_hours = (time.time() - process_start) / 3600
    print(f"Process runtime: {elapsed_hours:.2f} h")