import os
import time
import psutil
from collections import deque
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten
from transformers import AutoTokenizer
from aim import Run
from model import PhobetorModel
from dataset import TextDataset
from saver import PhobetorCheckpointManager
import glob
import json

SEQ_LEN = 1024
BATCH_SIZE = 2
LEARNING_RATE = 2.5e-4
WARMUP_STEPS = 1000
TOTAL_STEPS = 500_000
SAVE_INTERVAL = 15_000
SAMPLE_INTERVAL = 250
LOG_INTERVAL = 25
MAX_SWAP_ALLOWED_GB = 1.0

run = Run(repo="./.aim", experiment="phobetor_300m")
run["hparams"] = {
    "seq_len": SEQ_LEN,
    "batch_size": BATCH_SIZE,
    "lr": LEARNING_RATE,
    "layers": 8,
    "recurrent_passes": 2,
    "effective_layers": 16
}

print("Loading Tokenizer & Dataset...")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
dataset = TextDataset("./data/train_tokens.bin", seq_len=SEQ_LEN)
saver = PhobetorCheckpointManager(checkpoint_dir="./checkpoints", max_checkpoints=15)

MASK_TOKEN_ID = tokenizer.mask_token_id
VOCAB_SIZE = len(tokenizer)

print("Initializing Phobetor (8 layers, 16 effective)...")
model = PhobetorModel(vocab_size=VOCAB_SIZE, d_model=1024, n_layers=8, n_heads=16, d_ff=2816)

start_step = 1
ckpts = sorted(glob.glob("./checkpoints/ckpt_step_*"), key=os.path.getmtime)
if ckpts:
    latest_ckpt = ckpts[-1]
    weights_path = os.path.join(latest_ckpt, "model.safetensors")
    metrics_path = os.path.join(latest_ckpt, "metrics.json")

    if os.path.exists(weights_path):
        print(f"\nCheckpoint: {latest_ckpt}")
        model.load_weights(weights_path)
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
                start_step = data.get("step", 0) + 1
        print(f"Continuing learning from step: {start_step:,}...\n")

mx.eval(model.parameters())

total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
print(f"Model online! Total parameters: {total_params / 1e6:.2f}M")

optimizer = opt.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
sample_history = deque(maxlen=15)

def cosine_noise_schedule(t):
    return mx.cos((t + 0.008) / 1.008 * (1.5707963))

def loss_fn(model, x_clean):
    B, L = x_clean.shape
    t = mx.random.uniform(low=0.0, high=1.0, shape=(B, 1))
    mask_probs = 1.0 - cosine_noise_schedule(t)

    rand = mx.random.uniform(shape=(B, L))
    mask_indices = rand < mask_probs

    x_noisy = mx.where(mask_indices, mx.array(MASK_TOKEN_ID, dtype=x_clean.dtype), x_clean)

    logits = model(x_noisy)
    loss = nn.losses.cross_entropy(logits, x_clean)
    masked_loss = mx.sum(loss * mask_indices) / (mx.sum(mask_indices) + 1e-6)
    return masked_loss

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

def generate_sample(model, prompt_text, target_len=32, steps=12):
    prompt_tokens = tokenizer.encode(prompt_text)
    canvas = prompt_tokens + [MASK_TOKEN_ID] * target_len
    x = mx.array([canvas])

    for s in range(steps):
        logits = model(x)
        probs = mx.softmax(logits, axis=-1)
        preds = mx.argmax(probs, axis=-1)

        mask_pos = (x == MASK_TOKEN_ID)
        if not mx.any(mask_pos):
            break

        step_ratio = (s + 1) / steps
        x = mx.where(mask_pos & (mx.random.uniform(shape=x.shape) < step_ratio), preds, x)

    out_tokens = x[0].tolist()
    return tokenizer.decode(out_tokens)

INITIAL_SWAP_GB = psutil.swap_memory().used / (1024 ** 3)
print(f"Baseline swap: {INITIAL_SWAP_GB:.2f} GB. Guardian active.")

def check_swap_guardian():
    current_swap = psutil.swap_memory().used / (1024 ** 3)
    swap_growth = current_swap - INITIAL_SWAP_GB
    if swap_growth > MAX_SWAP_ALLOWED_GB:
        raise RuntimeError(
            f"[CRITICAL] Training caused new Swap allocation ({swap_growth:.2f} GB > {MAX_SWAP_ALLOWED_GB} GB). Stopping."
        )

print(f"Training starts! Target: {TOTAL_STEPS:,} steps.\n")
start_time = time.time()
tokens_processed = 0

try:
    for step in range(start_step, TOTAL_STEPS + 1):
        step_start = time.time()

        batch = dataset.get_batch(batch_size=BATCH_SIZE)
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)

        mx.eval(model.parameters(), optimizer.state, loss)

        tokens_processed += BATCH_SIZE * SEQ_LEN
        step_time = time.time() - step_start
        tok_s = (BATCH_SIZE * SEQ_LEN) / max(step_time, 1e-5)

        current_lr = LEARNING_RATE * min(step / WARMUP_STEPS, 1.0)

        if step % LOG_INTERVAL == 0:
            check_swap_guardian()
            saver.log_step(step, loss.item(), current_lr, tok_s)

            run.track(loss.item(), name="loss", step=step)
            run.track(tok_s, name="tokens_per_sec", step=step)
            run.track(current_lr, name="learning_rate", step=step)

            mx.clear_cache()

            print(
                f"Step {step:06d} | Loss: {loss.item():.4f} | Speed: {tok_s:.0f} tok/s | RAM: {psutil.virtual_memory().percent}%"
            )

        if step % SAMPLE_INTERVAL == 0:
            test_prompt = "The basic law of"
            prediction = generate_sample(model, test_prompt, target_len=24, steps=12)

            record = f"Step {step:06d} | Prompt: '{test_prompt}' --> Generated: '{prediction.strip()}'"
            sample_history.append(record)

            print("\n" + "=" * 70)
            print(f"SAMPLE {step}:")
            print(f"Task:      '{test_prompt}' [MASK x24]")
            print(f"Phobetor:  '{prediction.strip()}'")
            print("=" * 70 + "\n")

            with open("./latest_samples.log", "w", encoding="utf-8") as f:
                f.write("\n".join(sample_history) + "\n")

        if step % SAVE_INTERVAL == 0 or step == 100:
            elapsed = time.time() - start_time
            last_sample = sample_history[-1] if sample_history else "No samples yet."
            saver.save(
                model=model,
                step=step,
                loss=loss.item(),
                lr=current_lr,
                total_tokens=tokens_processed,
                elapsed_sec=elapsed,
                sample_text=last_sample
            )

except (KeyboardInterrupt, RuntimeError) as e:
    print(f"\nTraining halted: {e}")
    elapsed = time.time() - start_time
    saver.save(
        model=model,
        step=step,
        loss=loss.item(),
        lr=LEARNING_RATE,
        total_tokens=tokens_processed,
        elapsed_sec=elapsed,
        sample_text="Emergency save on stop"
    )
    print("Safe exit completed.")