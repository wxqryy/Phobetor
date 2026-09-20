import gc
import math
import os
import signal
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import psutil
from aim import Run, Text
from mlx.utils import tree_flatten, tree_map
from transformers import AutoTokenizer

from dataset import TextDataset
from model import PhobetorModel
from sampler import PhobetorSampler, SamplerConfig
from training_objectives import (
    BlockTrainingConfig, TRAIN_MICRO_MODES, choose_prefix_length, prepare_random,
    prepare_block, weighted_loss, legacy_validation_loss, evaluate_blocks, repetition_metrics,
)
from training_state import (
    load_best_loss, restore_latest, save_best as write_best,
    save_checkpoint as write_checkpoint, validation_matches,
)

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
SAMPLE_MAX_NEW_TOKENS = 96
VAL_INTERVAL = 2_000
VAL_BATCHES = 64
AUX_VAL_BATCHES = 16
SAVE_INTERVAL = 500
CLEANUP_INTERVAL = 500
MAX_CHECKPOINTS = 10
MAX_SWAP_ALLOWED_GB = 1.0
MAX_SWAP_WRITES_GB = 4.0
SEED = 1337
VAL_SHUFFLE_SEED = 1338
VAL_MASK_SEED = 20_000_000
VALIDATION_CONFIG = {
    "shuffle_seed": VAL_SHUFFLE_SEED,
    "mask_seed": VAL_MASK_SEED,
    "batches": VAL_BATCHES,
    "batch_size": BATCH_SIZE,
    "seq_len": SEQ_LEN,
    "mask_eps": MASK_EPS,
}

TOKENIZER_PATH = "./tokenizer"
TRAIN_DATA_PATH = "./data/train_tokens.bin"
VAL_DATA_PATH = "./data/val_tokens.bin"
CHECKPOINT_DIR = "./checkpoints"
SAMPLE_LOG_PATH = "./latest_samples.log"

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

SAMPLER_CONFIG = SamplerConfig(
    context_window=1024,
    block_size=64,
    overlap=16,
    refinement_steps=32,
    temperature=0.0,
    audit_every=8,
    audit_start_fraction=0.25,
    audit_end_fraction=0.75,
    audit_candidates=32,
    audit_remask_k=1,
    audit_current_prob_threshold=0.20,
    audit_replacement_conf=0.30,
    audit_position_budget=2,
    final_polish_rounds=2,
)

BLOCK_TRAINING_CONFIG = BlockTrainingConfig(
    block_size=SAMPLER_CONFIG.block_size, overlap=SAMPLER_CONFIG.overlap,
)
BLOCK_VAL_SEED = 30_000_000
SAMPLE_PROMPTS = ("The basic law of", "Water boils when", "A triangle has")


def main():
    BLOCK_TRAINING_CONFIG.validate(SEQ_LEN)
    corruption_config = BLOCK_TRAINING_CONFIG.to_dict()
    block_validation_config = {
        "version": 1, "seed": BLOCK_VAL_SEED, "batches": AUX_VAL_BATCHES,
        "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
        "block_size": BLOCK_TRAINING_CONFIG.block_size, "overlap": BLOCK_TRAINING_CONFIG.overlap,
        "data_shuffle_seed": VAL_SHUFFLE_SEED,
        "tasks": ["continuation_initial", "overlap_initial", "half_masked_refinement"],
    }
    initial_swap = psutil.swap_memory()
    initial_swap_gb = initial_swap.used / 1024**3
    initial_swap_out = getattr(initial_swap, "sout", 0)
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
        "loss": "random_diffusion_plus_cropped_block_denoising",
        "training_corruption": corruption_config,
        "block_validation": block_validation_config,
        "sample_prompts": list(SAMPLE_PROMPTS),
        "tokens_seen_definition": "tokens actually passed to the model, excluding internal SSD padding",
        "sampler": SAMPLER_CONFIG.to_dict(),
        "validation": VALIDATION_CONFIG,
        "save_interval": SAVE_INTERVAL,
    }

    print("Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    train_dataset = TextDataset(TRAIN_DATA_PATH, seq_len=SEQ_LEN, shuffle_seed=SEED)
    val_dataset = TextDataset(VAL_DATA_PATH, seq_len=SEQ_LEN, shuffle_seed=VAL_SHUFFLE_SEED)
    MASK_TOKEN_ID = tokenizer.mask_token_id
    VOCAB_SIZE = len(tokenizer)

    if MASK_TOKEN_ID is None:
        raise RuntimeError("Tokenizer does not define mask_token_id")
    if SEQ_LEN % 32 != 0:
        raise RuntimeError("SEQ_LEN must be divisible by 32 for mlx-recurrence ssd_scan")

    print("Initializing Phobetor v2...")
    mx.random.seed(SEED)
    model = PhobetorModel(vocab_size=VOCAB_SIZE, **MODEL_CONFIG)
    mx.eval(model.parameters())
    sampler = PhobetorSampler(model, tokenizer, config=SAMPLER_CONFIG, seed=SEED)

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


    progress, resume_metrics = restore_latest(
        CHECKPOINT_DIR, model, optimizer, MODEL_CONFIG, VOCAB_SIZE,
    )
    start_step = progress.step + 1
    best_val_loss = load_best_loss(CHECKPOINT_DIR, VALIDATION_CONFIG)
    new_training_phase = resume_metrics.get("training_corruption_config") != corruption_config
    phase_start_step = progress.step if new_training_phase else resume_metrics.get("training_phase_start_step", progress.step)
    run["resumed_from_step"] = progress.step
    run["training_phase_start_step"] = phase_start_step


    def checkpoint_metadata(val_loss, sample_text):
        return {
            "val_loss": val_loss,
            "sample": sample_text,
            "vocab_size": VOCAB_SIZE,
            "model_config": {**MODEL_CONFIG, "layer_pattern": list(MODEL_CONFIG["layer_pattern"])},
            "sampler_config": SAMPLER_CONFIG.to_dict(),
            "validation_config": VALIDATION_CONFIG,
            "training_corruption_config": corruption_config,
            "training_phase_start_step": phase_start_step,
            "block_validation_config": block_validation_config,
            "block_validation_losses": last_block_losses,
            "block_validation_step": last_block_val_step,
            "val_step": last_val_step,
            "tokens_seen_definition": "model_input_tokens",
        }


    def save_checkpoint(val_loss, sample_text):
        return write_checkpoint(
            CHECKPOINT_DIR, model, optimizer, progress,
            checkpoint_metadata(val_loss, sample_text), MAX_CHECKPOINTS,
        )


    def save_best(step, val_loss):
        write_best(CHECKPOINT_DIR, model, {
            **checkpoint_metadata(val_loss, ""),
            "step": step, "tokens_seen": progress.tokens_seen,
            "val_step": step,
        })
        print(f"New best validation loss: {val_loss:.4f} at step {step:,}")


    mx.eval(model.parameters(), optimizer.state)

    total_params = sum(value.size for _, value in tree_flatten(model.parameters()))
    trainable_params = sum(value.size for _, value in tree_flatten(model.trainable_parameters()))
    print(f"Parameters: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable")
    print(f"Vocab: {VOCAB_SIZE:,} | Effective batch: {EFFECTIVE_BATCH} | "
          f"Source tokens/update: {EFFECTIVE_BATCH * SEQ_LEN:,} | Model tokens vary with crops")
    print(
        f"Sampler: block={SAMPLER_CONFIG.block_size}, overlap={SAMPLER_CONFIG.overlap}, "
        f"stride={SAMPLER_CONFIG.stride}, refine={SAMPLER_CONFIG.refinement_steps}"
    )


    print("Train corruption: 50% random | 25% continuation | 25% overlap; "
          "cropped prefixes, initial + intermediate denoising states")
    if new_training_phase:
        print(f"New corruption phase at step {progress.step:,}; weights and AdamW state retained")
    loss_and_grad = nn.value_and_grad(model, weighted_loss)


    def eager_grad(noisy, targets, weights):
        return loss_and_grad(model, noisy, targets, weights)


    try:
        compiled_grad = mx.compile(eager_grad, inputs=model.state, outputs=model.state)
        test_batch = train_dataset.get_batch(BATCH_SIZE, batch_index=0)
        test_prepared = prepare_random(test_batch, mx.random.key(0), MASK_TOKEN_ID, MASK_EPS)
        test_loss, test_grads = compiled_grad(*test_prepared)
        mx.eval(test_loss, test_grads)
        del test_batch, test_prepared, test_loss, test_grads
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


    def evaluate():
        model.eval()
        total = 0.0
        try:
            for i in range(VAL_BATCHES):
                batch = val_dataset.get_batch(BATCH_SIZE, batch_index=i)
                key = mx.random.key(VAL_MASK_SEED + i)
                loss = legacy_validation_loss(model, batch, key, MASK_TOKEN_ID, MASK_EPS)
                mx.eval(loss)
                value = float(loss.item())
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite validation loss: {value}")
                total += value
        finally:
            model.train()
        return total / VAL_BATCHES


    def evaluate_generation_modes():
        model.eval()
        try:
            return evaluate_blocks(
                model, val_dataset, BLOCK_TRAINING_CONFIG, MASK_TOKEN_ID,
                BATCH_SIZE, AUX_VAL_BATCHES, BLOCK_VAL_SEED,
            )
        finally:
            model.train()


    def generate_training_sample(prompt_text):
        model.eval()
        try:
            ids = sampler.generate_ids(
                tokenizer.encode(prompt_text, add_special_tokens=False),
                max_new_tokens=SAMPLE_MAX_NEW_TOKENS,
                min_new_tokens=8,
            )
            return prompt_text + tokenizer.decode(ids, skip_special_tokens=True), repetition_metrics(ids)
        finally:
            model.train()


    def check_swap():
        swap = psutil.swap_memory()
        growth_gb = swap.used / 1024**3 - initial_swap_gb
        writes_gb = (getattr(swap, "sout", 0) - initial_swap_out) / 1024**3
        if growth_gb > MAX_SWAP_ALLOWED_GB:
            raise RuntimeError(f"New swap allocation {growth_gb:.2f} GB > {MAX_SWAP_ALLOWED_GB:.2f} GB")
        if writes_gb > MAX_SWAP_WRITES_GB:
            raise RuntimeError(f"Swap writes {writes_gb:.2f} GB > {MAX_SWAP_WRITES_GB:.2f} GB")


    stop_requested = False


    def request_stop(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print("\nStop requested; saving after the current safe operation (signal again to force stop)")


    def stop_if_requested():
        if stop_requested:
            raise KeyboardInterrupt


    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    model.train()
    last_val_loss = resume_metrics.get("val_loss") if validation_matches(resume_metrics, VALIDATION_CONFIG) else None
    last_val_step = resume_metrics.get("val_step") if last_val_loss is not None else None
    same_block_validation = resume_metrics.get("block_validation_config") == block_validation_config
    last_block_losses = resume_metrics.get("block_validation_losses", {}) if same_block_validation else {}
    last_block_val_step = resume_metrics.get("block_validation_step") if same_block_validation else None
    last_sample = resume_metrics.get("sample", "")
    last_saved_step = progress.step
    process_start = time.time()
    loss = grads = accumulated_grads = None
    mode_sums = {mode: 0.0 for mode in set(TRAIN_MICRO_MODES)}
    mode_counts = {mode: 0 for mode in mode_sums}

    print(f"Training from step {start_step:,} to {TOTAL_STEPS:,}")


    def emergency_save():
        if not progress.safe_to_save:
            print("Update was interrupted/invalid; preserving the previous disk checkpoint")
            return
        if progress.step <= last_saved_step:
            print("No unsaved completed steps; previous checkpoint preserved")
            return
        gc.collect()
        mx.clear_cache()
        try:
            save_checkpoint(last_val_loss, last_sample)
            print(f"Emergency checkpoint saved at completed step {progress.step}")
        except Exception as save_error:
            print(f"Emergency save failed; previous checkpoints preserved: {save_error}")


    try:
        if new_training_phase or not last_block_losses:
            check_swap()
            last_block_losses = evaluate_generation_modes()
            last_block_val_step = progress.step
            print(f"BLOCK BASELINE {progress.step}: " + " | ".join(
                f"{name}={value:.4f}" for name, value in last_block_losses.items()
            ))
            for name, value in last_block_losses.items():
                run.track(value, name=name, step=progress.step)
            gc.collect()
            mx.clear_cache()
            stop_if_requested()
        for step in range(start_step, TOTAL_STEPS + 1):
            stop_if_requested()
            check_swap()
            step_start = time.time()
            accumulated_grads = None
            train_loss = 0.0
            model_tokens_this_step = 0

            for micro in range(GRAD_ACCUM_STEPS):
                micro_id = (step - 1) * GRAD_ACCUM_STEPS + micro
                batch = train_dataset.get_batch(BATCH_SIZE, batch_index=micro_id)
                key = mx.random.key(2_000_000 + micro_id)
                mode = TRAIN_MICRO_MODES[micro_id % len(TRAIN_MICRO_MODES)]
                if mode == "random":
                    prepared = prepare_random(batch, key, MASK_TOKEN_ID, MASK_EPS)
                else:
                    prefix_length = choose_prefix_length(BLOCK_TRAINING_CONFIG, SEQ_LEN, micro_id, SEED)
                    prepared = prepare_block(
                        batch, key, MASK_TOKEN_ID, BLOCK_TRAINING_CONFIG, prefix_length, mode,
                    )
                model_tokens_this_step += prepared[0].size
                loss, grads = grad_runner(*prepared)
                accumulated_grads = accumulate(accumulated_grads, grads, 1.0 / GRAD_ACCUM_STEPS)
                mx.eval(loss, accumulated_grads)
                micro_loss = float(loss.item())
                if not math.isfinite(micro_loss):
                    raise FloatingPointError(f"Non-finite loss before optimizer update: {micro_loss}")
                train_loss += micro_loss / GRAD_ACCUM_STEPS
                mode_sums[mode] += micro_loss
                mode_counts[mode] += 1
                stop_if_requested()

            grad_norm_value = progress.update(
                model, optimizer, accumulated_grads, train_loss,
                model_tokens_this_step, GRAD_CLIP,
            )
            loss = grads = accumulated_grads = None
            stop_if_requested()
            elapsed = time.time() - step_start
            tok_s = model_tokens_this_step / max(elapsed, 1e-8)
            current_lr = float(lr_schedule(mx.array(step - 1)).item())

            if step % SAVE_INTERVAL == 0:
                save_checkpoint(last_val_loss, last_sample)
                last_saved_step = progress.step

            if step % LOG_INTERVAL == 0:
                check_swap()
                ram = psutil.virtual_memory().percent
                mlx_gb = mx.get_active_memory() / 1024**3
                mode_means = {mode: mode_sums[mode] / mode_counts[mode] for mode in mode_sums if mode_counts[mode]}
                mode_text = " ".join(f"{mode[0].upper()} {mode_means[mode]:.3f}" for mode in dict.fromkeys(TRAIN_MICRO_MODES) if mode in mode_means)
                print(
                    f"Step {step:06d} | Loss {train_loss:.4f} [{mode_text}] | LR {current_lr:.2e} | "
                    f"Grad {grad_norm_value:.2f} | {tok_s:.0f} tok/s | MLX {mlx_gb:.1f} GB | RAM {ram:.1f}%"
                )
                run.track(train_loss, name="train_loss", step=step)
                run.track(current_lr, name="learning_rate", step=step)
                run.track(tok_s, name="tokens_per_sec", step=step)
                run.track(grad_norm_value, name="grad_norm", step=step)
                for mode, value in mode_means.items():
                    run.track(value, name=f"train_{mode}_loss", step=step)
                for mode in mode_sums:
                    mode_sums[mode] = 0.0
                    mode_counts[mode] = 0

            if step % SAMPLE_INTERVAL == 0:
                sample_start = time.time()
                for prompt_index, prompt_text in enumerate(SAMPLE_PROMPTS):
                    text, sample_metrics = generate_training_sample(prompt_text)
                    if prompt_index == 0:
                        last_sample = text
                    print(f"SAMPLE {step} | {sample_metrics} | {text!r}")
                    context = {"prompt": prompt_text}
                    run.track(Text(text), name="sample", step=step, context=context)
                    for name, value in sample_metrics.items():
                        run.track(value, name=f"sample_{name}", step=step, context=context)
                    with open(SAMPLE_LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(f"Step {step:06d}\nPrompt: {prompt_text}\n"
                                f"Metrics: {sample_metrics}\nPhobetor: {text!r}\n{'=' * 80}\n")
                    stop_if_requested()
                sample_seconds = time.time() - sample_start
                run.track(sample_seconds, name="sample_seconds", step=step)
                gc.collect()
                mx.clear_cache()
                stop_if_requested()

            if step % VAL_INTERVAL == 0:
                last_val_loss = evaluate()
                last_val_step = step
                print(f"VAL {step}: {last_val_loss:.4f}")
                run.track(last_val_loss, name="val_loss", step=step)
                last_block_losses = evaluate_generation_modes()
                last_block_val_step = step
                print(f"BLOCK VAL {step}: " + " | ".join(
                    f"{name}={value:.4f}" for name, value in last_block_losses.items()
                ))
                for name, value in last_block_losses.items():
                    run.track(value, name=name, step=step)
                if last_val_loss < best_val_loss:
                    save_best(step, last_val_loss)
                    best_val_loss = last_val_loss
                gc.collect()
                mx.clear_cache()
                stop_if_requested()

            if step % CLEANUP_INTERVAL == 0:
                gc.collect()
                mx.clear_cache()

        if progress.step > last_saved_step:
            save_checkpoint(last_val_loss, last_sample)
            last_saved_step = progress.step

    except KeyboardInterrupt:
        print("\nTraining interrupted by user or termination signal")
        loss = grads = accumulated_grads = None
        emergency_save()

    except Exception as e:
        print(f"\nTraining crashed: {type(e).__name__}: {e}")
        loss = grads = accumulated_grads = None
        emergency_save()
        raise

    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        run.close()
        elapsed_hours = (time.time() - process_start) / 3600
        print(f"Process runtime: {elapsed_hours:.2f} h")


if __name__ == "__main__":
    main()
