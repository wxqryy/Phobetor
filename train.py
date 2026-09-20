"""Phobetor V3: eight shared hybrid blocks, two passes, fresh pretraining."""
import gc
import math
from pathlib import Path
import signal
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_map
import psutil
from aim import Run, Text
from transformers import AutoTokenizer

from dataset import TextDataset
from model import PhobetorModel
from sampler import PhobetorSampler, SamplerConfig
from training_state import load_best_loss, restore_latest, save_best, save_checkpoint

SEQ_LEN = 1024
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
TOTAL_STEPS = 250_000
WARMUP_STEPS = 2_000
PEAK_LR = 2.5e-4
MIN_LR = 2.5e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
SEED = 1337
LOG_INTERVAL = 25
SAMPLE_INTERVAL = 500
SAMPLE_MAX_NEW_TOKENS = 64
VAL_INTERVAL = 2_000
VAL_BATCHES = 64
VAL_BATCH_SIZE = 2
BLOCK_VAL_BATCHES = 8
SAVE_INTERVAL = 500
MAX_CHECKPOINTS = 3
MAX_SWAP_ALLOWED_GB = 1.0
MAX_SWAP_WRITES_GB = 4.0
CHECKPOINT_DIR = './checkpoints'
SAMPLE_LOG_PATH = './latest_samples.log'
TOKENIZER_PATH = './tokenizer'
TRAIN_DATA_PATH = './data/train_tokens.bin'
VAL_DATA_PATH = './data/val_tokens.bin'
AIM_REPO = './.aim'
SAMPLE_PROMPTS = ('The basic law of', 'Water boils when', 'A triangle has')
MODEL_CONFIG = dict(d_model=1024, n_heads=16, d_ff=3072, n_layers=8,
                    recurrent_passes=2, gradient_checkpointing=True,
                    mamba_expand=2.0, d_state=64, mamba_head_dim=64, d_conv=4)
SAMPLER_CONFIG = SamplerConfig(context_window=1024, block_size=32, overlap=24,
                               refinement_steps=16, temperature=0.8,
                               audit_every=0, final_polish_rounds=0)
VALIDATION_CONFIG = dict(version=3, objective='conditional_block_denoising', shuffle_seed=1338, mask_seed=20_000_000, batches=VAL_BATCHES,
                         batch_size=VAL_BATCH_SIZE, seq_len=SEQ_LEN, block_size=32, overlap=24, denoising_steps=16)
TRAINING_CONFIG = dict(version=3, objective='conditional_block_denoising', block_size=32, overlap=24, denoising_steps=16,
                       stages='uniform_sampler_stage', loss='mean_hidden_ce',
                       batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM_STEPS,
                       seq_len=SEQ_LEN, seed=SEED,
                       peak_lr=PEAK_LR, min_lr=MIN_LR, warmup_steps=WARMUP_STEPS,
                       total_steps=TOTAL_STEPS)


def choose_prefix_length(seq_len, sample_id):
    maximum = seq_len-SAMPLER_CONFIG.block_size
    short = [n for n in (0, 4, 8, 16, 32, 64, 128) if n <= maximum]
    long = [n for n in (256, 512, 768, 992) if n <= maximum]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, sample_id]))
    options = short if not long or rng.random() < 0.5 else long
    return int(rng.choice(options))


def prepare_training_block(clean, key, mask_id, prefix_length, stage=None, overlap=None):
    """Sample a conditional denoising stage of prefix + one editable window.

    Visible tokens are ground truth (teacher forcing), not sampled rollouts.
    Every example has >=1 mask. Each example averages CE over hidden tokens:
    a single late-stage mistake weighs more than one of 32 initial mistakes.
    This is a conditional denoising objective, not the old full-sequence bound.
    """
    batch, length = clean.shape
    block, steps = SAMPLER_CONFIG.block_size, SAMPLER_CONFIG.refinement_steps
    if not 0 <= prefix_length <= length-block:
        raise ValueError('Prefix and block do not fit source sequence')
    k_crop, k_stage, k_overlap, k_mask = mx.random.split(key, num=4)
    if stage is None:
        stage = int(mx.random.randint(0, steps, shape=(), key=k_stage).item())
    if not 0 <= stage < steps:
        raise ValueError('Invalid denoising stage')
    if overlap is None:
        overlap = bool((mx.random.uniform(shape=(), key=k_overlap) < 0.5).item())
    span = prefix_length+block
    starts = mx.random.randint(0, length-span+1, shape=(batch,1), key=k_crop)
    cropped = mx.take_along_axis(clean, starts+mx.arange(span)[None,:], axis=1)
    targets = cropped[:, prefix_length:]
    if stage == 0 and overlap:
        mask = mx.broadcast_to(mx.arange(block)[None,:] >= SAMPLER_CONFIG.overlap, targets.shape)
    else:
        count = block if stage == 0 else PhobetorSampler._next_mask_count(block, stage-1, steps)
        count = max(1, count)
        scores = mx.random.uniform(shape=targets.shape, key=k_mask)
        ranks = mx.argsort(mx.argsort(scores, axis=-1), axis=-1)
        mask = ranks < count
    noisy = mx.concatenate([cropped[:, :prefix_length], mx.where(mask, mask_id, targets)], axis=1)
    weights = mask.astype(mx.float32)*block/mx.sum(mask, axis=1, keepdims=True)
    return noisy, targets, weights


def block_loss(model, noisy, targets, weights):
    logits = model(noisy, output_start=noisy.shape[1]-targets.shape[1])
    return mx.sum(nn.losses.cross_entropy(logits, targets)*weights)/targets.size


def repetition_metrics(ids):
    grams = [tuple(ids[i:i+4]) for i in range(max(0, len(ids)-3))]
    return dict(generated_tokens=len(ids), unique_token_fraction=len(set(ids))/max(1, len(ids)),
                repeated_4gram_fraction=1-len(set(grams))/max(1, len(grams)) if grams else 0.0)


def evaluate(model, dataset, mask_id):
    previous_mode = model.training
    model.eval()
    try:
        stage_loss = 0.0
        for i in range(VAL_BATCHES):
            clean = dataset.get_batch(VAL_BATCH_SIZE, batch_index=i)
            prepared = prepare_training_block(clean, mx.random.key(20_000_000+i), mask_id,
                                              choose_prefix_length(SEQ_LEN, 20_000_000+i))
            stage_loss += float(block_loss(model, *prepared).item())
        metrics = {'val_loss': stage_loss / VAL_BATCHES}
        block = SAMPLER_CONFIG.block_size
        for name, old_tokens in (('val_continuation_loss', 0), ('val_overlap_loss', SAMPLER_CONFIG.overlap)):
            total = 0.0
            for i in range(BLOCK_VAL_BATCHES):
                prefix = min((4, 16, 64, 128, 256, 512, 768, 992)[i % 8], SEQ_LEN-block)
                clean = dataset.get_batch(VAL_BATCH_SIZE, batch_index=i)[:, :prefix+block]
                masked = mx.arange(clean.shape[1])[None, :] >= prefix + old_tokens
                noisy = mx.where(masked, mx.array(mask_id, dtype=clean.dtype), clean)
                start = prefix + old_tokens
                ce = nn.losses.cross_entropy(model(noisy, output_start=start), clean[:, start:])
                total += float(mx.mean(ce).item())
            metrics[name] = total / BLOCK_VAL_BATCHES
        late_loss = 0.0
        for i in range(BLOCK_VAL_BATCHES):
            clean = dataset.get_batch(VAL_BATCH_SIZE, batch_index=i)
            prepared = prepare_training_block(
                clean, mx.random.key(30_000_000+i), mask_id,
                choose_prefix_length(SEQ_LEN, 30_000_000+i),
                stage=SAMPLER_CONFIG.refinement_steps-1,
            )
            late_loss += float(block_loss(model, *prepared).item())
        metrics['val_refinement_loss'] = late_loss / BLOCK_VAL_BATCHES
        if not all(math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('Non-finite validation result')
        return metrics
    finally:
        model.train(previous_mode)


def make_optimizer():
    schedule = opt.join_schedules([
        opt.linear_schedule(0.0, PEAK_LR, WARMUP_STEPS),
        opt.cosine_decay(PEAK_LR, TOTAL_STEPS-WARMUP_STEPS, end=MIN_LR),
    ], [WARMUP_STEPS])
    def exempt(path, weight):
        return weight.ndim < 2 or path.rsplit('.', 1)[-1] in {'A_log', 'D', 'dt_bias'}
    optimizer = opt.MultiOptimizer([
        opt.AdamW(learning_rate=schedule, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.0, bias_correction=True),
        opt.AdamW(learning_rate=schedule, betas=[0.9, 0.95], eps=1e-8, weight_decay=WEIGHT_DECAY, bias_correction=True),
    ], filters=[exempt])
    return optimizer, schedule


def main():
    if SEQ_LEN < SAMPLER_CONFIG.block_size:
        raise ValueError('Training context is shorter than the generation block')
    SAMPLER_CONFIG.validate()
    initial_swap = psutil.swap_memory()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    mask_id = tokenizer.mask_token_id
    if mask_id is None or tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must define MASK and EOS')
    train_data = TextDataset(TRAIN_DATA_PATH, SEQ_LEN, shuffle_seed=SEED)
    val_data = TextDataset(VAL_DATA_PATH, SEQ_LEN, shuffle_seed=1338)
    mx.random.seed(SEED)
    model = PhobetorModel(vocab_size=len(tokenizer), **MODEL_CONFIG)
    optimizer, schedule = make_optimizer()
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    progress, resumed = restore_latest(CHECKPOINT_DIR, model, optimizer, MODEL_CONFIG, len(tokenizer))
    if resumed and resumed.get('training_config') != TRAINING_CONFIG:
        raise ValueError('Checkpoint training configuration differs; use a separate checkpoint directory')
    mx.eval(model.parameters(), optimizer.state)
    model.train()
    sampler = PhobetorSampler(model, tokenizer, SAMPLER_CONFIG, SEED)
    best_loss = load_best_loss(CHECKPOINT_DIR, VALIDATION_CONFIG)
    last_metrics = resumed.get('validation_metrics', {})
    last_val_step = resumed.get('val_step')
    last_sample = resumed.get('sample', '')
    last_saved_step = progress.step
    stopped = False
    run = Run(repo=AIM_REPO, experiment='phobetor_v3_hybrid_twopass')
    run['model'] = MODEL_CONFIG
    run['training'] = TRAINING_CONFIG
    run['sampler'] = SAMPLER_CONFIG.to_dict()
    run['validation'] = VALIDATION_CONFIG
    run['resumed_from_step'] = progress.step
    print(f'V3: {MODEL_CONFIG["n_layers"]} hybrid blocks x {MODEL_CONFIG["recurrent_passes"]} shared passes')
    print(f'Generation: window={SAMPLER_CONFIG.block_size}, editable overlap={SAMPLER_CONFIG.overlap}, '
          f'stride={SAMPLER_CONFIG.stride}, denoising steps={SAMPLER_CONFIG.refinement_steps}')
    print(f'Training: conditional block denoising, batch={BATCH_SIZE} x accumulation={GRAD_ACCUM_STEPS}; '
          f'steps {progress.step+1:,}..{TOTAL_STEPS:,}')

    def metadata():
        return dict(vocab_size=len(tokenizer), model_config=MODEL_CONFIG,
                    training_config=TRAINING_CONFIG, sampler_config=SAMPLER_CONFIG.to_dict(),
                    validation_config=VALIDATION_CONFIG, validation_metrics=last_metrics,
                    block_validation_config=dict(version=3, block_size=SAMPLER_CONFIG.block_size,
                                                 overlap=SAMPLER_CONFIG.overlap, batches=BLOCK_VAL_BATCHES),
                    val_step=last_val_step, val_loss=last_metrics.get('val_loss'), sample=last_sample)

    def snapshot():
        return save_checkpoint(CHECKPOINT_DIR, model, optimizer, progress, metadata(), MAX_CHECKPOINTS)

    def check_swap():
        current = psutil.swap_memory()
        if (current.used-initial_swap.used)/1024**3 > MAX_SWAP_ALLOWED_GB:
            raise RuntimeError('New swap allocation exceeded memory guard')
        if (getattr(current, 'sout', 0)-getattr(initial_swap, 'sout', 0))/1024**3 > MAX_SWAP_WRITES_GB:
            raise RuntimeError('Swap writes exceeded memory guard')

    def request_stop(signum, frame):
        nonlocal stopped
        if stopped:
            raise KeyboardInterrupt
        stopped = True
        print('\nStop requested; saving at the next safe boundary. Signal again to force stop.')

    def stop_if_requested():
        if stopped:
            raise KeyboardInterrupt

    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    grad_function = nn.value_and_grad(model, block_loss)
    def eager_grad(noisy, targets, weights):
        return grad_function(model, noisy, targets, weights)
    grad_runner = mx.compile(eager_grad, inputs=model.state, outputs=model.state)
    first_grad = True
    loss = grads = accumulated = None
    start_time = time.monotonic()
    try:
        for step in range(progress.step+1, TOTAL_STEPS+1):
            stop_if_requested()
            check_swap()
            tick = time.monotonic()
            accumulated, total_loss, model_tokens = None, 0.0, 0
            for micro in range(GRAD_ACCUM_STEPS):
                micro_id = (step-1)*GRAD_ACCUM_STEPS + micro
                clean = train_data.get_batch(BATCH_SIZE, batch_index=micro_id)
                key = mx.random.key(2_000_000+micro_id)
                prepared = prepare_training_block(clean, key, mask_id, choose_prefix_length(SEQ_LEN, micro_id))
                try:
                    loss, grads = grad_runner(*prepared)
                    mx.eval(loss, grads)
                except (RuntimeError, ValueError) as exc:
                    if not first_grad:
                        raise
                    loss = grads = None
                    mx.clear_cache()
                    print(f'Compiled gradient unavailable; using eager gradient: {exc}')
                    grad_runner = eager_grad
                    loss, grads = grad_runner(*prepared)
                    mx.eval(loss, grads)
                first_grad = False
                value = float(loss.item())
                if not math.isfinite(value):
                    raise FloatingPointError('Non-finite training loss before optimizer update')
                model_tokens += prepared[0].size
                scale = 1.0/GRAD_ACCUM_STEPS
                accumulated = tree_map(lambda g: g*scale, grads) if accumulated is None else tree_map(lambda a,g:a+g*scale, accumulated, grads)
                mx.eval(accumulated)
                total_loss += value*scale
                loss = grads = None
                stop_if_requested()
            grad_norm = progress.update(model, optimizer, accumulated, total_loss,
                                        model_tokens, GRAD_CLIP)
            accumulated = None
            stop_if_requested()
            seconds = time.monotonic()-tick
            if step % SAVE_INTERVAL == 0:
                snapshot()
                last_saved_step = step
            if step % LOG_INTERVAL == 0:
                lr = float(schedule(mx.array(step-1)).item())
                tokens_per_second = model_tokens/seconds
                print(f'Step {step:06d} | Loss {total_loss:.4f} | LR {lr:.2e} | Grad {grad_norm:.2f} | '
                      f'{tokens_per_second:.0f} tok/s | MLX {mx.get_active_memory()/1024**3:.1f} GB', flush=True)
                for name, value in dict(train_loss=total_loss, learning_rate=lr, grad_norm=grad_norm,
                                        tokens_per_sec=tokens_per_second).items():
                    run.track(value, name=name, step=step)
            if step % SAMPLE_INTERVAL == 0:
                model.eval()
                try:
                    for prompt in SAMPLE_PROMPTS:
                        ids = sampler.generate_ids(tokenizer.encode(prompt, add_special_tokens=False),
                                                   max_new_tokens=SAMPLE_MAX_NEW_TOKENS, min_new_tokens=8)
                        text = prompt+tokenizer.decode(ids, skip_special_tokens=True)
                        metrics = repetition_metrics(ids)
                        if prompt == SAMPLE_PROMPTS[0]: last_sample = text
                        print(f'SAMPLE {step} | {metrics} | {text!r}', flush=True)
                        context = {'prompt':prompt}
                        run.track(Text(text), name='sample', step=step, context=context)
                        for name,value in metrics.items():run.track(value,name='sample_'+name,step=step,context=context)
                        with open(SAMPLE_LOG_PATH, 'a') as f:
                            f.write(f'Step {step:06d}\nPrompt: {prompt}\nMetrics: {metrics}\nPhobetor: {text!r}\n'+ '='*80+'\n')
                        stop_if_requested()
                finally:
                    model.train()
            if step % VAL_INTERVAL == 0 or step == TOTAL_STEPS:
                last_metrics = evaluate(model, val_data, mask_id)
                last_val_step = step
                print(f'VAL {step}: {last_metrics}', flush=True)
                for name,value in last_metrics.items():run.track(value,name=name,step=step)
                if last_metrics['val_loss'] < best_loss:
                    save_best(CHECKPOINT_DIR, model, {**metadata(), 'step':step, 'tokens_seen':progress.tokens_seen})
                    best_loss = last_metrics['val_loss']
                stop_if_requested()
            if step % SAVE_INTERVAL == 0:
                gc.collect(); mx.clear_cache()
        if progress.step > 0:
            snapshot()
            last_saved_step = progress.step
    except KeyboardInterrupt:
        print('\nTraining interrupted; preserving completed updates.')
    finally:
        loss = grads = accumulated = None
        if progress.step > last_saved_step and progress.safe_to_save:
            gc.collect(); mx.clear_cache()
            try:
                snapshot()
                print(f'Emergency checkpoint saved at completed step {progress.step}')
            except Exception as exc:
                print(f'Emergency save failed; previous checkpoints preserved: {exc}')
        elif not progress.safe_to_save:
            print('Incomplete optimizer update; previous disk checkpoint preserved.')
        for sig, handler in old_handlers.items():signal.signal(sig, handler)
        run.close()
        print(f'Process runtime: {(time.monotonic()-start_time)/3600:.2f} h')
    return progress.step


if __name__ == '__main__':
    main()
