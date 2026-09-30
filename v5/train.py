import gc
import argparse
import hashlib
import json
from functools import lru_cache
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
from drafts import prepare_self_draft_block
from import_cuda import import_cuda_checkpoint
from model import PhobetorModel
from sampler import PhobetorSampler, SamplerConfig
from training_state import load_best_loss, restore_latest, save_best, save_checkpoint

SEQ_LEN = 1024
BATCH_SIZE = 4
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
SAMPLE_MAX_NEW_TOKENS = 32
VAL_INTERVAL = 500
VAL_BATCHES = 64
VAL_BATCH_SIZE = 2
BLOCK_VAL_BATCHES = 8
SAVE_INTERVAL = 2500
MAX_CHECKPOINTS = 3
MAX_SWAP_ALLOWED_GB = 1.0
MAX_SWAP_WRITES_GB = 4.0
ROOT = Path(__file__).resolve().parent.parent
V5_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = V5_DIR / 'checkpoints'
SOURCE_CHECKPOINT = ROOT / 'cuda_v4/checkpoints_cuda/emergency_checkpoint'
SAMPLE_LOG_PATH = V5_DIR / 'latest_samples.log'
TOKENIZER_PATH = ROOT / 'tokenizer'
TRAIN_DATA_PATH = ROOT / 'data/v4_books/train_tokens.bin'
VAL_DATA_PATH = ROOT / 'data/v4_books/val_tokens.bin'
DATA_MANIFEST_PATH = ROOT / 'data/v4_books/manifest.json'
MLX_CACHE_LIMIT_GIB = 1
AIM_REPO = V5_DIR / '.aim'
SAMPLE_PROMPTS = ('One morning, a little girl',
                  'At normal atmospheric pressure, water boils at',
                  'Two plus three equals')
SAMPLE_LABELS = ('Speech', 'Physics', 'Addition')
MODEL_CONFIG = dict(d_model=1024, n_heads=16, d_ff=3072, n_layers=8,
                    recurrent_passes=2, gradient_checkpointing=True,
                    mamba_expand=2.0, d_state=64, mamba_head_dim=64, d_conv=4)
SAMPLER_CONFIG = SamplerConfig(context_window=1024, block_size=16, overlap=12,
                               refinement_steps=12, temperature=0.8,
                               audit_every=1, audit_start_fraction=0.0,
                               audit_end_fraction=1.0, audit_candidates=16,
                               final_polish_rounds=0)
VALIDATION_CONFIG = dict(version=5, objective='conditional_block_self_draft_denoising', shuffle_seed=1338, mask_seed=20_000_000, batches=VAL_BATCHES,
                         batch_size=VAL_BATCH_SIZE, seq_len=SEQ_LEN, block_size=16, overlap=12, denoising_steps=12)
TRAINING_CONFIG = dict(version=5, objective='conditional_block_self_draft_denoising', block_size=16, overlap=12, denoising_steps=12,
                       stages='full_50_clean_25_self_draft_25_v1', loss='mean_hidden_ce',
                       batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM_STEPS,
                       seq_len=SEQ_LEN, seed=SEED,
                       peak_lr=PEAK_LR, min_lr=MIN_LR, warmup_steps=WARMUP_STEPS,
                       total_steps=TOTAL_STEPS)


def choose_prefix_length(seq_len, sample_id):
    maximum = seq_len-SAMPLER_CONFIG.block_size
    short = [n for n in (0, 4, 8, 16, 32, 64, 128) if n <= maximum]
    long = [n for n in (256, 512, 768, maximum) if n <= maximum]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, sample_id]))
    options = short if not long or rng.random() < 0.5 else long
    return int(rng.choice(options))


@lru_cache(maxsize=32)
def _noise_cycle(cycle, block_size, seed):
    if block_size < 2 or block_size % 2:
        raise ValueError('Balanced noise requires an even block size >=2')
    rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 4001]))
    low = rng.permutation(np.arange(1, block_size//2+1))
    high = rng.permutation(np.arange(block_size//2+1, block_size+1))
    return tuple(int(n) for pair in zip(low, high) for n in pair)


def balanced_mask_counts(sample_ids, block_size=None, seed=SEED):
    block_size = block_size or SAMPLER_CONFIG.block_size
    return [_noise_cycle(int(i)//block_size, block_size, seed)[int(i)%block_size]
            for i in sample_ids]


def prepare_training_block(clean, key, mask_id, prefix_length, mask_counts, overlap=None):
    batch, length = clean.shape
    block = SAMPLER_CONFIG.block_size
    if not 0 <= prefix_length <= length-block:
        raise ValueError('Prefix and block do not fit source sequence')
    if len(mask_counts) != batch or any(not 1 <= n <= block for n in mask_counts):
        raise ValueError('One mask count in [1, block_size] is required per row')
    k_crop, k_overlap, k_mask = mx.random.split(key, num=3)
    counts = mx.array(mask_counts)[:, None]
    span = prefix_length+block
    starts = mx.random.randint(0, length-span+1, shape=(batch,1), key=k_crop)
    cropped = mx.take_along_axis(clean, starts+mx.arange(span)[None,:], axis=1)
    targets = cropped[:, prefix_length:]
    scores = mx.random.uniform(shape=targets.shape, key=k_mask)
    ranks = mx.argsort(mx.argsort(scores, axis=-1), axis=-1)
    mask = ranks < counts
    if overlap is None:
        overlap = mx.random.uniform(shape=(batch,1), key=k_overlap) < 0.5
    tail_mask = mx.arange(block)[None,:] >= SAMPLER_CONFIG.overlap
    use_tail = (counts == SAMPLER_CONFIG.stride) & mx.array(overlap)
    mask = mx.where(use_tail, tail_mask, mask)
    noisy = mx.concatenate([cropped[:, :prefix_length], mx.where(mask, mask_id, targets)], axis=1)
    weights = mask.astype(mx.float32)*block/counts
    return noisy, targets, weights


def verify_corpus():
    def file_hash(filename):
        digest = hashlib.sha256()
        with open(filename, 'rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    path = Path(DATA_MANIFEST_PATH)
    manifest = json.loads(path.read_text())
    if manifest.get('version') != 4 or manifest.get('name') != 'phobetor_v4_plain_english':
        raise ValueError('V4 requires its own prepared corpus; run prepare_data.py')
    content = dict(manifest)
    corpus_id = content.pop('corpus_id')
    if hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() != corpus_id:
        raise ValueError('Corpus manifest fingerprint differs')
    if manifest['tokenizer_sha256'] != file_hash(Path(TOKENIZER_PATH)/'tokenizer.json'):
        raise ValueError('Corpus tokenizer differs from the current tokenizer')
    for split, filename in (('train', TRAIN_DATA_PATH), ('val', VAL_DATA_PATH)):
        info = manifest['splits'][split]
        if Path(filename).stat().st_size != info['tokens']*2 or file_hash(filename) != info['sha256']:
            raise ValueError(f'{split} data differs from the prepared V4 corpus')
    return manifest['corpus_id']


def block_loss(model, noisy, targets, weights):
    logits = model(noisy, output_start=noisy.shape[1]-targets.shape[1])
    return mx.sum(nn.losses.cross_entropy(logits, targets)*weights)/targets.size


def repetition_metrics(ids):
    grams = [tuple(ids[i:i+4]) for i in range(max(0, len(ids)-3))]
    return dict(generated_tokens=len(ids), unique_token_fraction=len(set(ids))/max(1, len(ids)),
                repeated_4gram_fraction=1-len(set(grams))/max(1, len(grams)) if grams else 0.0)


def console_table(headers, rows):
    def cell(value):
        return json.dumps(str(value), ensure_ascii=False)[1:-1].replace('\u2028', r'\u2028').replace('\u2029', r'\u2029')
    table = [[cell(v) for v in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    border = '+' + '+'.join('-'*(w+2) for w in widths) + '+'
    line = lambda row: '| ' + ' | '.join(v.ljust(w) for v,w in zip(row,widths)) + ' |'
    return '\n'.join([border, line(table[0]), border, *[line(row) for row in table[1:]], border])


def format_progress(step, total, loss, lr, grad, speed, seconds, interval_seconds, memory):
    return (f'Step {step}/{total} | Loss {loss:.4f} | LR {lr:.2e} | Grad {grad:.2f} | '
            f'{speed:.0f} tok/s | {seconds:.2f} s / {interval_seconds:.2f} s | {memory}')


def format_validation(step, metrics):
    keys = ('val_loss', 'val_continuation_loss', 'val_overlap_loss', 'val_refinement_loss', 'val_self_draft_loss')
    return console_table(['VAL step', 'Loss', 'Continue', 'Overlap', 'Refine', 'Self draft'],
                         [[step, *[f'{metrics[k]:.4f}' for k in keys]]])


def format_samples(step, records):
    rows = [[r['label'], r['prompt']+' >>> '+r['continuation'], r['generated_tokens'],
             f"{r['unique_token_fraction']:.1%}", f"{r['repeated_4gram_fraction']:.1%}"] for r in records]
    return console_table([str(step), 'Prompt >>> continuation', 'Tokens', 'Unique', 'Repeat-4'], rows)


def evaluate(model, dataset, mask_id, sampler):
    previous_mode = model.training
    model.eval()
    try:
        stage_loss = 0.0
        by_count = {n: [] for n in range(1, SAMPLER_CONFIG.block_size+1)}
        for i in range(VAL_BATCHES):
            clean = dataset.get_batch(VAL_BATCH_SIZE, batch_index=i)
            counts = balanced_mask_counts(range(i*VAL_BATCH_SIZE, (i+1)*VAL_BATCH_SIZE), seed=SEED+1)
            prepared = prepare_training_block(clean, mx.random.key(20_000_000+i), mask_id,
                                              choose_prefix_length(SEQ_LEN, 20_000_000+i), counts)
            noisy, targets, weights = prepared
            logits = model(noisy, output_start=noisy.shape[1]-targets.shape[1])
            losses = mx.mean(nn.losses.cross_entropy(logits, targets)*weights, axis=1).tolist()
            stage_loss += sum(losses)/len(losses)
            for n, value in zip(counts, losses): by_count[n].append(value)
        metrics = {'val_loss': stage_loss / VAL_BATCHES}
        metrics.update({f'val_mask_{n:02d}_loss': sum(values)/len(values)
                        for n, values in by_count.items() if values})
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
                mask_counts=[max(1, PhobetorSampler._next_mask_count(
                    block, SAMPLER_CONFIG.refinement_steps-2, SAMPLER_CONFIG.refinement_steps))]*VAL_BATCH_SIZE,
            )
            late_loss += float(block_loss(model, *prepared).item())
        metrics['val_refinement_loss'] = late_loss / BLOCK_VAL_BATCHES
        draft_loss = 0.0
        for i in range(BLOCK_VAL_BATCHES):
            clean = dataset.get_batch(VAL_BATCH_SIZE, batch_index=i)
            prepared = prepare_self_draft_block(
                model, sampler, clean, mx.random.key(40_000_000+i), mask_id,
                choose_prefix_length(SEQ_LEN, 40_000_000+i), [4, 12],
                prepare_training_block)
            draft_loss += float(block_loss(model, *prepared).item())
        metrics['val_self_draft_loss'] = draft_loss / BLOCK_VAL_BATCHES
        if not all(math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('Non-finite validation result')
        return metrics
    finally:
        model.train(previous_mode)


def make_optimizer(total_steps=None):
    total_steps = TOTAL_STEPS if total_steps is None else total_steps
    warmup = min(WARMUP_STEPS, max(1, total_steps-1))
    schedule = opt.join_schedules([
        opt.linear_schedule(0.0, PEAK_LR, warmup),
        opt.cosine_decay(PEAK_LR, max(1, total_steps-warmup), end=MIN_LR),
    ], [warmup])
    def exempt(path, weight):
        return weight.ndim < 2 or path.rsplit('.', 1)[-1] in {'A_log', 'D', 'dt_bias'}
    optimizer = opt.MultiOptimizer([
        opt.AdamW(learning_rate=schedule, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.0, bias_correction=True),
        opt.AdamW(learning_rate=schedule, betas=[0.9, 0.95], eps=1e-8, weight_decay=WEIGHT_DECAY, bias_correction=True),
    ], filters=[exempt])
    return optimizer, schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop-step', type=int)
    args = parser.parse_args()
    if SEQ_LEN < SAMPLER_CONFIG.block_size:
        raise ValueError('Training context is shorter than the generation block')
    SAMPLER_CONFIG.validate()
    corpus_id = verify_corpus()
    training_config = {**TRAINING_CONFIG, 'corpus_id': corpus_id}
    validation_config = {**VALIDATION_CONFIG, 'corpus_id': corpus_id}
    mx.set_cache_limit(MLX_CACHE_LIMIT_GIB*1024**3)
    initial_swap = psutil.swap_memory()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    mask_id = tokenizer.mask_token_id
    if mask_id is None or tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must define MASK and EOS')
    train_data = TextDataset(TRAIN_DATA_PATH, SEQ_LEN, shuffle_seed=SEED, repeat=False)
    val_data = TextDataset(VAL_DATA_PATH, SEQ_LEN, shuffle_seed=1338)
    single_pass_steps = train_data.num_sequences // (BATCH_SIZE*GRAD_ACCUM_STEPS)
    final_step = min(TOTAL_STEPS, single_pass_steps)
    if final_step < 1:
        raise ValueError('Training corpus is too small for one full optimizer update')
    training_config.update(data_policy='single_pass_no_replacement', schedule_steps=single_pass_steps)
    mx.random.seed(SEED)
    model = PhobetorModel(vocab_size=len(tokenizer), **MODEL_CONFIG)
    optimizer, schedule = make_optimizer(single_pass_steps)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    progress, resumed = restore_latest(CHECKPOINT_DIR, model, optimizer, MODEL_CONFIG, len(tokenizer))
    if not resumed:
        progress, source = import_cuda_checkpoint(
            SOURCE_CHECKPOINT, model, optimizer, MODEL_CONFIG, len(tokenizer),
            corpus_id, schedule)
        resumed = {'source_checkpoint': str(SOURCE_CHECKPOINT), 'source_step': source['step']}
        print(f'Imported CUDA V4 checkpoint at step {progress.step}: {SOURCE_CHECKPOINT}', flush=True)
    if 'training_config' in resumed and resumed['training_config'] != training_config:
        raise ValueError('Checkpoint training configuration differs; use a separate checkpoint directory')
    if args.stop_step is not None:
        if args.stop_step < 1:
            raise ValueError('--stop-step must be positive')
        final_step = min(final_step, args.stop_step)
    if final_step <= progress.step:
        print(f'No training left: checkpoint step {progress.step}, requested final step {final_step}')
        return
    mx.eval(model.parameters(), optimizer.state)
    model.train()
    sampler = PhobetorSampler(model, tokenizer, SAMPLER_CONFIG, SEED)
    best_loss = load_best_loss(CHECKPOINT_DIR, validation_config, metric='val_continuation_loss')
    last_metrics = resumed.get('validation_metrics', {})
    last_val_step = resumed.get('val_step')
    last_sample = resumed.get('sample', '')
    last_saved_step = progress.step
    stopped = False
    run = Run(repo=str(AIM_REPO), experiment='phobetor_v5_self_draft')
    run['model'] = MODEL_CONFIG
    run['training'] = training_config
    run['sampler'] = SAMPLER_CONFIG.to_dict()
    run['validation'] = validation_config
    run['sample_evaluation'] = dict(prompts=list(SAMPLE_PROMPTS), seed=SEED, max_new_tokens=SAMPLE_MAX_NEW_TOKENS)
    run['resumed_from_step'] = progress.step
    run['checkpoint_policy'] = dict(periodic_interval=SAVE_INTERVAL, periodic_keep=MAX_CHECKPOINTS, emergency_keep=1, best_keep=1)
    print(f'V5: {MODEL_CONFIG["n_layers"]} hybrid blocks x {MODEL_CONFIG["recurrent_passes"]} shared passes')
    print(f'Generation: window={SAMPLER_CONFIG.block_size}, editable overlap={SAMPLER_CONFIG.overlap}, '
          f'stride={SAMPLER_CONFIG.stride}, denoising steps={SAMPLER_CONFIG.refinement_steps}')
    print(f'Training: 50% full-mask + 25% clean partial + 25% self-draft, batch={BATCH_SIZE} x accumulation={GRAD_ACCUM_STEPS}; '
          f'steps {progress.step+1:,}..{final_step:,}; single pass, no repeated examples')
    print(f'Noise: own draft tokens retained at confident positions, low-confidence positions remasked; '
          f'corpus={corpus_id[:12]}')
    print(f'Checkpoints: every {SAVE_INTERVAL} steps, keep {MAX_CHECKPOINTS} periodic + 1 emergency + 1 best.')
    print('Generation uses masked confidence checks in addition to denoising fills '
          '(default: 12 fills + 22 checks per window).')
    print('tok/s counts processed context + target tokens. '
          'Time: last training step / elapsed since the previous progress line, '
          'including evaluation and saves. RAM is system-wide.')
    print(f'Samples: three fixed prompts, seed={SEED}, up to {SAMPLE_MAX_NEW_TOKENS} NEW tokens; '
          'prompt stays fixed. Detailed mask-level validation and MLX memory metrics remain in Aim.')

    def log_memory(phase):
        ram, swap = psutil.virtual_memory(), psutil.swap_memory()
        gib = 1024**3
        metrics = dict(ram_percent=ram.percent, ram_available_gib=ram.available/gib,
                       swap_used_gib=swap.used/gib,
                       swap_growth_gib=(swap.used-initial_swap.used)/gib,
                       mlx_active_gib=mx.get_active_memory()/gib,
                       mlx_cache_gib=mx.get_cache_memory()/gib,
                       mlx_peak_gib=mx.get_peak_memory()/gib)
        for name, value in metrics.items():
            run.track(value, name=name, step=progress.step, context={'phase': phase})
        memory = f'RAM {ram.percent:.1f}%'
        if phase == 'memory_guard':
            memory += (f' | Swap {metrics["swap_used_gib"]:.2f} GiB '
                       f'({metrics["swap_growth_gib"]:+.2f} since start)')
        return memory


    def metadata():
        return dict(vocab_size=len(tokenizer), model_config=MODEL_CONFIG,
                    training_config=training_config, sampler_config=SAMPLER_CONFIG.to_dict(),
                    validation_config=validation_config, validation_metrics=last_metrics,
                    block_validation_config=dict(version=5, block_size=SAMPLER_CONFIG.block_size,
                                                 overlap=SAMPLER_CONFIG.overlap, batches=BLOCK_VAL_BATCHES),
                    val_step=last_val_step, val_loss=last_metrics.get('val_loss'), sample=last_sample,
                    sample_evaluation=dict(prompts=list(SAMPLE_PROMPTS), seed=SEED, max_new_tokens=SAMPLE_MAX_NEW_TOKENS))

    def snapshot(kind="periodic"):
        return save_checkpoint(CHECKPOINT_DIR, model, optimizer, progress, metadata(), MAX_CHECKPOINTS, kind=kind)

    def check_swap():
        current = psutil.swap_memory()
        growth = (current.used-initial_swap.used)/1024**3
        writes = (getattr(current, 'sout', 0)-getattr(initial_swap, 'sout', 0))/1024**3
        if growth > MAX_SWAP_ALLOWED_GB:
            print(f'Memory guard | {log_memory("memory_guard")}', flush=True)
            raise RuntimeError(f'New swap allocation {growth:.2f} GiB exceeded '
                               f'memory guard ({MAX_SWAP_ALLOWED_GB:.2f} GiB)')
        if writes > MAX_SWAP_WRITES_GB:
            print(f'Memory guard | {log_memory("memory_guard")}', flush=True)
            raise RuntimeError(f'Swap writes {writes:.2f} GiB exceeded '
                               f'memory guard ({MAX_SWAP_WRITES_GB:.2f} GiB)')

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
    interval_start, interval_steps = start_time, 0
    try:
        print(f'Training start | {log_memory("start")}', flush=True)
        for step in range(progress.step+1, final_step+1):
            stop_if_requested()
            check_swap()
            tick = time.monotonic()
            accumulated, total_loss, model_tokens, hidden_tokens = None, 0.0, 0, 0
            objective_losses = {'full': [], 'clean': [], 'draft': []}
            for micro in range(GRAD_ACCUM_STEPS):
                micro_id = (step-1)*GRAD_ACCUM_STEPS + micro
                clean = train_data.get_batch(BATCH_SIZE, batch_index=micro_id)
                key = mx.random.key(2_000_000+micro_id)
                objective = ('draft' if micro % 4 == 0 else 'clean' if micro % 4 == 3 else 'full')
                counts = ([4, 8, 12, 15] if objective == 'draft'
                          else [2, 4, 12, 14] if objective == 'clean'
                          else [SAMPLER_CONFIG.block_size] * BATCH_SIZE)
                hidden_tokens += sum(counts)
                prefix = choose_prefix_length(SEQ_LEN, micro_id)
                if objective == 'draft':
                    prepared = prepare_self_draft_block(
                        model, sampler, clean, key, mask_id, prefix, counts,
                        prepare_training_block)
                else:
                    prepared = prepare_training_block(
                        clean, key, mask_id, prefix, counts, overlap=False)
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
                model_tokens += prepared[0].size * (2 if objective == 'draft' else 1)
                scale = 1.0/GRAD_ACCUM_STEPS
                accumulated = tree_map(lambda g: g*scale, grads) if accumulated is None else tree_map(lambda a,g:a+g*scale, accumulated, grads)
                mx.eval(accumulated)
                total_loss += value*scale
                objective_losses[objective].append(value)
                loss = grads = None
                stop_if_requested()
            grad_norm = progress.update(model, optimizer, accumulated, total_loss,
                                        model_tokens, GRAD_CLIP)
            accumulated = None
            stop_if_requested()
            seconds = time.monotonic()-tick
            interval_steps += 1
            if step % LOG_INTERVAL == 0 or step == final_step:
                lr = float(schedule(mx.array(step-1)).item())
                tokens_per_second = model_tokens/seconds
                memory = log_memory('train')
                logged_at = time.monotonic()
                interval_seconds = logged_at-interval_start
                print(format_progress(step, final_step, total_loss, lr, grad_norm, tokens_per_second,
                                      seconds, interval_seconds, memory), flush=True)
                print('Objective loss | ' + ' | '.join(
                    f'{name} {sum(values)/len(values):.4f}' for name, values in objective_losses.items()), flush=True)
                run.track(interval_seconds, name='log_interval_seconds', step=step)
                run.track(interval_steps, name='log_interval_steps', step=step)
                interval_start, interval_steps = logged_at, 0
                for name, value in dict(train_loss=total_loss, learning_rate=lr, grad_norm=grad_norm,
                                        tokens_per_sec=tokens_per_second, step_seconds=seconds,
                                        input_tokens_per_step=model_tokens, hidden_tokens_per_step=hidden_tokens,
                                        hidden_tokens_per_sec=hidden_tokens/seconds).items():
                    run.track(value, name=name, step=step)
            if step % SAMPLE_INTERVAL == 0:
                model.eval()
                sample_records = []
                try:
                    for label, prompt in zip(SAMPLE_LABELS, SAMPLE_PROMPTS):
                        ids = sampler.generate_ids(tokenizer.encode(prompt, add_special_tokens=False),
                                                   max_new_tokens=SAMPLE_MAX_NEW_TOKENS, min_new_tokens=8)
                        continuation = tokenizer.decode(ids, skip_special_tokens=True)
                        text = prompt+continuation
                        metrics = repetition_metrics(ids)
                        if prompt == SAMPLE_PROMPTS[0]: last_sample = text
                        sample_records.append(dict(label=label, prompt=prompt, continuation=continuation, **metrics))
                        context = {'prompt':prompt}
                        run.track(Text(text), name='sample', step=step, context=context)
                        for name,value in metrics.items():run.track(value,name='sample_'+name,step=step,context=context)
                        with open(SAMPLE_LOG_PATH, 'a') as f:
                            f.write(f'Step {step:06d}\nPrompt: {prompt}\nMetrics: {metrics}\nPhobetor: {text!r}\n'+ '='*80+'\n')
                        stop_if_requested()
                finally:
                    model.train()
                print(format_samples(step, sample_records), flush=True)
                log_memory('after_sampling')
            if step % VAL_INTERVAL == 0 or step == final_step:
                last_metrics = evaluate(model, val_data, mask_id, sampler)
                last_val_step = step
                print(format_validation(step, last_metrics), flush=True)
                for name,value in last_metrics.items():run.track(value,name=name,step=step)
                if last_metrics['val_continuation_loss'] < best_loss:
                    save_best(CHECKPOINT_DIR, model, {**metadata(), 'step':step, 'tokens_seen':progress.tokens_seen})
                    best_loss = last_metrics['val_continuation_loss']
                log_memory('after_validation')
                stop_if_requested()
            if step % SAVE_INTERVAL == 0:
                snapshot()
                last_saved_step = step
                gc.collect(); mx.clear_cache()
        if progress.step > last_saved_step:
            snapshot(kind="emergency")
            last_saved_step = progress.step
        if progress.step >= single_pass_steps:
            print('Single pass completed. Final checkpoint saved; training will not restart the corpus.')
    except KeyboardInterrupt:
        print('\nTraining interrupted; preserving completed updates.')
    finally:
        loss = grads = accumulated = None
        if progress.step > last_saved_step and progress.safe_to_save:
            gc.collect(); mx.clear_cache()
            try:
                snapshot(kind="emergency")
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
