import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from characters import VOCAB_SIZE, decode, encode
from dataset import CharacterDataset
from model import PhobetorModel
from noise import STEPS, corrupt
from sampler import CharacterSampler, SamplerConfig
from training_state import checkpoint_paths, load_best_loss, make_optimizer, restore_checkpoint, save_best, save_checkpoint


SEQ_LEN = 1024
BLOCK_SIZE = 256
OVERLAP = 192
OLD_START_NOISE = 8
EFFECTIVE_BATCH = 16
SEED = 1337
WARMUP_STEPS = 2000
PEAK_LR = 2.5e-4
MIN_LR = 2.5e-5
GRAD_CLIP = 1.0
LOG_INTERVAL = 25
VAL_INTERVAL = 1000
SAMPLE_INTERVAL = 2500
SAVE_INTERVAL = 2500
MODEL_CONFIG = dict(d_model=1024, n_heads=16, d_ff=3072, n_layers=16,
                    recurrent_passes=1, gradient_checkpointing=True)
SAMPLER_CONFIG = SamplerConfig()
PROMPTS = ('One morning, a little girl',
           'At normal atmospheric pressure, water boils at',
           'Two plus three equals')
STAGE = 'initial_overlap_half_uniform_t_v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/v5_characters')
    parser.add_argument('--checkpoint-dir', default=str(Path(__file__).resolve().parent / 'checkpoints_v5_cuda'))
    parser.add_argument('--micro-batch', type=int, default=16)
    parser.add_argument('--stop-step', type=int)
    parser.add_argument('--skip-samples', action='store_true')
    parser.add_argument('--skip-validation', action='store_true')
    parser.add_argument('--aim', action='store_true')
    return parser.parse_args()


def verify_corpus(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    content = dict(manifest)
    corpus_id = content.pop('corpus_id')
    if hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() != corpus_id:
        raise ValueError('Character corpus manifest has changed')
    if manifest['version'] != 5 or manifest['vocab_size'] != VOCAB_SIZE:
        raise ValueError('Wrong character corpus or alphabet')
    for split in ('train', 'val'):
        info = manifest['splits'][split]
        if (path / info['file']).stat().st_size != info['characters']:
            raise ValueError(f'{split} file size differs from manifest')
    return manifest


def learning_rate(step, final_step):
    warmup = min(WARMUP_STEPS, max(1, final_step - 1))
    if step < warmup:
        return PEAK_LR * step / warmup
    progress = min(1.0, (step - warmup) / max(1, final_step - warmup))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def noise_level(sample_id):
    pair = sample_id // 2
    return STEPS if pair % 2 == 0 else (pair // 2 * 13) % (STEPS - 1) + 1


def prepare_block(clean, sample_ids, seed):
    batch = clean.shape[0]
    prefix_length = (0, 16, 64, 128, 256, 512, 768)[seed % 7]
    span = prefix_length + BLOCK_SIZE
    rng = np.random.default_rng(SEED + seed)
    starts = rng.integers(0, SEQ_LEN - span + 1, size=batch)
    indices = torch.as_tensor(starts[:, None] + np.arange(span)[None], device=clean.device)
    cropped = clean.gather(1, indices)
    targets = cropped[:, prefix_length:]
    active_levels = torch.empty_like(targets)
    weights = torch.ones_like(targets, dtype=torch.float32)
    for index, sample_id in enumerate(sample_ids):
        level = noise_level(sample_id)
        active_levels[index].fill_(level)
        if sample_id % 2:
            active_levels[index, :OVERLAP] = min(level, OLD_START_NOISE)
            weights[index, OVERLAP:] = 4.0
    noisy = corrupt(targets, active_levels, VOCAB_SIZE)
    model_tokens = torch.cat((cropped[:, :prefix_length], noisy), dim=1)
    model_levels = torch.cat((torch.zeros_like(cropped[:, :prefix_length]), active_levels), dim=1)
    return model_tokens, model_levels, targets, weights


def loss_for(model, prepared):
    tokens, levels, targets, weights = prepared
    logits = model(tokens, levels, output_start=tokens.shape[1] - BLOCK_SIZE).float()
    losses = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1),
                             reduction='none').reshape_as(targets)
    return (losses * weights).sum() / weights.sum()


@torch.inference_mode()
def evaluate(model, dataset, device):
    was_training = model.training
    model.eval()
    result = {}
    try:
        for label, level, overlap in [('full_noise', 32, False),
                                       ('low_noise', 4, False),
                                       ('overlap', 32, True)]:
            values = []
            for index in range(8):
                clean = dataset.get_rows(index, 1, device)
                prefix_length = (0, 16, 64, 128, 256, 512, 768, 128)[index]
                targets = clean[:, prefix_length:prefix_length + BLOCK_SIZE]
                noise_levels = torch.full_like(targets, level)
                if overlap:
                    noise_levels[:, :OVERLAP] = OLD_START_NOISE
                generator = torch.Generator(device=device).manual_seed(SEED + 1000 + index)
                noisy = corrupt(targets, noise_levels, VOCAB_SIZE, generator)
                prefix = clean[:, :prefix_length]
                tokens = torch.cat((prefix, noisy), dim=1)
                levels = torch.cat((torch.zeros_like(prefix), noise_levels), dim=1)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(tokens, levels, output_start=prefix_length).float()
                start = OVERLAP if overlap else 0
                values.append(F.cross_entropy(logits[:, start:].reshape(-1, VOCAB_SIZE),
                                              targets[:, start:].reshape(-1)).item())
            result[f'val_{label}_loss'] = float(np.mean(values))
        return result
    finally:
        model.train(was_training)


@torch.inference_mode()
def sample_outputs(model, step, path):
    was_training = model.training
    model.eval()
    try:
        sampler = CharacterSampler(model, SAMPLER_CONFIG, SEED)
        rows = []
        for label, prompt in zip(('Speech', 'Physics', 'Addition'), PROMPTS):
            continuation = decode(sampler.generate_ids(encode(prompt), 128))
            rows.append((label, prompt, continuation))
            print(f'{step} | {label} | {prompt} >>> {continuation!r}', flush=True)
        with path.open('a') as output:
            output.write(json.dumps({'step': step, 'samples': rows}, ensure_ascii=False) + '\n')
    finally:
        model.train(was_training)


def main():
    args = parse_args()
    if args.micro_batch < 1 or EFFECTIVE_BATCH % args.micro_batch:
        raise ValueError('--micro-batch must divide 16')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA-enabled PyTorch and NVIDIA GPU are required')
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda')
    manifest = verify_corpus(args.data_dir)
    corpus_id = manifest['corpus_id']
    train_data = CharacterDataset(Path(args.data_dir) / 'train_chars.bin')
    val_data = CharacterDataset(Path(args.data_dir) / 'val_chars.bin', seed=SEED + 1, repeat=True)
    schedule_steps = train_data.num_sequences // EFFECTIVE_BATCH
    final_step = min(schedule_steps, args.stop_step) if args.stop_step else schedule_steps
    model = PhobetorModel(VOCAB_SIZE, **MODEL_CONFIG).to(device)
    optimizer = make_optimizer(model)
    checkpoint_dir = Path(args.checkpoint_dir)
    paths = checkpoint_paths(checkpoint_dir)
    if paths:
        failures = []
        for path in paths:
            try:
                resumed = restore_checkpoint(path, model, optimizer, MODEL_CONFIG, corpus_id)
                print(f'Resumed: {path}', flush=True)
                break
            except (ValueError, KeyError, OSError, RuntimeError) as error:
                failures.append(f'{path}: {error}')
        else:
            raise RuntimeError('No valid checkpoint: ' + '; '.join(failures))
    else:
        resumed = {'step': 0, 'tokens_seen': 0, 'train_loss': None}
        print('Starting character V5 from random initialization', flush=True)
    step_completed = resumed['step']
    if step_completed >= final_step:
        print(f'No training left: step {step_completed}/{final_step}', flush=True)
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stop_file = Path(__file__).resolve().parent / 'STOP'
    validation = resumed.get('validation_metrics', {})
    val_step = resumed.get('val_step')
    best_loss = load_best_loss(checkpoint_dir, 'val_full_noise_loss')
    tokens_seen = resumed['tokens_seen']
    last_loss = resumed.get('train_loss')
    training_config = {'corpus_id': corpus_id, 'data_policy': 'single_pass_no_replacement',
                       'seq_len': SEQ_LEN, 'objective': 'uniform_character_diffusion_x0',
                       'block_size': BLOCK_SIZE, 'overlap': OVERLAP,
                       'denoising_steps': STEPS, 'overlap_start_noise': OLD_START_NOISE,
                       'loss': 'weighted_full_block_x0_ce', 'peak_lr': PEAK_LR,
                       'min_lr': MIN_LR, 'warmup_steps': WARMUP_STEPS,
                       'seed': SEED, 'stages': STAGE, 'batch_size': args.micro_batch,
                       'grad_accum_steps': EFFECTIVE_BATCH // args.micro_batch}
    def metadata():
        return {'vocab_size': VOCAB_SIZE, 'model_config': MODEL_CONFIG,
                'training_config': training_config, 'sampler_config': SAMPLER_CONFIG.to_dict(),
                'step': step_completed, 'tokens_seen': tokens_seen, 'train_loss': last_loss,
                'val_step': val_step, 'validation_metrics': validation,
                'precision': 'bf16_autocast_fp32_weights', 'effective_batch': EFFECTIVE_BATCH}
    run = None
    if args.aim:
        from aim import Run
        run = Run(repo='.aim', experiment='phobetor_v5_character_cuda')
    stop_requested = False
    safe_to_save = True
    def request_stop(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print('Stop requested; finishing current update.', flush=True)
    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    last_saved_step = step_completed
    interval_start = time.monotonic()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    print(f'GPU {torch.cuda.get_device_name()} | V5 characters | '
          f'{sum(parameter.numel() for parameter in model.parameters()):,} parameters | '
          f'step {step_completed + 1}/{final_step} | micro-batch {args.micro_batch} x '
          f'{EFFECTIVE_BATCH // args.micro_batch}', flush=True)
    try:
        while step_completed < final_step:
            step = step_completed + 1
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            processed = 0
            for micro in range(EFFECTIVE_BATCH // args.micro_batch):
                first = (step - 1) * EFFECTIVE_BATCH + micro * args.micro_batch
                clean = train_data.get_rows(first, args.micro_batch, device)
                prepared = prepare_block(clean, range(first, first + args.micro_batch), first)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss = loss_for(model, prepared)
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError('Non-finite loss')
                (loss * args.micro_batch / EFFECTIVE_BATCH).backward()
                total_loss += value * args.micro_batch / EFFECTIVE_BATCH
                processed += prepared[0].numel()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP,
                                                        error_if_nonfinite=True))
            lr = learning_rate(step - 1, schedule_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            safe_to_save = False
            optimizer.step()
            torch.cuda.synchronize()
            safe_to_save = True
            step_completed = step
            tokens_seen += processed
            last_loss = total_loss
            seconds = time.monotonic() - tick
            if stop_file.exists():
                stop_file.unlink()
                stop_requested = True
            if step % LOG_INTERVAL == 0 or step == final_step:
                elapsed = time.monotonic() - interval_start
                print(f'Step {step}/{final_step} | Loss {total_loss:.4f} | LR {lr:.2e} | '
                      f'Grad {grad:.2f} | {processed / seconds:.0f} char/s | '
                      f'{seconds:.2f} s / {elapsed:.2f} s | RAM {psutil.virtual_memory().percent:.1f}% | '
                      f'VRAM peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GiB', flush=True)
                interval_start = time.monotonic()
                if run:
                    for name, value in [('train_loss', total_loss), ('learning_rate', lr),
                                        ('grad_norm', grad), ('characters_per_second', processed / seconds)]:
                        run.track(value, name=name, step=step)
            if stop_requested:
                path = save_checkpoint(checkpoint_dir, model, optimizer, metadata(), kind='emergency')
                print(f'Emergency checkpoint saved: {path}', flush=True)
                last_saved_step = step
                break
            if not args.skip_validation and (step % VAL_INTERVAL == 0 or step == final_step):
                validation = evaluate(model, val_data, device)
                val_step = step
                print(f'VAL {step} | full-noise {validation["val_full_noise_loss"]:.4f} | '
                      f'low-noise {validation["val_low_noise_loss"]:.4f} | '
                      f'overlap-new {validation["val_overlap_loss"]:.4f}', flush=True)
                if run:
                    for name, value in validation.items():
                        run.track(value, name=name, step=step)
                if validation['val_full_noise_loss'] < best_loss:
                    save_best(checkpoint_dir, model, metadata())
                    best_loss = validation['val_full_noise_loss']
            if not args.skip_samples and step % SAMPLE_INTERVAL == 0:
                sample_outputs(model, step, Path(__file__).resolve().parent / 'latest_samples_v5_cuda.log')
            if step % SAVE_INTERVAL == 0:
                path = save_checkpoint(checkpoint_dir, model, optimizer, metadata(), keep=3)
                print(f'Checkpoint saved: {path}', flush=True)
                last_saved_step = step
        if step_completed > last_saved_step:
            path = save_checkpoint(checkpoint_dir, model, optimizer, metadata(), kind='emergency')
            print(f'Final checkpoint saved: {path}', flush=True)
    except BaseException as error:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if safe_to_save and step_completed > last_saved_step:
            try:
                path = save_checkpoint(checkpoint_dir, model, optimizer, metadata(), kind='emergency')
                print(f'Emergency checkpoint saved after {type(error).__name__}: {path}', flush=True)
            except Exception as save_error:
                print(f'Emergency checkpoint failed: {save_error}', flush=True)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if run:
            run.close()


if __name__ == '__main__':
    main()
