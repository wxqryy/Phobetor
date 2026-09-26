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
from transformers import AutoTokenizer

from dataset import TextDataset, balanced_mask_counts, choose_prefix_length, prepare_training_block
from model import PhobetorModel
from sampler import PhobetorSampler, SamplerConfig
from training_state import checkpoint_paths, load_best_loss, make_optimizer, restore_checkpoint, save_best, save_checkpoint


SEQ_LEN = 1024
EFFECTIVE_BATCH = 32
TOTAL_STEPS = 250000
WARMUP_STEPS = 2000
PEAK_LR = 2.5e-4
MIN_LR = 2.5e-5
GRAD_CLIP = 1.0
SEED = 1337
LOG_INTERVAL = 25
VAL_INTERVAL = 500
SAMPLE_INTERVAL = 500
SAVE_INTERVAL = 2500
VAL_BATCHES = 64
BLOCK_VAL_BATCHES = 8
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--micro-batch', type=int, default=4)
    parser.add_argument('--benchmark-steps', type=int, default=0)
    parser.add_argument('--benchmark-warmup', type=int, default=2)
    parser.add_argument('--stop-step', type=int)
    parser.add_argument('--checkpoint-dir', default='checkpoints_cuda')
    parser.add_argument('--source-checkpoint', default='source_checkpoint')
    parser.add_argument('--data-dir', default='data/v4_books')
    parser.add_argument('--tokenizer-dir', default='tokenizer')
    parser.add_argument('--skip-samples', action='store_true')
    parser.add_argument('--skip-validation', action='store_true')
    parser.add_argument('--aim', action='store_true')
    return parser.parse_args()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_corpus(data_dir, tokenizer_dir):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    if manifest.get('version') != 4 or manifest.get('name') != 'phobetor_v4_plain_english':
        raise ValueError('Expected V4 plain English corpus')
    content = dict(manifest)
    corpus_id = content.pop('corpus_id')
    if hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() != corpus_id:
        raise ValueError('Corpus manifest fingerprint differs')
    if file_hash(Path(tokenizer_dir) / 'tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer hash differs from corpus')
    for split in ('train', 'val'):
        info = manifest['splits'][split]
        path = data_dir / info['file']
        if path.stat().st_size != info['tokens'] * 2 or file_hash(path) != info['sha256']:
            raise ValueError(f'{split} corpus file differs from manifest')
    return manifest


def learning_rate(step_index, schedule_steps):
    warmup = min(WARMUP_STEPS, max(1, schedule_steps - 1))
    if step_index < warmup:
        return PEAK_LR * step_index / warmup
    progress = min(1.0, (step_index - warmup) / max(1, schedule_steps - warmup))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def block_loss(model, noisy, targets, weights):
    logits = model(noisy, output_start=noisy.shape[1] - targets.shape[1]).float()
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='none')
    return (ce.reshape_as(targets) * weights).mean()


def repetition_metrics(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(max(0, len(ids) - 3))]
    return len(ids), len(set(ids)) / max(1, len(ids)), 1 - len(set(grams)) / max(1, len(grams)) if grams else 0.0


def table(headers, rows):
    def cell(value):
        return json.dumps(str(value), ensure_ascii=False)[1:-1].replace('\u2028', r'\u2028').replace('\u2029', r'\u2029')
    cells = [[cell(value) for value in row] for row in (headers, *rows)]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    border = '+' + '+'.join('-' * (width + 2) for width in widths) + '+'
    return '\n'.join([border, '| ' + ' | '.join(value.ljust(width) for value, width in zip(cells[0], widths)) + ' |',
                      border, *['| ' + ' | '.join(value.ljust(width) for value, width in zip(row, widths)) + ' |'
                                for row in cells[1:]], border])


@torch.inference_mode()
def evaluate(model, dataset, mask_id, device):
    was_training = model.training
    model.eval()
    metrics = {}
    by_count = {n: [] for n in range(1, 17)}
    random_losses = []
    try:
        for i in range(VAL_BATCHES):
            clean = dataset.get_rows(i * 2, 2, device)
            counts = balanced_mask_counts(range(i * 2, (i + 1) * 2), seed=SEED + 1)
            prefix = choose_prefix_length(SEQ_LEN, 20_000_000 + i)
            noisy, targets, weights = prepare_training_block(clean, mask_id, prefix, counts, 20_000_000 + i)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logits = model(noisy, output_start=noisy.shape[1] - 16).float()
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='none').reshape_as(targets)
            losses = (ce * weights).mean(dim=1).tolist()
            random_losses.extend(losses)
            for count, value in zip(counts, losses):
                by_count[count].append(value)
        metrics['val_loss'] = float(np.mean(random_losses))
        for count, values in by_count.items():
            if values:
                metrics[f'val_mask_{count:02d}_loss'] = float(np.mean(values))
        for name, old_tokens in (('val_continuation_loss', 0), ('val_overlap_loss', 12)):
            losses = []
            for i in range(BLOCK_VAL_BATCHES):
                prefix = min((4, 16, 64, 128, 256, 512, 768, 992)[i], SEQ_LEN - 16)
                clean = dataset.get_rows(i * 2, 2, device)[:, :prefix + 16]
                noisy = clean.clone()
                noisy[:, prefix + old_tokens:] = mask_id
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    logits = model(noisy, output_start=prefix + old_tokens).float()
                losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                              clean[:, prefix + old_tokens:].reshape(-1)).item())
            metrics[name] = float(np.mean(losses))
        late_count = max(1, PhobetorSampler._next_mask_count(16, 10, 12))
        losses = []
        for i in range(BLOCK_VAL_BATCHES):
            clean = dataset.get_rows(i * 2, 2, device)
            prefix = choose_prefix_length(SEQ_LEN, 30_000_000 + i)
            prepared = prepare_training_block(clean, mask_id, prefix, [late_count] * 2, 30_000_000 + i)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                losses.append(block_loss(model, *prepared).item())
        metrics['val_refinement_loss'] = float(np.mean(losses))
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError('Non-finite validation result')
        return metrics
    finally:
        model.train(was_training)


def sample_outputs(model, tokenizer, step, sample_log):
    was_training = model.training
    model.eval()
    sampler = PhobetorSampler(model, tokenizer, SAMPLER_CONFIG, SEED)
    records = []
    try:
        for label, prompt in zip(SAMPLE_LABELS, SAMPLE_PROMPTS):
            ids = sampler.generate_ids(tokenizer.encode(prompt, add_special_tokens=False), 32, 8)
            continuation = tokenizer.decode(ids, skip_special_tokens=True)
            tokens, unique, repeat = repetition_metrics(ids)
            records.append([label, prompt + ' >>> ' + continuation, tokens, f'{unique:.1%}', f'{repeat:.1%}'])
        print(table([str(step), 'Prompt >>> continuation', 'Tokens', 'Unique', 'Repeat-4'], records), flush=True)
        with sample_log.open('a') as stream:
            stream.write(json.dumps({'step': step, 'samples': records}, ensure_ascii=False) + '\n')
        return records
    finally:
        model.train(was_training)


def train_update(model, optimizer, dataset, mask_id, step, micro_batch, device, schedule_steps):
    accum = EFFECTIVE_BATCH // micro_batch
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    tokens = 0
    hidden = 0
    for micro in range(accum):
        first = (step - 1) * EFFECTIVE_BATCH + micro * micro_batch
        clean = dataset.get_rows(first, micro_batch, device)
        micro_id = (step - 1) * accum + micro
        counts = balanced_mask_counts(range(first, first + micro_batch))
        prefix = choose_prefix_length(SEQ_LEN, micro_id)
        prepared = prepare_training_block(clean, mask_id, prefix, counts, 2_000_000 + micro_id)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss = block_loss(model, *prepared)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError('Non-finite training loss')
        (loss / accum).backward()
        total_loss += value / accum
        tokens += prepared[0].numel()
        hidden += sum(counts)
    gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True))
    lr = learning_rate(step - 1, schedule_steps)
    for group in optimizer.param_groups:
        group['lr'] = lr
    return total_loss, gradient, lr, tokens, hidden


def main():
    args = parse_args()
    if args.micro_batch < 1 or EFFECTIVE_BATCH % args.micro_batch:
        raise ValueError('--micro-batch must divide 32')
    if args.benchmark_steps < 0 or args.benchmark_warmup < 0:
        raise ValueError('Benchmark steps must be nonnegative')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU and CUDA-enabled PyTorch are required')
    device = torch.device('cuda')
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = verify_corpus(args.data_dir, args.tokenizer_dir)
    corpus_id = manifest['corpus_id']
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    if len(tokenizer) != manifest['vocab_size'] or tokenizer.mask_token_id != manifest['mask_token_id']:
        raise ValueError('Tokenizer identity differs from data manifest')
    train_data = TextDataset(Path(args.data_dir) / 'train_tokens.bin', SEQ_LEN, shuffle_seed=SEED)
    val_data = TextDataset(Path(args.data_dir) / 'val_tokens.bin', SEQ_LEN, shuffle_seed=1338, repeat=True)
    schedule_steps = train_data.num_sequences // EFFECTIVE_BATCH
    final_step = min(TOTAL_STEPS, schedule_steps)
    if args.stop_step is not None:
        final_step = min(final_step, args.stop_step)
    model = PhobetorModel(len(tokenizer), **MODEL_CONFIG).to(device)
    optimizer = make_optimizer(model)
    cuda_paths = checkpoint_paths(args.checkpoint_dir)
    if cuda_paths:
        errors = []
        for path in cuda_paths:
            try:
                resumed = restore_checkpoint(path, model, optimizer, MODEL_CONFIG, corpus_id)
                print(f'Resumed CUDA checkpoint: {path}', flush=True)
                break
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                errors.append(f'{path}: {exc}')
        else:
            raise RuntimeError('No valid CUDA checkpoint: ' + '; '.join(errors))
    else:
        resumed = restore_checkpoint(args.source_checkpoint, model, optimizer, MODEL_CONFIG, corpus_id)
        print(f'Imported complete MLX checkpoint: {args.source_checkpoint}', flush=True)
    step_completed = resumed['step']
    tokens_seen = resumed['tokens_seen']
    last_loss = resumed.get('train_loss')
    if final_step <= step_completed:
        print(f'No training left: checkpoint step {step_completed}, target step {final_step}')
        return
    if args.benchmark_steps and step_completed + args.benchmark_steps + args.benchmark_warmup > final_step:
        raise ValueError('Benchmark would pass the final dataset step')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_log = Path('latest_samples_cuda.log')
    validation_metrics = resumed.get('validation_metrics', {}) if resumed.get('backend') == 'cuda' else {}
    val_step = resumed.get('val_step') if resumed.get('backend') == 'cuda' else None
    best_loss = load_best_loss(checkpoint_dir)
    aim_run = None
    if args.aim and not args.benchmark_steps:
        from aim import Run
        aim_run = Run(repo='.aim', experiment='phobetor_v4_cuda')
        aim_run['model'] = MODEL_CONFIG
        aim_run['corpus_id'] = corpus_id
        aim_run['effective_batch'] = EFFECTIVE_BATCH
    def metadata():
        config = dict(resumed['training_config'])
        config['batch_size'] = args.micro_batch
        config['grad_accum_steps'] = EFFECTIVE_BATCH // args.micro_batch
        return dict(vocab_size=len(tokenizer), model_config=MODEL_CONFIG, training_config=config,
                    sampler_config=SAMPLER_CONFIG.to_dict(), step=step_completed,
                    tokens_seen=tokens_seen, train_loss=last_loss,
                    val_step=val_step, val_loss=validation_metrics.get('val_loss'),
                    validation_metrics=validation_metrics,
                    validation_config={'backend': 'cuda', 'method': 'v4_distribution_numpy_seeded_v1'},
                    precision='bf16_autocast_fp32_weights', effective_batch=EFFECTIVE_BATCH)
    stop_requested = False
    safe_to_save = True
    def request_stop(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print('\nStop requested; finishing current update then saving.', flush=True)
    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    benchmark = args.benchmark_steps > 0
    updates = args.benchmark_steps + args.benchmark_warmup if benchmark else final_step - step_completed
    last_saved_step = step_completed
    interval_start = time.monotonic()
    measured_times = []
    print(f'GPU {torch.cuda.get_device_name()} | V4 8 hybrid blocks x 2 shared passes | '
          f'params {sum(p.numel() for p in model.parameters()):,}', flush=True)
    print(f'Start {step_completed + 1}/{final_step} | micro-batch {args.micro_batch} x '
          f'accumulation {EFFECTIVE_BATCH // args.micro_batch} = effective {EFFECTIVE_BATCH} | '
          f'{"benchmark only" if benchmark else "training"}', flush=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    try:
        for index in range(updates):
            step = step_completed + 1
            tick = time.monotonic()
            loss, gradient, lr, tokens, hidden = train_update(
                model, optimizer, train_data, tokenizer.mask_token_id, step, args.micro_batch, device, schedule_steps)
            safe_to_save = False
            optimizer.step()
            torch.cuda.synchronize()
            safe_to_save = True
            step_completed = step
            tokens_seen += tokens
            last_loss = loss
            seconds = time.monotonic() - tick
            if benchmark and index >= args.benchmark_warmup:
                measured_times.append(seconds)
            if benchmark:
                print(f'Benchmark {index + 1}/{updates} | step {step} | {seconds:.2f} s | '
                      f'{tokens / seconds:.0f} tok/s | GPU peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GiB', flush=True)
                continue
            if step % LOG_INTERVAL == 0 or step == final_step:
                elapsed = time.monotonic() - interval_start
                ram = psutil.virtual_memory().percent
                used_gpu = torch.cuda.max_memory_allocated() / 1024 ** 3
                print(f'Step {step}/{final_step} | Loss {loss:.4f} | LR {lr:.2e} | Grad {gradient:.2f} | '
                      f'{tokens / seconds:.0f} tok/s | {seconds:.2f} s / {elapsed:.2f} s | '
                      f'RAM {ram:.1f}% | VRAM peak {used_gpu:.1f} GiB', flush=True)
                interval_start = time.monotonic()
                if aim_run is not None:
                    for name, value in dict(train_loss=loss, learning_rate=lr, grad_norm=gradient,
                                            tokens_per_sec=tokens / seconds, step_seconds=seconds,
                                            input_tokens_per_step=tokens, hidden_tokens_per_step=hidden,
                                            vram_peak_gib=used_gpu).items():
                        aim_run.track(value, name=name, step=step)
            if stop_requested:
                path = save_checkpoint(checkpoint_dir, model, optimizer, metadata(), kind='emergency')
                print(f'Emergency checkpoint saved: {path}', flush=True)
                last_saved_step = step
                break
            if not args.skip_samples and step % SAMPLE_INTERVAL == 0:
                records = sample_outputs(model, tokenizer, step, sample_log)
                if aim_run is not None:
                    from aim import Text
                    for prompt, record in zip(SAMPLE_PROMPTS, records):
                        aim_run.track(Text(record[1]), name='sample', step=step, context={'prompt': prompt})
            if not args.skip_validation and (step % VAL_INTERVAL == 0 or step == final_step):
                validation_metrics = evaluate(model, val_data, tokenizer.mask_token_id, device)
                val_step = step
                print(table(['VAL step', 'Loss', 'Continue', 'Overlap', 'Refine'],
                            [[step, *[f'{validation_metrics[name]:.4f}' for name in
                                      ('val_loss', 'val_continuation_loss', 'val_overlap_loss', 'val_refinement_loss')]]]), flush=True)
                if aim_run is not None:
                    for name, value in validation_metrics.items():
                        aim_run.track(value, name=name, step=step)
                if validation_metrics['val_loss'] < best_loss:
                    save_best(checkpoint_dir, model, metadata())
                    best_loss = validation_metrics['val_loss']
            if step % SAVE_INTERVAL == 0:
                path = save_checkpoint(checkpoint_dir, model, optimizer, metadata())
                print(f'Checkpoint saved: {path}', flush=True)
                last_saved_step = step
            if stop_requested:
                path = save_checkpoint(checkpoint_dir, model, optimizer, metadata(), kind='emergency')
                print(f'Emergency checkpoint saved: {path}', flush=True)
                last_saved_step = step
                break
        if benchmark:
            mean = float(np.mean(measured_times))
            remaining = max(0, final_step - resumed['step'])
            print(f'Training-only estimate from {len(measured_times)} measured steps: {mean:.2f} s/step | '
                  f'{remaining * mean / 3600:.1f} h | about {remaining * mean / 3600 * 40:.0f} RUB at 40 RUB/h; '
                  f'validation, sampling, saves and setup are additional', flush=True)
            print('Benchmark changed weights in memory only; the source checkpoint was not overwritten.', flush=True)
        elif step_completed > last_saved_step:
            path = save_checkpoint(checkpoint_dir, model, optimizer, metadata(), kind='emergency')
            print(f'Final checkpoint saved: {path}', flush=True)
    except BaseException as exc:
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not benchmark and safe_to_save and step_completed > last_saved_step:
            try:
                path = save_checkpoint(checkpoint_dir, model, optimizer, metadata(), kind='emergency')
                print(f'Emergency checkpoint saved after {type(exc).__name__}: {path}', flush=True)
            except Exception as save_error:
                print(f'Emergency save failed: {save_error}', flush=True)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if aim_run is not None:
            aim_run.close()


if __name__ == '__main__':
    main()
