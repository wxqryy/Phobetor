import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def checkpoint_step(path):
    path = Path(path)
    if path.name in {'emergency_checkpoint', 'norm_f.weight'}:
        try:
            return int(json.loads((path / 'metrics.json').read_text())['step'])
        except (OSError, ValueError, KeyError, TypeError):
            return -1
    match = re.fullmatch(r'ckpt_step_(\d+)', path.name)
    return int(match.group(1)) if match else -1


def checkpoint_paths(directory):
    root = Path(directory)
    paths = [p for p in root.glob('ckpt_step_*') if p.is_dir() and checkpoint_step(p) >= 0]
    emergency = root / 'emergency_checkpoint'
    previous = root / '.emergency_checkpoint.previous'
    if not emergency.exists() and previous.exists():
        os.replace(previous, emergency)
    if emergency.is_dir():
        paths.append(emergency)
    legacy = root / 'norm_f.weight'
    legacy_previous = root / '.norm_f.weight.previous'
    if not legacy.exists() and legacy_previous.exists():
        os.replace(legacy_previous, legacy)
    if legacy.is_dir() and checkpoint_step(legacy) >= 0:
        paths.append(legacy)
    return sorted(paths, key=lambda p: (checkpoint_step(p), p.stat().st_mtime_ns), reverse=True)


def _source_group(name, parameter):
    exempt = parameter.ndim < 2
    return 0 if exempt else 1


def make_optimizer(model, learning_rate=2.5e-4, weight_decay=0.1):
    groups = [[], []]
    for name, parameter in model.named_parameters():
        groups[_source_group(name, parameter)].append(parameter)
    return torch.optim.AdamW([
        {'params': groups[0], 'weight_decay': 0.0},
        {'params': groups[1], 'weight_decay': weight_decay},
    ], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=next(model.parameters()).is_cuda)


def _tensor_from_file(handle, key, expected_shape, device):
    value = handle.get_tensor(key)
    if tuple(value.shape) != tuple(expected_shape):
        raise ValueError(f'{key}: shape {tuple(value.shape)} != {tuple(expected_shape)}')
    if not torch.isfinite(value).all():
        raise ValueError(f'{key}: non-finite tensor')
    return value.to(device=device)


def restore_checkpoint(path, model, optimizer, expected_model_config, expected_corpus_id):
    path = Path(path)
    metrics = json.loads((path / 'metrics.json').read_text())
    step = metrics.get('step')
    if not isinstance(step, int) or step < 0:
        raise ValueError('Invalid checkpoint step')
    if metrics.get('model_config') != expected_model_config:
        raise ValueError('Model architecture differs from checkpoint')
    if metrics.get('vocab_size') != model.vocab_size:
        raise ValueError('Vocabulary differs from checkpoint')
    if metrics.get('training_config', {}).get('corpus_id') != expected_corpus_id:
        raise ValueError('Corpus differs from checkpoint')
    if metrics.get('training_config', {}).get('data_policy') != 'single_pass_no_replacement':
        raise ValueError('Unsupported source data policy')
    config = metrics['training_config']
    if config['batch_size'] * config['grad_accum_steps'] != 16 or config['seq_len'] != 1024:
        raise ValueError('Effective batch or sequence length differs')
    expected = dict(objective='uniform_character_diffusion_x0', block_size=256, overlap=192,
                    denoising_steps=32, overlap_start_noise=8,
                    loss='weighted_full_block_x0_ce', peak_lr=2.5e-4, min_lr=2.5e-5,
                    warmup_steps=2000, seed=1337)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Training setting {key} differs from V5 checkpoint')
    if config.get('stages') != 'initial_overlap_half_uniform_t_v1':
        raise ValueError('Unsupported V5 noise schedule')
    if not (path / 'optimizer.safetensors').is_file():
        raise ValueError('Full optimizer checkpoint is required to resume training')
    if metrics.get('backend') != 'cuda':
        raise ValueError('Only CUDA checkpoints can be resumed')
    names = dict(model.named_parameters())
    with safe_open(str(path / 'model.safetensors'), framework='pt', device='cpu') as handle:
        if set(handle.keys()) != set(names):
            missing = set(names) - set(handle.keys())
            extra = set(handle.keys()) - set(names)
            raise ValueError(f'Model tensor names differ: missing={list(missing)[:3]} extra={list(extra)[:3]}')
        with torch.no_grad():
            for name, parameter in names.items():
                value = _tensor_from_file(handle, name, parameter.shape, parameter.device)
                parameter.copy_(value)
    with safe_open(str(path / 'optimizer.safetensors'), framework='pt', device='cpu') as handle:
        for group_index in (0, 1):
            saved_step = int(handle.get_tensor(f'states.{group_index}.step').item())
            if saved_step != step:
                raise ValueError(f'Optimizer group {group_index} is at step {saved_step}, metadata says {step}')
        for name, parameter in names.items():
            group = _source_group(name, parameter)
            stem = f'states.{group}.{name}'
            state = optimizer.state[parameter]
            state['step'] = torch.tensor(float(step), dtype=torch.float32, device=parameter.device)
            state['exp_avg'] = _tensor_from_file(handle, stem + '.m', parameter.shape, parameter.device)
            state['exp_avg_sq'] = _tensor_from_file(handle, stem + '.v', parameter.shape, parameter.device)
    return metrics


def _safe_snapshot(root, name, model, optimizer, metrics):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.tmp_', dir=root))
    try:
        model_tensors = {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters()}
        save_file(model_tensors, temporary / 'model.safetensors')
        del model_tensors
        if optimizer is not None:
            optimizer_tensors = {}
            for parameter_name, parameter in model.named_parameters():
                group = _source_group(parameter_name, parameter)
                state = optimizer.state[parameter]
                if int(state['step'].item()) != metrics['step']:
                    raise ValueError(f'Optimizer step differs for {parameter_name}')
                stem = f'states.{group}.{parameter_name}'
                optimizer_tensors[stem + '.m'] = state['exp_avg'].detach().cpu().contiguous()
                optimizer_tensors[stem + '.v'] = state['exp_avg_sq'].detach().cpu().contiguous()
            for group in (0, 1):
                optimizer_tensors[f'states.{group}.step'] = torch.tensor(metrics['step'], dtype=torch.int64)
            save_file(optimizer_tensors, temporary / 'optimizer.safetensors')
            del optimizer_tensors
        (temporary / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
        for item in temporary.iterdir():
            with item.open('r+b') as stream:
                os.fsync(stream.fileno())
        destination = root / name
        previous = root / f'.{name}.previous'
        if previous.exists():
            shutil.rmtree(previous)
        if destination.exists():
            os.replace(destination, previous)
        try:
            os.replace(temporary, destination)
        except BaseException:
            if previous.exists() and not destination.exists():
                os.replace(previous, destination)
            raise
        if previous.exists():
            shutil.rmtree(previous)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def save_checkpoint(directory, model, optimizer, metrics, kind='periodic', keep=3):
    if kind not in {'periodic', 'emergency'}:
        raise ValueError('Invalid checkpoint kind')
    name = 'emergency_checkpoint' if kind == 'emergency' else f"ckpt_step_{metrics['step']:07d}"
    path = _safe_snapshot(directory, name, model, optimizer, {**metrics, 'backend': 'cuda', 'checkpoint_kind': kind})
    if kind == 'periodic':
        periodic = [p for p in checkpoint_paths(directory) if p.name.startswith('ckpt_step_')]
        for old in periodic[keep:]:
            shutil.rmtree(old)
    return path


def save_best(directory, model, metrics):
    return _safe_snapshot(directory, 'best_model', model, None, {**metrics, 'backend': 'cuda', 'checkpoint_kind': 'best'})


def load_best_loss(directory, metric='val_loss'):
    path = Path(directory) / 'best_model' / 'metrics.json'
    if not path.is_file():
        return math.inf
    metrics = json.loads(path.read_text())
    return float(metrics.get('validation_metrics', {}).get(metric, metrics.get(metric, math.inf)))
