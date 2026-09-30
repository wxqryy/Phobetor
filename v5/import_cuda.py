import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open

from training_state import TrainingProgress


def import_cuda_checkpoint(path, model, optimizer, model_config, vocab_size,
                           corpus_id, schedule):
    path = Path(path)
    metrics = json.loads((path / 'metrics.json').read_text())
    config = metrics.get('training_config', {})
    if metrics.get('backend') != 'cuda':
        raise ValueError('Expected a CUDA V4 source checkpoint')
    if metrics.get('model_config') != model_config or metrics.get('vocab_size') != vocab_size:
        raise ValueError('Source architecture or vocabulary differs')
    if config.get('corpus_id') != corpus_id or config.get('data_policy') != 'single_pass_no_replacement':
        raise ValueError('Source corpus or data policy differs')
    if config.get('batch_size', 0) * config.get('grad_accum_steps', 0) != 32:
        raise ValueError('Source effective batch differs')
    step = metrics.get('step')
    if type(step) is not int or step < 1 or step >= config.get('schedule_steps', 0):
        raise ValueError('Invalid source step')
    model_shapes = {name: tuple(value.shape) for name, value in tree_flatten(model.parameters())}
    tensors = {}
    with safe_open(str(path / 'model.safetensors'), framework='numpy') as handle:
        if set(handle.keys()) != set(model_shapes):
            raise ValueError('Source model tensor names differ')
        for name, shape in model_shapes.items():
            value = handle.get_tensor(name)
            if name.endswith('.conv.weight'):
                value = value.transpose(0, 2, 1).copy()
            if value.shape != shape:
                raise ValueError(f'Source tensor {name} shape {value.shape} != {shape}')
            tensors[name] = mx.array(value)
    model.update(tree_unflatten(tensors))
    del tensors
    state = {}
    expected = {f'states.{group}.{name}.{moment}'
                for name, shape in model_shapes.items()
                for group in [0 if len(shape) < 2 or name.rsplit('.', 1)[-1] in {'A_log', 'D', 'dt_bias'} else 1]
                for moment in ('m', 'v')}
    expected |= {'states.0.step', 'states.1.step'}
    with safe_open(str(path / 'optimizer.safetensors'), framework='numpy') as handle:
        if set(handle.keys()) != expected:
            raise ValueError('Source optimizer tensor names differ')
        for group in (0, 1):
            key = f'states.{group}.step'
            if int(handle.get_tensor(key).item()) != step:
                raise ValueError('Source optimizer step differs from checkpoint metadata')
            state[key] = mx.array(step, dtype=mx.uint64)
            state[f'states.{group}.learning_rate'] = mx.array(
                schedule(mx.array(step - 1)).item(), dtype=mx.float32)
        for key in sorted(expected - {'states.0.step', 'states.1.step'}):
            name = key.split('.', 2)[2].rsplit('.', 1)[0]
            value = handle.get_tensor(key)
            if name.endswith('.conv.weight'):
                value = value.transpose(0, 2, 1).copy()
            if value.shape != model_shapes[name]:
                raise ValueError(f'Source optimizer tensor {key} shape differs')
            state[key] = mx.array(value)
    optimizer.state = tree_unflatten(state)
    mx.eval(model.parameters(), optimizer.state)
    return TrainingProgress(step, metrics['tokens_seen'], metrics.get('train_loss')), metrics
