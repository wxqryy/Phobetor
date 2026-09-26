import argparse
import importlib.util
import io
import json
from pathlib import Path
import tarfile

from safetensors import safe_open


def parity_bytes(root, checkpoint, metrics):
    import mlx.core as mx
    from transformers import AutoTokenizer
    source = importlib.util.spec_from_file_location('phobetor_mlx_source', root / 'model.py')
    module = importlib.util.module_from_spec(source)
    source.loader.exec_module(module)
    tokenizer = AutoTokenizer.from_pretrained(root / 'tokenizer', local_files_only=True)
    model = module.PhobetorModel(metrics['vocab_size'], **metrics['model_config'])
    model.load_weights(str(checkpoint / 'model.safetensors'))
    model.eval()
    ids = tokenizer.encode('One morning, a little girl', add_special_tokens=False)
    ids = (ids + [tokenizer.mask_token_id] * 128)[:128]
    logits = model(mx.array([ids], dtype=mx.int32)).astype(mx.float32)
    mx.eval(logits)
    positions = [0, 63, 127]
    fixture = {'checkpoint_step': metrics['step'], 'input_ids': ids, 'positions': positions,
               'first_64_logits': [logits[0, pos, :64].tolist() for pos in positions]}
    return json.dumps(fixture).encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', default=str(Path.home() / 'Downloads' / 'phobetor_cuda_v4.tar'))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    code_dir = Path(__file__).resolve().parent
    root = code_dir.parent
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).resolve()
    else:
        candidates = []
        for path in (root / 'checkpoints').glob('ckpt_step_*'):
            if path.is_dir():
                candidates.append(path)
        emergency = root / 'checkpoints' / 'emergency_checkpoint'
        if emergency.is_dir():
            candidates.append(emergency)
        candidates = [path for path in candidates if all((path / name).is_file() for name in
                      ('metrics.json', 'model.safetensors', 'optimizer.safetensors'))]
        if not candidates:
            raise FileNotFoundError('No complete MLX checkpoint found')
        checkpoint = max(candidates, key=lambda path: (json.loads((path / 'metrics.json').read_text())['step'],
                                                        path.stat().st_mtime_ns))
    metrics = json.loads((checkpoint / 'metrics.json').read_text())
    with safe_open(str(checkpoint / 'optimizer.safetensors'), framework='numpy') as optimizer:
        steps = [int(optimizer.get_tensor(f'states.{index}.step').item()) for index in (0, 1)]
    if steps != [metrics['step'], metrics['step']]:
        raise ValueError(f'Checkpoint optimizer steps {steps} differ from metadata step {metrics["step"]}')
    required = [code_dir / name for name in ('model.py', 'dataset.py', 'sampler.py', 'training_state.py',
                                             'train.py', 'requirements.txt', 'verify.py', 'pack.py', 'setup_cuda.sh')]
    required.extend(root / 'tokenizer' / name for name in ('tokenizer.json', 'tokenizer_config.json'))
    required.extend(root / 'data' / 'v4_books' / name for name in
                    ('manifest.json', 'train_tokens.bin', 'val_tokens.bin'))
    required.extend(checkpoint / name for name in ('metrics.json', 'model.safetensors', 'optimizer.safetensors'))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing required files: ' + ', '.join(missing))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(output == path.resolve() for path in required):
        raise ValueError('Output would overwrite an input file')
    print(f'Packing MLX step {metrics["step"]} from {checkpoint}', flush=True)
    print(f'Output: {output}', flush=True)
    fixture = parity_bytes(root, checkpoint, metrics)
    if args.check_only:
        print(f'Ready to pack {len(required)} files and parity fixture ({len(fixture)} bytes)', flush=True)
        return
    with tarfile.open(output, 'w') as archive:
        for path in required:
            if path.parent == code_dir:
                relative = path.name
            elif path.parent == root / 'tokenizer':
                relative = 'tokenizer/' + path.name
            elif path.parent == root / 'data' / 'v4_books':
                relative = 'data/v4_books/' + path.name
            else:
                relative = 'source_checkpoint/' + path.name
            archive.add(path, arcname='cuda_v4/' + relative, recursive=False)
        info = tarfile.TarInfo('cuda_v4/parity.json')
        info.size = len(fixture)
        archive.addfile(info, io.BytesIO(fixture))
    print(f'Ready: {output} | {output.stat().st_size / 1024 ** 3:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
