import argparse
import json
from pathlib import Path

import torch

from model import PhobetorModel, reference_scan
from training_state import make_optimizer, restore_checkpoint


def verify_scan(device):
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    torch.manual_seed(17)
    u = (torch.randn(1, 32, 2, 64, device=device, dtype=torch.bfloat16) * 0.1).requires_grad_()
    delta = (torch.rand(1, 32, 2, device=device, dtype=torch.float32) * 0.1).requires_grad_()
    b = (torch.randn(1, 32, 1, 64, device=device, dtype=torch.bfloat16) * 0.1).requires_grad_()
    c = (torch.randn(1, 32, 1, 64, device=device, dtype=torch.bfloat16) * 0.1).requires_grad_()
    a = (-torch.rand(2, device=device, dtype=torch.float32)).requires_grad_()
    actual = mamba_chunk_scan_combined(u, delta, a, b, c, chunk_size=32, state_dtype=torch.float32)
    expected = reference_scan(u, delta, b, c, a)
    maximum = (actual.float() - expected.float()).abs().max().item()
    print(f'SSD CUDA/reference max difference: {maximum:.5f}', flush=True)
    if maximum > 0.03:
        raise RuntimeError('CUDA SSD kernel does not match the reference recurrence')
    actual.float().square().mean().backward()
    if not all(value.grad is not None and torch.isfinite(value.grad).all() for value in (u, delta, b, c, a)):
        raise RuntimeError('CUDA SSD backward produced missing or non-finite gradients')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='source_checkpoint')
    parser.add_argument('--parity', default='parity.json')
    parser.add_argument('--data-dir', default='data/v4_books')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA-enabled PyTorch and NVIDIA GPU are required')
    device = torch.device('cuda')
    verify_scan(device)
    metrics = json.loads((Path(args.checkpoint) / 'metrics.json').read_text())
    model = PhobetorModel(metrics['vocab_size'], **metrics['model_config']).to(device)
    optimizer = make_optimizer(model)
    corpus_id = json.loads((Path(args.data_dir) / 'manifest.json').read_text())['corpus_id']
    restored = restore_checkpoint(args.checkpoint, model, optimizer, metrics['model_config'], corpus_id)
    print(f'Model and AdamW loaded: step {restored["step"]}, '
          f'{sum(parameter.numel() for parameter in model.parameters()):,} parameters', flush=True)
    parity_path = Path(args.parity)
    if parity_path.is_file():
        fixture = json.loads(parity_path.read_text())
        if fixture['checkpoint_step'] != restored['step']:
            raise ValueError('Parity fixture was produced for a different checkpoint')
        model.eval()
        tokens = torch.tensor([fixture['input_ids']], device=device, dtype=torch.long)
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(tokens).float()
        positions = fixture['positions']
        expected = torch.tensor(fixture['first_64_logits'], dtype=torch.float32, device=device)
        actual = logits[0, positions, :64]
        maximum = (actual - expected).abs().max().item()
        mean = (actual - expected).abs().mean().item()
        print(f'MLX/CUDA checkpoint logits: mean difference {mean:.5f}, max difference {maximum:.5f}', flush=True)
        if mean > 0.15 or maximum > 0.8:
            raise RuntimeError('Converted model diverges from the MLX parity fixture')
    else:
        raise FileNotFoundError('MLX parity fixture is required')
    print('Verification passed. No training update was run.', flush=True)


if __name__ == '__main__':
    main()
