import argparse
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from model import PhobetorModel
from sampler import CharacterSampler, SamplerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt')
    parser.add_argument('--checkpoint', default=str(Path(__file__).resolve().parent / 'checkpoints_v5_cuda' / 'best_model'))
    parser.add_argument('--max-new-chars', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    path = Path(args.checkpoint)
    metrics = json.loads((path / 'metrics.json').read_text())
    model = PhobetorModel(metrics['vocab_size'], **metrics['model_config'])
    model.load_state_dict(load_file(path / 'model.safetensors'))
    model.to('cuda').eval()
    config = SamplerConfig(**metrics['sampler_config'])
    with torch.inference_mode():
        print(CharacterSampler(model, config, args.seed).generate(args.prompt, args.max_new_chars))


if __name__ == '__main__':
    main()
