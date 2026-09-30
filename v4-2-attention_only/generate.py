import argparse
import json
from pathlib import Path

from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer

from model import PhobetorModel
from sampler import PhobetorSampler, SamplerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt')
    parser.add_argument('--checkpoint', default=str(Path(__file__).resolve().parent / 'checkpoints_cuda' / 'best_model'))
    parser.add_argument('--tokenizer', default='tokenizer')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    path = Path(args.checkpoint)
    metrics = json.loads((path / 'metrics.json').read_text())
    model = PhobetorModel(metrics['vocab_size'], **metrics['model_config'])
    model.load_state_dict(load_file(path / 'model.safetensors'))
    model.to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    config = SamplerConfig(**metrics['sampler_config'])
    with torch.inference_mode():
        print(PhobetorSampler(model, tokenizer, config, args.seed).generate(
            args.prompt, args.max_new_tokens))


if __name__ == '__main__':
    main()
