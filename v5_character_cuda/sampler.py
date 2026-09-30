from dataclasses import asdict, dataclass

import torch

from characters import VOCAB_SIZE, decode, encode
from noise import STEPS, alpha_bar, corrupt, reverse_probabilities


@dataclass(frozen=True)
class SamplerConfig:
    context_window: int = 1024
    block_size: int = 256
    overlap: int = 192
    denoising_steps: int = STEPS
    overlap_start_noise: int = 8
    temperature: float = 1.0

    @property
    def stride(self):
        return self.block_size - self.overlap

    def validate(self):
        if self.block_size != 256 or self.overlap != 192 or self.context_window < self.block_size:
            raise ValueError('This experiment uses 256-character blocks and 192-character overlap')
        if self.denoising_steps != STEPS or not 1 <= self.overlap_start_noise <= STEPS:
            raise ValueError('Invalid categorical noise schedule')
        if self.temperature <= 0:
            raise ValueError('Temperature must be positive')

    def to_dict(self):
        return asdict(self)


class CharacterSampler:
    def __init__(self, model, config=None, seed=1337):
        self.model = model
        self.config = config or SamplerConfig()
        self.config.validate()
        self.seed = seed
        self.device = next(model.parameters()).device

    def _denoise(self, prefix, active, old_count, generator):
        cfg = self.config
        prefix = prefix[-(cfg.context_window - cfg.block_size):]
        prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=self.device)
        if old_count:
            old_level = torch.full((1, old_count), cfg.overlap_start_noise,
                                   dtype=torch.long, device=self.device)
            active[:, :old_count] = corrupt(active[:, :old_count], old_level, VOCAB_SIZE, generator)
        for step in range(STEPS, 0, -1):
            levels = torch.full_like(active, step)
            if old_count:
                levels[:, :old_count] = min(step, cfg.overlap_start_noise)
            model_tokens = torch.cat((prefix_tensor, active), dim=1)
            model_levels = torch.cat((torch.zeros_like(prefix_tensor), levels), dim=1)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == 'cuda'):
                logits = self.model(model_tokens, model_levels, output_start=len(prefix)).float()
            probabilities = reverse_probabilities(logits, active, step, VOCAB_SIZE, cfg.temperature)
            replacement = torch.multinomial(probabilities.reshape(-1, VOCAB_SIZE), 1,
                                            generator=generator).reshape_as(active)
            if old_count and step > cfg.overlap_start_noise:
                active[:, old_count:] = replacement[:, old_count:]
            else:
                active = replacement
        return active[0].tolist()

    @torch.inference_mode()
    def generate_ids(self, prompt_ids, max_new_chars=256):
        if max_new_chars < 0:
            raise ValueError('Requested length must be nonnegative')
        if max_new_chars == 0:
            return []
        cfg = self.config
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        locked = []
        editable = []
        while len(locked) + len(editable) < max_new_chars:
            old_count = len(editable)
            random_new = torch.randint(VOCAB_SIZE, (1, cfg.block_size - old_count),
                                       generator=generator, device=self.device)
            old = torch.tensor([editable], dtype=torch.long, device=self.device)
            active = torch.cat((old, random_new), dim=1)
            refined = self._denoise(list(prompt_ids) + locked, active, old_count, generator)
            if len(locked) + len(refined) >= max_new_chars:
                return (locked + refined)[:max_new_chars]
            locked.extend(refined[:cfg.stride])
            editable = refined[cfg.stride:]
        return (locked + editable)[:max_new_chars]

    def generate(self, prompt, max_new_chars=256):
        output = self.generate_ids(encode(prompt), max_new_chars)
        return prompt + decode(output)
