import math

import torch


STEPS = 32


def alpha_bar(steps):
    return torch.cos(steps.float() * (math.pi / (2 * STEPS))).square().clamp(0, 1)


def corrupt(clean, levels, vocab_size, generator=None):
    if clean.shape != levels.shape:
        raise ValueError('Noise levels must match character positions')
    keep = torch.rand(clean.shape, device=clean.device, generator=generator) < alpha_bar(levels)
    random_chars = torch.randint(vocab_size, clean.shape, device=clean.device, generator=generator)
    return torch.where(keep, clean, random_chars)


def reverse_probabilities(logits, observed, step, vocab_size, temperature=1.0):
    if not 1 <= step <= STEPS:
        raise ValueError('Reverse step is outside the schedule')
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if step == 1:
        return probabilities
    current = alpha_bar(torch.tensor(step, device=logits.device)).item()
    previous = alpha_bar(torch.tensor(step - 1, device=logits.device)).item()
    ratio = current / previous
    base = (1.0 - current) / vocab_size
    denominator = torch.full_like(probabilities, base)
    denominator.scatter_(-1, observed.unsqueeze(-1), current + base)
    weighted = probabilities / denominator.clamp_min(1e-12)
    sum_weighted = weighted.sum(dim=-1, keepdim=True)
    prior = previous * weighted + (1.0 - previous) * sum_weighted / vocab_size
    transition = torch.full_like(probabilities, (1.0 - ratio) / vocab_size)
    transition.scatter_(-1, observed.unsqueeze(-1), ratio + (1.0 - ratio) / vocab_size)
    posterior = prior * transition
    return posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
