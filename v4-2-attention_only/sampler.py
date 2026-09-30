import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class SamplerConfig:
    context_window: int = 1024
    block_size: int = 16
    overlap: int = 12
    refinement_steps: int = 12
    temperature: float = 0.8
    audit_every: int = 1
    audit_start_fraction: float = 0.0
    audit_end_fraction: float = 1.0
    audit_candidates: int = 16
    audit_remask_k: int = 1
    audit_current_prob_threshold: float = 0.20
    audit_replacement_conf: float = 0.30
    audit_position_budget: int = 2
    final_polish_rounds: int = 0

    def validate(self):
        if not 0 < self.block_size <= self.context_window:
            raise ValueError('Invalid block or context size')
        if not 0 <= self.overlap < self.block_size:
            raise ValueError('Invalid overlap')
        if self.refinement_steps < 2 or self.temperature < 0:
            raise ValueError('Invalid refinement settings')
        if self.audit_every < 0 or self.audit_candidates < 1 or self.audit_remask_k < 1:
            raise ValueError('Invalid audit settings')
        if not 0 <= self.audit_start_fraction < self.audit_end_fraction <= 1:
            raise ValueError('Invalid audit interval')
        if not 0 <= self.audit_current_prob_threshold <= 1 or not 0 <= self.audit_replacement_conf <= 1:
            raise ValueError('Invalid audit thresholds')
        if self.audit_position_budget < 1 or self.final_polish_rounds < 0:
            raise ValueError('Invalid audit budget')

    @property
    def stride(self):
        return self.block_size - self.overlap

    def to_dict(self):
        return asdict(self)


class PhobetorSampler:
    def __init__(self, model, tokenizer, config=None, seed=1337):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SamplerConfig()
        self.config.validate()
        self.seed = int(seed)
        self.device = next(model.parameters()).device
        self.mask_id = tokenizer.mask_token_id
        self.eos_id = tokenizer.eos_token_id
        if self.mask_id is None:
            raise ValueError('Tokenizer has no mask token')
        forbidden = {tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.bos_token_id, self.mask_id}
        self.forbidden_ids = tuple(i for i in forbidden if i is not None and i != self.eos_id)

    def _active_logits(self, locked_prefix, active, generated_offset, min_new_tokens):
        prefix = list(locked_prefix)[-(self.config.context_window - self.config.block_size):]
        tokens = torch.tensor([prefix + active], dtype=torch.long, device=self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            logits = self.model(tokens, output_start=len(prefix)).float()
        if self.forbidden_ids:
            logits[:, :, self.forbidden_ids] = -torch.inf
        if self.eos_id is not None:
            for pos in range(self.config.block_size):
                if generated_offset + pos < min_new_tokens:
                    logits[:, pos, self.eos_id] = -torch.inf
        return logits

    def _sample(self, logits, generator):
        if self.config.temperature == 0:
            return logits.argmax(dim=-1)
        uniform = torch.rand(logits.shape, device=logits.device, generator=generator).clamp_(1e-6, 1 - 1e-6)
        return (logits / self.config.temperature - torch.log(-torch.log(uniform))).argmax(dim=-1)

    @staticmethod
    def _logp(logits, token_ids):
        return logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)

    @staticmethod
    def _next_mask_count(initial, step, total):
        return max(0, int(math.floor(initial * math.cos(0.5 * math.pi * (step + 1) / total))))

    def _should_audit(self, step):
        cfg = self.config
        progress = step / cfg.refinement_steps
        return cfg.audit_every > 0 and step % cfg.audit_every == 0 and cfg.audit_start_fraction <= progress < cfg.audit_end_fraction

    def _audit(self, prefix, active, confidence, audit_counts, audit_visits, generated_offset, min_new_tokens):
        cfg = self.config
        count = min(cfg.audit_candidates, cfg.block_size)
        groups = max(1, math.ceil(cfg.block_size / count))
        candidates = sorted(range(cfg.block_size), key=lambda i: (audit_visits[i], i % groups, i))[:count]
        current_ids = active[0].tolist()
        previous_conf = confidence[0].tolist()
        current_lp = confidence.clone()
        replacement = active.clone()
        replacement_lp = confidence.clone()
        max_group = max(1, (cfg.block_size + 1) // 2)
        probes = math.ceil(len(candidates) / max_group)
        for offset in range(probes):
            positions = candidates[offset::probes]
            probe = current_ids.copy()
            for pos in positions:
                probe[pos] = self.mask_id
            logits = self._active_logits(prefix, probe, generated_offset, min_new_tokens)
            proposal = logits.argmax(dim=-1)
            current_values = self._logp(logits, active)
            replacement_values = self._logp(logits, proposal)
            current_lp[:, positions] = current_values[:, positions]
            replacement[:, positions] = proposal[:, positions]
            replacement_lp[:, positions] = replacement_values[:, positions]
        for pos in candidates:
            audit_visits[pos] += 1
        confidence[:, candidates] = current_lp[:, candidates]
        replacement_ids = replacement[0].tolist()
        current_values = current_lp[0].tolist()
        replacement_values = replacement_lp[0].tolist()
        eligible = []
        low_threshold = math.log(max(cfg.audit_current_prob_threshold, 1e-12))
        replace_threshold = math.log(max(cfg.audit_replacement_conf, 1e-12))
        for pos in candidates:
            if audit_counts[pos] >= cfg.audit_position_budget or replacement_ids[pos] == current_ids[pos]:
                continue
            better = replacement_values[pos] > current_values[pos] + 1e-6
            if better and (current_values[pos] < low_threshold or replacement_values[pos] >= replace_threshold):
                drop = max(0.0, previous_conf[pos] - current_values[pos]) if math.isfinite(previous_conf[pos]) else 0.0
                eligible.append((pos, -current_values[pos] + drop, replacement_values[pos] - current_values[pos]))
        eligible.sort(key=lambda item: (item[1], item[2]), reverse=True)
        return confidence, [item[0] for item in eligible[:cfg.audit_remask_k]]

    def _remask(self, confidence, target, forced):
        chosen = list(dict.fromkeys(forced))
        target = min(self.config.block_size, max(target, len(chosen)))
        scores = confidence[0].tolist()
        remaining = sorted((i for i in range(self.config.block_size) if i not in chosen), key=lambda i: scores[i])
        chosen.extend(remaining[:target - len(chosen)])
        return chosen

    def _fill(self, prefix, active, confidence, generator, generated_offset, min_new_tokens):
        logits = self._active_logits(prefix, active[0].tolist(), generated_offset, min_new_tokens)
        sampled = self._sample(logits, generator)
        sampled_logp = self._logp(logits, sampled)
        masked = active == self.mask_id
        active = torch.where(masked, sampled, active)
        confidence = torch.where(masked, sampled_logp, confidence)
        return active, confidence

    def _refine_block(self, prefix, initial, initial_conf, block_index, generated_offset, min_new_tokens):
        cfg = self.config
        active = torch.tensor([initial], dtype=torch.long, device=self.device)
        confidence = torch.tensor([initial_conf], dtype=torch.float32, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(self.seed + block_index * 100_003)
        audit_counts = [0] * cfg.block_size
        audit_visits = [0] * cfg.block_size
        for step in range(cfg.refinement_steps):
            active, confidence = self._fill(prefix, active, confidence, generator, generated_offset, min_new_tokens)
            forced = []
            if step < cfg.refinement_steps - 1 and self._should_audit(step):
                confidence, forced = self._audit(prefix, active, confidence, audit_counts, audit_visits, generated_offset, min_new_tokens)
                for pos in forced:
                    audit_counts[pos] += 1
            if step == cfg.refinement_steps - 1:
                break
            target = self._next_mask_count(cfg.block_size, step, cfg.refinement_steps)
            selected = self._remask(confidence, target, forced)
            if selected:
                active[:, selected] = self.mask_id
                confidence[:, selected] = -torch.inf
        for _ in range(cfg.final_polish_rounds):
            confidence, forced = self._audit(prefix, active, confidence, audit_counts, audit_visits, generated_offset, min_new_tokens)
            for pos in forced:
                audit_counts[pos] += 1
            if forced:
                active[:, forced] = self.mask_id
                confidence[:, forced] = -torch.inf
                active, confidence = self._fill(prefix, active, confidence, generator, generated_offset, min_new_tokens)
        return active[0].tolist(), confidence[0].tolist()

    @torch.inference_mode()
    def generate_ids(self, prompt_ids, max_new_tokens=128, min_new_tokens=1):
        if max_new_tokens <= 0:
            return []
        if not 0 <= min_new_tokens <= max_new_tokens:
            raise ValueError('Invalid minimum generated length')
        cfg = self.config
        locked = []
        editable = []
        editable_conf = []
        block_index = 0
        while len(locked) + len(editable) < max_new_tokens:
            active = editable + [self.mask_id] * (cfg.block_size - len(editable))
            confidence = editable_conf + [-math.inf] * (cfg.block_size - len(editable))
            refined, refined_conf = self._refine_block(list(prompt_ids) + locked, active, confidence,
                                                       block_index, len(locked), min_new_tokens)
            visible = refined[:max_new_tokens - len(locked)]
            if self.eos_id is not None:
                for i, token_id in enumerate(visible):
                    if token_id == self.eos_id and len(locked) + i >= min_new_tokens:
                        return locked + visible[:i]
            if len(locked) + len(refined) >= max_new_tokens:
                return (locked + refined)[:max_new_tokens]
            locked.extend(refined[:cfg.stride])
            editable = refined[cfg.stride:]
            editable_conf = refined_conf[cfg.stride:]
            block_index += 1
        return (locked + editable)[:max_new_tokens]

    def generate(self, prompt, max_new_tokens=128, min_new_tokens=1):
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        output = self.generate_ids(prompt_ids, max_new_tokens, min_new_tokens)
        return prompt + self.tokenizer.decode(output, skip_special_tokens=True)
