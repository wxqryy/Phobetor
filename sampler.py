import math
from dataclasses import asdict, dataclass

import mlx.core as mx


@dataclass(frozen=True)
class SamplerConfig:
    context_window: int = 1024
    block_size: int = 32
    overlap: int = 24
    refinement_steps: int = 16
    temperature: float = 0.8

    audit_every: int = 1
    audit_start_fraction: float = 0.0
    audit_end_fraction: float = 1.0
    audit_candidates: int = 32
    audit_remask_k: int = 1
    audit_current_prob_threshold: float = 0.20
    audit_replacement_conf: float = 0.30
    audit_position_budget: int = 2

    final_polish_rounds: int = 0

    def validate(self):
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")
        if self.block_size <= 0 or self.block_size > self.context_window:
            raise ValueError("block_size must satisfy 0 < block_size <= context_window")
        if not 0 <= self.overlap < self.block_size:
            raise ValueError("overlap must satisfy 0 <= overlap < block_size")
        if self.refinement_steps < 2:
            raise ValueError("refinement_steps must be >= 2")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.audit_every < 0:
            raise ValueError("audit_every must be >= 0")
        if not 0 <= self.audit_start_fraction < self.audit_end_fraction <= 1:
            raise ValueError("audit window must satisfy 0 <= start < end <= 1")
        if self.audit_candidates < 1:
            raise ValueError("audit_candidates must be >= 1")
        if self.audit_remask_k < 1:
            raise ValueError("audit_remask_k must be >= 1")
        if not 0 <= self.audit_current_prob_threshold <= 1:
            raise ValueError("audit_current_prob_threshold must be in [0, 1]")
        if not 0 <= self.audit_replacement_conf <= 1:
            raise ValueError("audit_replacement_conf must be in [0, 1]")
        if self.audit_position_budget < 1:
            raise ValueError("audit_position_budget must be >= 1")
        if self.final_polish_rounds < 0:
            raise ValueError("final_polish_rounds must be >= 0")

    @property
    def stride(self):
        return self.block_size - self.overlap

    def to_dict(self):
        return asdict(self)


class PhobetorSampler:
    """
    Long-form masked-diffusion sampler.

    State:
        LOCKED PREFIX | EDITABLE OVERLAP | NEW MASKS

    Inside each active block:
      * predict all masked positions in parallel;
      * progressively keep more tokens;
      * already-filled tokens can be Token-to-Mask remasked;
      * masked context audits refresh old-token confidence before remasking;
      * the last overlap tokens remain editable in the next block.
    """

    def __init__(self, model, tokenizer, config=None, seed=1337):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SamplerConfig()
        self.config.validate()
        self.seed = int(seed)
        self.mask_id = tokenizer.mask_token_id
        self.eos_id = tokenizer.eos_token_id

        if self.mask_id is None:
            raise RuntimeError("Tokenizer does not define mask_token_id")

        forbidden = {
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
            tokenizer.bos_token_id,
            tokenizer.mask_token_id,
        }
        self.forbidden_ids = tuple(
            token_id for token_id in forbidden
            if token_id is not None and token_id != self.eos_id
        )

    def _suppress_forbidden(self, logits):
        if not self.forbidden_ids:
            return logits
        vocab_ids = mx.arange(logits.shape[-1])
        for token_id in self.forbidden_ids:
            logits = mx.where(
                vocab_ids == token_id,
                mx.array(float("-inf"), dtype=logits.dtype),
                logits,
            )
        return logits

    def _sample_tokens(self, logits, key):
        if self.config.temperature <= 0:
            return mx.argmax(logits, axis=-1)

        scaled = logits / self.config.temperature
        u = mx.random.uniform(shape=scaled.shape, key=key)
        u = mx.clip(u, 1e-6, 1.0 - 1e-6)
        gumbel = -mx.log(-mx.log(u))
        return mx.argmax(scaled + gumbel, axis=-1)

    @staticmethod
    def _gather_logits(logits, token_ids):
        return mx.take_along_axis(logits, token_ids[..., None], axis=-1)[..., 0]

    @classmethod
    def _gather_token_log_probs(cls, logits, token_ids):
        chosen = cls._gather_logits(logits, token_ids)
        return chosen - mx.logsumexp(logits, axis=-1)

    @staticmethod
    def _next_mask_count(initial_mask_count, step, total_steps):
        progress = (step + 1) / total_steps
        ratio = math.cos(0.5 * math.pi * progress)
        return max(0, int(math.floor(initial_mask_count * ratio)))

    def _build_model_input(self, locked_prefix, active_tokens):
        cfg = self.config
        if len(active_tokens) != cfg.block_size:
            raise ValueError(f"active block must contain exactly {cfg.block_size} tokens")

        prefix = list(locked_prefix)
        max_prefix = cfg.context_window - cfg.block_size
        if len(prefix) > max_prefix:
            prefix = prefix[-max_prefix:] if max_prefix else []

        tokens = prefix + list(active_tokens)
        active_start = len(prefix)
        active_end = active_start + cfg.block_size
        return mx.array([tokens], dtype=mx.int32), active_start, active_end

    def _active_logits(self, locked_prefix, active_tokens, generated_offset=0, min_new_tokens=0):
        x, active_start, active_end = self._build_model_input(locked_prefix, active_tokens)
        logits = self.model(x, output_start=active_start, output_end=active_end)
        logits = self._suppress_forbidden(logits)
        if self.eos_id is not None and generated_offset < min_new_tokens:
            early = generated_offset + mx.arange(self.config.block_size) < min_new_tokens
            eos = mx.arange(logits.shape[-1]) == self.eos_id
            logits = mx.where(early[None, :, None] & eos[None, None, :], -mx.inf, logits)
        return logits

    def _should_audit(self, step):
        cfg = self.config
        if cfg.audit_every <= 0:
            return False
        progress = step / cfg.refinement_steps
        return (
            step % cfg.audit_every == 0
            and cfg.audit_start_fraction <= progress < cfg.audit_end_fraction
        )

    def _context_audit(
        self, locked_prefix, active, confidence, audit_counts, audit_visits,
        trace=False, label="AUDIT", generated_offset=0, min_new_tokens=0,
    ):
        cfg = self.config
        m = min(cfg.audit_candidates, cfg.block_size)
        # A limit on forced replacements must not disable confidence updates.
        available = list(range(cfg.block_size))
        groups = max(1, math.ceil(cfg.block_size / m))
        candidates = sorted(available, key=lambda i: (audit_visits[i], i % groups, i))[:m]
        if not candidates:
            return confidence, []

        current_ids = active[0].tolist()
        current_logp = confidence
        replacement = active
        replacement_logp = confidence
        # Hide the token being scored: visible-position logits can just copy it.
        # When auditing the whole block, use complementary groups rather than
        # removing ALL its words. These are grouped masked estimates, not exact
        # leave-one-out probabilities. Both probes see the same frozen draft.
        max_group = max(1, (cfg.block_size + 1) // 2)
        probe_count = math.ceil(len(candidates) / max_group)
        for offset in range(probe_count):
            positions = candidates[offset::probe_count]
            probe_tokens = current_ids.copy()
            selector = [False] * cfg.block_size
            for pos in positions:
                probe_tokens[pos] = self.mask_id
                selector[pos] = True
            selector = mx.array([selector], dtype=mx.bool_)
            logits = self._active_logits(
                locked_prefix, probe_tokens, generated_offset, min_new_tokens,
            )
            proposed = mx.argmax(logits, axis=-1)
            current_logp = mx.where(selector, self._gather_token_log_probs(logits, active), current_logp)
            replacement = mx.where(selector, proposed, replacement)
            replacement_logp = mx.where(selector, self._gather_token_log_probs(logits, proposed), replacement_logp)
            mx.eval(current_logp, replacement, replacement_logp)
        for pos in candidates:
            audit_visits[pos] += 1

        previous_conf = confidence[0].tolist()

        cand_mask = [False] * cfg.block_size
        for pos in candidates:
            cand_mask[pos] = True
        cand_mask = mx.array([cand_mask], dtype=mx.bool_)
        confidence = mx.where(cand_mask, current_logp, confidence)
        mx.eval(confidence)

        replacement_ids = replacement[0].tolist()
        current_lp = current_logp[0].tolist()
        replacement_lp = replacement_logp[0].tolist()

        eligible = []
        lowprob_threshold = math.log(max(cfg.audit_current_prob_threshold, 1e-12))
        replacement_threshold = math.log(max(cfg.audit_replacement_conf, 1e-12))
        for pos in candidates:
            if audit_counts[pos] >= cfg.audit_position_budget:
                continue
            changed = replacement_ids[pos] != current_ids[pos]
            low_current_prob = current_lp[pos] < lowprob_threshold
            strong_replacement = changed and replacement_lp[pos] >= replacement_threshold
            better_alternative = replacement_lp[pos] > current_lp[pos] + 1e-6
            if changed and better_alternative and (low_current_prob or strong_replacement):
                old_lp = previous_conf[pos]
                confidence_drop = max(0.0, old_lp - current_lp[pos]) if math.isfinite(old_lp) else 0.0
                surprise = -current_lp[pos]
                alt_advantage = replacement_lp[pos] - current_lp[pos]
                eligible.append((pos, surprise + confidence_drop, alt_advantage))

        eligible.sort(key=lambda item: (item[1], item[2]), reverse=True)
        forced = [pos for pos, _, _ in eligible[:cfg.audit_remask_k]]

        if trace:
            details = ", ".join(
                f"{pos}:{self.tokenizer.decode([current_ids[pos]])}->{self.tokenizer.decode([replacement_ids[pos]])}"
                for pos in forced
            ) or "none"
            print(f"  {label}: candidates={len(candidates)} remask={forced} {details}")

        return confidence, forced

    def _make_remask(self, confidence, num_to_mask, forced_positions):
        cfg = self.config
        if num_to_mask <= 0 and not forced_positions:
            return mx.zeros((1, cfg.block_size), dtype=mx.bool_)

        forced = []
        seen = set()
        for pos in forced_positions:
            if 0 <= pos < cfg.block_size and pos not in seen:
                forced.append(pos)
                seen.add(pos)

        target = max(num_to_mask, len(forced))
        target = min(target, cfg.block_size)

        scores = confidence[0].tolist()
        remaining = [i for i in range(cfg.block_size) if i not in seen]
        remaining.sort(key=lambda i: scores[i])
        selected = forced + remaining[:max(0, target - len(forced))]

        mask = [False] * cfg.block_size
        for pos in selected:
            mask[pos] = True
        return mx.array([mask], dtype=mx.bool_)

    def _fill_current_masks(self, locked_prefix, active, confidence, key, generated_offset=0, min_new_tokens=0):
        logits = self._active_logits(
            locked_prefix, active[0].tolist(), generated_offset, min_new_tokens,
        )
        masked = active == self.mask_id
        key, sample_key = mx.random.split(key, num=2)
        sampled = self._sample_tokens(logits, sample_key)
        sampled_logp = self._gather_token_log_probs(logits, sampled)
        active = mx.where(masked, sampled, active)
        confidence = mx.where(masked, sampled_logp, confidence)
        mx.eval(active, confidence)
        return active, confidence, key

    def _refine_block(
        self, locked_prefix, initial_active, initial_confidence, seed_offset,
        trace=False, block_index=0, generated_offset=0, min_new_tokens=0,
    ):
        cfg = self.config
        active = mx.array([initial_active], dtype=mx.int32)
        confidence = mx.array([initial_confidence], dtype=mx.float32)
        initial_mask_count = cfg.block_size

        key = mx.random.key(self.seed + seed_offset)
        audit_counts = [0] * cfg.block_size
        audit_visits = [0] * cfg.block_size

        for step in range(cfg.refinement_steps):
            active, confidence, key = self._fill_current_masks(
                locked_prefix, active, confidence, key, generated_offset, min_new_tokens,
            )

            forced = []
            # The final fill must leave no masks. Do not spend an audit whose
            # replacements would never be regenerated.
            if step < cfg.refinement_steps - 1 and self._should_audit(step):
                confidence, forced = self._context_audit(
                    locked_prefix,
                    active,
                    confidence,
                    audit_counts,
                    audit_visits,
                    trace=trace,
                    label=f"AUDIT S{step + 1:02d}",
                    generated_offset=generated_offset,
                    min_new_tokens=min_new_tokens,
                )
                for pos in forced:
                    audit_counts[pos] += 1

            if step == cfg.refinement_steps - 1:
                if trace:
                    text = self.tokenizer.decode(active[0].tolist(), skip_special_tokens=False)
                    print(f"B{block_index:02d} S{step + 1:02d}/{cfg.refinement_steps} remask=00 | {text}")
                break

            num_to_mask = self._next_mask_count(initial_mask_count, step, cfg.refinement_steps)
            remask = self._make_remask(confidence, num_to_mask, forced)

            if trace:
                text = self.tokenizer.decode(active[0].tolist(), skip_special_tokens=False)
                actual = int(sum(remask[0].tolist()))
                print(f"B{block_index:02d} S{step + 1:02d}/{cfg.refinement_steps} remask={actual:02d} | {text}")

            active = mx.where(remask, mx.array(self.mask_id, dtype=active.dtype), active)
            confidence = mx.where(remask, mx.array(float("-inf"), dtype=confidence.dtype), confidence)
            mx.eval(active, confidence)

        for polish in range(cfg.final_polish_rounds):
            confidence, forced = self._context_audit(
                locked_prefix,
                active,
                confidence,
                audit_counts,
                audit_visits,
                trace=trace,
                label=f"POLISH {polish + 1}",
                generated_offset=generated_offset,
                min_new_tokens=min_new_tokens,
            )
            for pos in forced:
                audit_counts[pos] += 1
            if not forced:
                continue
            remask = self._make_remask(confidence, 0, forced)
            active = mx.where(remask, mx.array(self.mask_id, dtype=active.dtype), active)
            confidence = mx.where(remask, mx.array(float("-inf"), dtype=confidence.dtype), confidence)
            active, confidence, key = self._fill_current_masks(
                locked_prefix, active, confidence, key, generated_offset, min_new_tokens,
            )

        mx.eval(active, confidence)
        return active[0].tolist(), confidence[0].tolist()

    def generate_ids(self, prompt_ids, max_new_tokens=128, min_new_tokens=1, trace=False):
        cfg = self.config
        if max_new_tokens <= 0:
            return []
        if not 0 <= min_new_tokens <= max_new_tokens:
            raise ValueError("min_new_tokens must satisfy 0 <= min_new_tokens <= max_new_tokens")

        locked_generated = []
        editable_tail = []
        editable_conf = []
        block_index = 0

        while len(locked_generated) + len(editable_tail) < max_new_tokens:
            new_slots = cfg.block_size - len(editable_tail)
            active = editable_tail + [self.mask_id] * new_slots
            active_conf = editable_conf + [float("-inf")] * new_slots
            locked_prefix = list(prompt_ids) + locked_generated

            refined, refined_conf = self._refine_block(
                locked_prefix,
                active,
                active_conf,
                seed_offset=block_index * 100_003,
                trace=trace,
                block_index=block_index,
                generated_offset=len(locked_generated),
                min_new_tokens=min_new_tokens,
            )

            candidate = locked_generated + refined
            remaining = max_new_tokens - len(locked_generated)
            visible = refined[:remaining]

            if self.eos_id is not None:
                for i, token_id in enumerate(visible):
                    total_new = len(locked_generated) + i
                    if token_id == self.eos_id and total_new >= min_new_tokens:
                        return (locked_generated + visible[:i])[:max_new_tokens]

            if len(candidate) >= max_new_tokens:
                return candidate[:max_new_tokens]

            locked_generated.extend(refined[:cfg.stride])
            editable_tail = refined[cfg.stride:]
            editable_conf = refined_conf[cfg.stride:]
            block_index += 1

        return (locked_generated + editable_tail)[:max_new_tokens]

    def generate(self, prompt_text, max_new_tokens=128, min_new_tokens=1, trace=False):
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        generated_ids = self.generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            trace=trace,
        )
        continuation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return prompt_text + continuation
