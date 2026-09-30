import mlx.core as mx


def prepare_self_draft_block(model, sampler, clean, key, mask_id, prefix_length,
                             remask_counts, prepare_training_block):
    batch = clean.shape[0]
    block = sampler.config.block_size
    if len(remask_counts) != batch or any(not 1 <= n <= block for n in remask_counts):
        raise ValueError('One remask count in [1, block_size] is required per row')
    crop_key, sample_key = mx.random.split(key)
    all_mask, targets, _ = prepare_training_block(
        clean, crop_key, mask_id, prefix_length, [block] * batch, overlap=False)
    previous_mode = model.training
    model.eval()
    try:
        logits = model(all_mask, output_start=prefix_length)
        logits = sampler._suppress_forbidden(logits)
        draft = sampler._sample_tokens(logits, sample_key).astype(targets.dtype)
        confidence = sampler._gather_token_log_probs(logits, draft)
        mx.eval(draft, confidence)
    finally:
        model.train(previous_mode)
    rank = mx.argsort(mx.argsort(confidence, axis=-1), axis=-1)
    counts = mx.array(remask_counts, dtype=mx.int32)[:, None]
    remask = rank < counts
    active = mx.where(remask, mask_id, draft)
    noisy = mx.concatenate([all_mask[:, :prefix_length], active], axis=1)
    weights = remask.astype(mx.float32) * block / counts
    return noisy, targets, weights
