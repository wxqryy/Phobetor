from pathlib import Path

import numpy as np
import torch


class TextDataset:
    def __init__(self, path, seq_len=1024, shuffle_seed=1337, repeat=False):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.seq_len = seq_len
        self.shuffle_seed = shuffle_seed
        self.repeat = repeat
        self.data = np.memmap(self.path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)
        self.num_sequences = self.total_tokens // seq_len
        if not self.num_sequences:
            raise ValueError('Corpus is shorter than one sequence')
        self._cached_epoch = None
        self._cached_order = None

    def _order(self, epoch):
        if self._cached_epoch != epoch:
            self._cached_order = np.random.default_rng(self.shuffle_seed + epoch).permutation(self.num_sequences)
            self._cached_epoch = epoch
        return self._cached_order

    def get_rows(self, first, count, device):
        if first < 0 or count < 1:
            raise ValueError('Invalid batch range')
        if not self.repeat and first + count > self.num_sequences:
            raise StopIteration('Single corpus pass completed')
        rows = np.empty((count, self.seq_len), dtype=np.int64)
        for i in range(count):
            cursor = first + i
            epoch, offset = divmod(cursor, self.num_sequences)
            sequence_id = int(self._order(epoch)[offset])
            start = sequence_id * self.seq_len
            rows[i] = self.data[start:start + self.seq_len]
        return torch.from_numpy(rows).to(device=device, non_blocking=True)


def choose_prefix_length(seq_len, sample_id, block_size=16, seed=1337):
    maximum = seq_len - block_size
    short = [n for n in (0, 4, 8, 16, 32, 64, 128) if n <= maximum]
    long = [n for n in (256, 512, 768, maximum) if n <= maximum]
    rng = np.random.default_rng(np.random.SeedSequence([seed, sample_id]))
    options = short if not long or rng.random() < 0.5 else long
    return int(rng.choice(options))


def balanced_mask_counts(sample_ids, block_size=16, seed=1337):
    if block_size < 2 or block_size % 2:
        raise ValueError('Block size must be even and at least two')
    cycles = {}
    counts = []
    for sample_id in sample_ids:
        cycle, offset = divmod(int(sample_id), block_size)
        if cycle not in cycles:
            rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 4001]))
            low = rng.permutation(np.arange(1, block_size // 2 + 1))
            high = rng.permutation(np.arange(block_size // 2 + 1, block_size + 1))
            cycles[cycle] = [int(n) for pair in zip(low, high) for n in pair]
        counts.append(cycles[cycle][offset])
    return counts


def prepare_training_block(clean, mask_id, prefix_length, mask_counts, random_seed, block_size=16, overlap=12):
    batch, length = clean.shape
    if not 0 <= prefix_length <= length - block_size:
        raise ValueError('Prefix and block do not fit in the source sequence')
    if len(mask_counts) != batch or any(n < 1 or n > block_size for n in mask_counts):
        raise ValueError('Invalid mask counts')
    rng = np.random.default_rng(random_seed)
    span = prefix_length + block_size
    starts = rng.integers(0, length - span + 1, size=batch)
    indices = torch.as_tensor(starts[:, None] + np.arange(span)[None, :], device=clean.device)
    cropped = clean.gather(1, indices)
    targets = cropped[:, prefix_length:]
    mask_np = np.zeros((batch, block_size), dtype=np.bool_)
    for i, count in enumerate(mask_counts):
        if count == block_size - overlap and rng.random() < 0.5:
            mask_np[i, overlap:] = True
        else:
            mask_np[i, rng.choice(block_size, count, replace=False)] = True
    mask = torch.from_numpy(mask_np).to(clean.device)
    noisy = torch.cat((cropped[:, :prefix_length], targets.masked_fill(mask, mask_id)), dim=1)
    counts = torch.as_tensor(mask_counts, dtype=torch.float32, device=clean.device)[:, None]
    weights = mask.float() * block_size / counts
    return noisy, targets, weights
