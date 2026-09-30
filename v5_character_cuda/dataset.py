from pathlib import Path

import numpy as np
import torch


class CharacterDataset:
    def __init__(self, path, seq_len=1024, seed=1337, repeat=False):
        self.path = Path(path)
        self.data = np.memmap(self.path, dtype=np.uint8, mode='r')
        self.seq_len = seq_len
        self.seed = seed
        self.repeat = repeat
        self.num_sequences = len(self.data) // seq_len
        if not self.num_sequences:
            raise ValueError('Character corpus is shorter than one sequence')
        self._epoch = None
        self._order_cache = None

    def _order(self, epoch):
        if self._epoch != epoch:
            self._order_cache = np.random.default_rng(self.seed + epoch).permutation(self.num_sequences)
            self._epoch = epoch
        return self._order_cache

    def get_rows(self, first, count, device):
        if first < 0 or count <= 0:
            raise ValueError('Invalid batch range')
        if not self.repeat and first + count > self.num_sequences:
            raise StopIteration('Single corpus pass completed')
        rows = np.empty((count, self.seq_len), dtype=np.int64)
        for index in range(count):
            epoch, offset = divmod(first + index, self.num_sequences)
            sequence = int(self._order(epoch)[offset])
            start = sequence * self.seq_len
            rows[index] = self.data[start:start + self.seq_len]
        return torch.from_numpy(rows).to(device=device, non_blocking=True)
