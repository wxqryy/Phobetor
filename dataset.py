import os
import numpy as np
import mlx.core as mx


class TextDataset:
    def __init__(self, path: str, seq_len: int, dtype=np.uint16, shuffle_seed: int = 1337, repeat: bool = True):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        self.path = path
        self.seq_len = seq_len
        self.shuffle_seed = shuffle_seed
        self.repeat = repeat
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.total_tokens = len(self.data)

        if self.total_tokens < seq_len:
            raise ValueError(f"Dataset {path} is too small for seq_len={seq_len}")

        self.num_sequences = self.total_tokens // self.seq_len
        self.tokens_per_epoch = self.num_sequences * self.seq_len

        self._cached_epoch = None
        self._cached_order = None

        print(
            f"Dataset {os.path.basename(path)}: "
            f"{self.total_tokens:,} tokens | "
            f"{self.num_sequences:,} non-overlapping sequences"
        )

    def _get_order(self, epoch: int):
        if self._cached_epoch != epoch:
            rng = np.random.default_rng(self.shuffle_seed + epoch)
            self._cached_order = rng.permutation(self.num_sequences)
            self._cached_epoch = epoch

        return self._cached_order

    def get_batch(self, batch_size: int, batch_index: int = 0):
        if batch_size <= 0 or batch_index < 0:
            raise ValueError('Expected positive batch size and non-negative batch index')
        first_sequence = batch_index * batch_size
        if not self.repeat and first_sequence + batch_size > self.num_sequences:
            raise StopIteration('Single pass completed; training examples will not be repeated')

        sequence_ids = []
        cursor = first_sequence

        while len(sequence_ids) < batch_size:
            epoch = cursor // self.num_sequences
            offset = cursor % self.num_sequences

            order = self._get_order(epoch)

            take = min(
                batch_size - len(sequence_ids),
                self.num_sequences - offset
            )

            sequence_ids.extend(order[offset:offset + take])
            cursor += take

        batch = np.empty((batch_size, self.seq_len), dtype=np.int32)

        for i, sequence_id in enumerate(sequence_ids):
            start = int(sequence_id) * self.seq_len
            batch[i] = self.data[start:start + self.seq_len]

        return mx.array(batch)
