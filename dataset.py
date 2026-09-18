import numpy as np
import mlx.core as mx


class TextDataset:
    def __init__(self, bin_path="./data/train_tokens.bin", seq_len=1024):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.total_tokens = len(self.data)

        print(f"loaded {self.total_tokens:,} tokens from bin.")

    def get_batch(self, batch_size=8):
        max_start = self.total_tokens - self.seq_len - 1
        indices = np.random.randint(0, max_start, size=batch_size)

        batch = np.stack([self.data[idx: idx + self.seq_len] for idx in indices])
        return mx.array(batch.astype(np.int32))