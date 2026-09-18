import json
import os
import time

import numpy as np
from datasets import interleave_datasets, load_dataset
from transformers import AutoTokenizer

TOKENIZER_DIR = "./tokenizer"
TRAIN_PATH = "./data/train_tokens.bin"
VAL_PATH = "./data/val_tokens.bin"
STATE_PATH = "./data/prepare_state.json"

TRAIN_TARGET = 2_000_000_000
VAL_TARGET = 10_000_000
FLUSH_TOKENS = 2_000_000
VAL_EVERY_N_DOCS = 200
SEED = 1337
DTYPE = np.uint16


def make_stream():
    finemath = load_dataset(
        "HuggingFaceTB/finemath",
        "finemath-4plus",
        split="train",
        streaming=True,
    )
    textbooks = load_dataset(
        "Locutusque/UltraTextbooks-2.0",
        split="train",
        streaming=True,
    )
    textbooks = textbooks.filter(
        lambda row: row.get("source") != "nampdn-ai/tiny-strange-textbooks"
    )
    return interleave_datasets(
        [finemath, textbooks],
        probabilities=[0.60, 0.40],
        seed=SEED,
        stopping_strategy="all_exhausted",
    )


def disk_tokens(path):
    if not os.path.exists(path):
        return 0
    return os.path.getsize(path) // np.dtype(DTYPE).itemsize


def append_tokens(path, values):
    if not values:
        return 0
    arr = np.asarray(values, dtype=DTYPE)
    with open(path, "ab") as f:
        f.write(arr.tobytes())
    return len(arr)


def save_state(processed_docs, train_written, val_written):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "processed_docs": processed_docs,
                "train_tokens": train_written,
                "val_tokens": val_written,
            },
            f,
        )


os.makedirs("./data", exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

if len(tokenizer) > np.iinfo(DTYPE).max + 1:
    raise RuntimeError("Tokenizer is too large for uint16 storage")
if tokenizer.eos_token_id is None or tokenizer.mask_token_id is None:
    raise RuntimeError("Tokenizer must define EOS and MASK tokens")

train_written = disk_tokens(TRAIN_PATH)
val_written = disk_tokens(VAL_PATH)
processed_docs = 0

if os.path.exists(STATE_PATH):
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    processed_docs = int(state.get("processed_docs", 0))

print(f"Existing train: {train_written / 1e6:.2f}M tokens")
print(f"Existing val:   {val_written / 1e6:.2f}M tokens")
print(f"Resume after:   {processed_docs:,} streamed rows")

train_buffer = []
val_buffer = []


def flush():
    global train_written, val_written
    train_remaining = max(TRAIN_TARGET - train_written, 0)
    val_remaining = max(VAL_TARGET - val_written, 0)

    if train_buffer and train_remaining:
        chunk = train_buffer[:train_remaining]
        train_written += append_tokens(TRAIN_PATH, chunk)
    if val_buffer and val_remaining:
        chunk = val_buffer[:val_remaining]
        val_written += append_tokens(VAL_PATH, chunk)

    train_buffer.clear()
    val_buffer.clear()
    save_state(processed_docs, train_written, val_written)


while train_written < TRAIN_TARGET or val_written < VAL_TARGET:
    try:
        stream = make_stream()
        if processed_docs:
            stream = stream.skip(processed_docs)

        for row in stream:
            processed_docs += 1
            text = row.get("text", "")
            if not text or len(text.strip()) < 100:
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)
            ids.append(tokenizer.eos_token_id)

            if processed_docs % VAL_EVERY_N_DOCS == 0 and val_written + len(val_buffer) < VAL_TARGET:
                val_buffer.extend(ids)
            elif train_written + len(train_buffer) < TRAIN_TARGET:
                train_buffer.extend(ids)

            targets_buffered = (
                    train_written + len(train_buffer) >= TRAIN_TARGET
                    and val_written + len(val_buffer) >= VAL_TARGET
            )

            if len(train_buffer) + len(val_buffer) >= FLUSH_TOKENS or targets_buffered:
                flush()

                print(
                    f"Train {train_written / 1e9:.3f}B/{TRAIN_TARGET / 1e9:.1f}B | "
                    f"Val {val_written / 1e6:.2f}M/{VAL_TARGET / 1e6:.0f}M | "
                    f"Rows {processed_docs:,}"
                )

            if train_written >= TRAIN_TARGET and val_written >= VAL_TARGET:
                break

        flush()
        if train_written >= TRAIN_TARGET and val_written >= VAL_TARGET:
            break
        raise RuntimeError("Dataset stream ended before token targets were reached")

    except KeyboardInterrupt:
        flush()
        raise
    except Exception as e:
        print(f"Data stream error: {e}")
        flush()
        time.sleep(5)

print(f"Done. Train: {train_written:,} tokens, val: {val_written:,} tokens")