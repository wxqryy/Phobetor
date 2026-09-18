import os
import time
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

DATA_BIN_PATH = "./data/train_tokens.bin"
TOKENIZER_DIR = "./tokenizer"
STATE_FILE = "./data/prepare_state.json"
TARGET_TOKENS = 2_000_000_000
BUFFER_FLUSH_SIZE = 2_000_000

if not os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer.json")):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    tokenizer.save_pretrained(TOKENIZER_DIR)
    print(f"Tokenizer has been saved!")
else:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

tokens_written = 0
skip_docs = 0
if os.path.exists(DATA_BIN_PATH):
    file_bytes = os.path.getsize(DATA_BIN_PATH)
    tokens_written = file_bytes // 4
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            skip_docs = json.load(f).get("processed_docs", 0)
    print(f"Prepared {tokens_written / 1e6:.2f}M tokens. Proceeding...")

if tokens_written >= TARGET_TOKENS:
    print(f"Complete dataset ({tokens_written:,} tokens).")
    exit(0)

print("Start loading dataset")
dataset = load_dataset("Locutusque/UltraTextbooks", split="train", streaming=True)

buffer = []
docs_count = 0

def save_buffer_to_disk(buf, count):
    if not buf:
        return 0
    arr = np.array(buf, dtype=np.uint32)
    with open(DATA_BIN_PATH, "ab") as f:
        f.write(arr.tobytes())
    with open(STATE_FILE, "w") as f:
        json.dump({"processed_docs": count}, f)
    return len(buf)

while tokens_written < TARGET_TOKENS:
    try:
        for row in dataset:
            docs_count += 1
            if docs_count <= skip_docs:
                continue

            text = row.get("text", "")
            if not text or len(text.strip()) < 50:
                continue

            tokens = tokenizer.encode(text)
            tokens.append(tokenizer.eos_token_id)
            buffer.extend(tokens)

            if len(buffer) >= BUFFER_FLUSH_SIZE:
                written = save_buffer_to_disk(buffer, docs_count)
                tokens_written += written
                buffer.clear()
                print(f"Done: {tokens_written / 1e6:.2f}M / {TARGET_TOKENS / 1e6:.0f}M tokens. "
                      f"({(tokens_written / TARGET_TOKENS) * 100:.1f}%) | Files: {docs_count:,}")

            if tokens_written >= TARGET_TOKENS:
                break

    except Exception as e:
        print(f"\n⚠️ Internet issue: {e}")
        written = save_buffer_to_disk(buffer, docs_count)
        tokens_written += written
        buffer.clear()
        time.sleep(5)
        dataset = load_dataset("Locutusque/UltraTextbooks", split="train", streaming=True)
        skip_docs = docs_count

if buffer:
    tokens_written += save_buffer_to_disk(buffer, docs_count)

print(f"\nSuccess! {tokens_written:,} tokens saved to {DATA_BIN_PATH}.")