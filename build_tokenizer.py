import os

from datasets import interleave_datasets, load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

TOKENIZER_DIR = "./tokenizer"
VOCAB_SIZE = 32_768
MAX_TRAIN_DOCS = 300_000
SEED = 1337

SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|mask|>"]


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


def text_iterator():
    seen = 0
    for row in make_stream():
        text = row.get("text", "")
        if not text or len(text.strip()) < 100:
            continue
        yield text
        seen += 1
        if seen >= MAX_TRAIN_DOCS:
            break


os.makedirs(TOKENIZER_DIR, exist_ok=True)

tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
tokenizer.normalizer = normalizers.NFC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=MAX_TRAIN_DOCS)

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token="<|pad|>",
    unk_token="<|unk|>",
    bos_token="<|bos|>",
    eos_token="<|eos|>",
    mask_token="<|mask|>",
    model_max_length=2048,
)

hf_tokenizer.save_pretrained(TOKENIZER_DIR)
print(f"Saved tokenizer to {TOKENIZER_DIR}")
print(f"Vocab size: {len(hf_tokenizer):,}")
print(f"MASK id: {hf_tokenizer.mask_token_id}, EOS id: {hf_tokenizer.eos_token_id}")
