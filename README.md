# Phobetor

I'm building a small text generation model through experiments with architecture,
training, and sampling. The idea is to generate text in overlapping blocks and
let the model revise them before moving on. I started with MLX on an Apple
Silicon Mac and later ran CUDA experiments on an RTX 3070 Ti.

## How V4 works

In V4, I combine bidirectional Mamba-2, attention, and SwiGLU in eight blocks
and run through them twice with shared weights. The model can look at the whole
available context, but generation happens inside a small active window.

In V4, that window holds 16 tokens. The sampler fills masked positions, checks
its confidence, and masks some predictions again so the model can revise them.
After 12 rounds of refinement, the window moves four tokens to the right:
12 tokens remain editable and four new positions open up. The prompt and
committed text stay fixed.

For training, I take a text prefix and mask part or all of the next block.
Mask counts from 1 to 16 are balanced across training cycles. Each example teaches
one denoising step; the 12-round generation loop runs during inference.

## What changed between versions

### V1

I started with eight hybrid blocks and two passes through the same weights.
Each block combined attention, SwiGLU, and bidirectional exponential filter. I used the
Qwen 2.5-0.5B tokenizer with around 152k tokens and random masking across the context.

After the first few thousand steps, I saw fewer repetitions and some recognizable
phrases. The text was still rough, but there was enough progress to keep going.
Once I found the mistake in the recurrent layer, I decided to implement
selective Mamba.

### V2

I replaced the filter with selective bidirectional Mamba-2 and switched to a
custom 32,768-token tokenizer, leaving more of the parameter budget for the
rest of the model. But the architecture also drifted away from what I intended:
it became a single pass through nine Mamba layers and three attention layers.
The second pass was gone, and there was much less attention.

Reconstruction loss kept falling, but generated text got stuck in repetitions.
I tried sampler changes and added continuation and overlap tasks to training.
That did not give me a consistent fix, so I went back to the original block layout.

### V3

I restored eight blocks with Mamba-2, attention, and SwiGLU in each, plus two
passes with shared weights. Training used a prefix and a masked continuation
block. Generation used a 32-token window, a 24-token overlap, and 16 refinements.

At around 7,000 steps on UltraTextbooks, I was still getting incoherent text,
numbers, and markup. I decided to try books and a smaller generation window.
I don't know how much of the problem came from the data versus the training setup.

### V4

I kept the V3 architecture and tokenizer. I changed the corpus to English books,
reduced the window to 16 tokens with a 12-token overlap, and set refinement to
12 rounds. I also balanced the mask counts and increased the effective batch
from eight to 32 examples.

In the early run, validation loss dropped from about 6.99 at step 500 to 6.59 at
step 1,000. The samples were still incoherent, so I haven't reached the result
I'm after yet. My goal is to get this model generating coherent English and
understand which choices actually help it learn.

I changed too many things between versions to treat their losses or step counts
as a fair comparison.

### V4.2

I wanted to check whether Mamba was causing the generation problems, so I made
an attention-only CUDA version. It uses 16 attention and SwiGLU layers in one
pass. I kept the 32,768-token tokenizer, the book corpus, and the 16-token
generation window with a 12-token overlap.

I trained it on my RTX 3070 Ti to step 55,869. The text looked more like English,
but the model still did not reliably continue a prompt or answer simple questions.
The best fully masked continuation loss was 6.65 at step 46,500. Removing Mamba
did not fix generation.

### V5 — character noise

In a separate run from the earlier MLX self-draft test below, I tried random
character noise instead of MASK tokens. This CUDA model has
16 attention layers and about 218 million parameters. It uses a 96-character
alphabet, a 1,024-character context, a 256-character generation block, a
192-character overlap, and 32 denoising steps. I converted the same book corpus
to characters.

I stopped at step 65,450. Full-noise validation loss barely changed: 3.041 at
step 1,000 and 3.033 at step 65,000, with a best of 3.028 at step 53,000. A
simple character-frequency baseline gives about 3.038 on those validation
fragments. Low-noise loss fell from 0.094 to 0.028, but the three fixed prompts
still produced similar made-up English. The sample script printed only 128
characters, so those samples did not test the overlap. I kept the best weights
and stopped the run rather than train through the rest of the corpus.

## A couple of V1 samples

These are two samples I kept from the first version. You can see phrases starting
to form, along with the repetition and awkward wording I was dealing with.

**Step 6,250**

![V1 text sample at step 6250](screenshots/v1-sample-step-006250.png)

**Step 35,500**

![V1 text sample at step 35500](screenshots/v1-sample-step-035500.png)

## V4 setup

| Setting | Value |
| --- | --- |
| Parameters | 262,902,272 |
| Precision | FP32 |
| Context | 1,024 tokens |
| Blocks | 8, applied twice with shared weights |
| Attention | Bidirectional self-attention with RoPE |
| Embeddings | Shared with the output projection |
| Generation window / overlap / stride | 16 / 12 / 4 tokens |
| Refinement rounds | 12 |
| Batch / gradient accumulation | 4 / 8, giving 32 examples per update |
| Training length | 61,035 updates, one pass through the source windows |

## Data

For V4, I use [PG-19](https://huggingface.co/datasets/emozilla/pg19) and English books
from [Project Gutenberg](https://huggingface.co/datasets/manu/project_gutenberg),
both downloaded from Hugging Face. The prepared corpus contains
**2,000,001,433 training tokens** and **10,000,913 validation tokens**.

I split by book, exclude repeated Gutenberg IDs across the two sources, and
remove exact duplicate normalized fragments. Older English, poetry, and lists
still show up, and near-duplicates may remain. Source revisions and checksums
are saved in `data/v4_books/manifest.json`.

## Running V4

I use a local environment with MLX, mlx-recurrence, transformers, NumPy, psutil,
and Aim. From the project root:

```sh
set -o pipefail
.venv/bin/python -u train.py 2>&1 | tee -a train_v4.log
```

Training resumes from the latest valid full V4 checkpoint, including AdamW state.
To stop, press `Ctrl+C` once and wait for the save confirmation. The `-u` flag
makes console output appear without buffering.

I keep three full periodic checkpoints, saved every **2,500 steps**, one replaceable
emergency checkpoint for interruptions, and one best model by validation loss.
The best model contains weights and metadata; full checkpoints also include
optimizer state. A full V4 checkpoint takes about **2.94 GiB**, and weights alone
about **0.98 GiB**. New emergency and best saves are written before replacing
the previous copy.

Example progress line:

```text
Step 25/61035 | Loss 10.2993 | LR 3.00e-06 | Grad 51.37 | 522 tok/s | 12.32 s / 325.40 s | RAM 48.0%
```

The two times are the last training step's computation time and elapsed time
since the previous log line, usually 25 steps. `tok/s` counts processed input
tokens, including context. RAM is system-wide memory usage.

Every 500 steps, I log validation and three fixed prompts: ordinary prose,
water boiling, and addition. Each gets up to 32 new tokens, with fixed seeds
so I can compare samples over time.

`prepare_data.py` supports resuming data preparation and also needs
huggingface-hub and PyArrow. Data, checkpoints, tests, experiments, and old
version archives stay local and are excluded from Git.

## Later experiments

I also ported V4 to CUDA and trained it on a rented RTX 3090. The run reached
step 38,475, and I saved a full checkpoint with the optimizer state. Training
was much faster than on my Mac, but the generated text was still incoherent.
At step 38,000, the validation loss for a fully masked continuation block was
about 6.65. Faster hardware let me test the model sooner; it did not solve the
generation problem by itself.

I then tried V5 as a separate MLX experiment on my Mac, starting from that
CUDA checkpoint. In V5, some training examples used a continuation drafted by
the model itself: I masked its lower-confidence tokens again and trained it to
recover the corresponding book text. By step 39,500, the loss on this draft
task had improved, but the fully masked continuation loss had not, and the
fixed-prompt samples became dominated by commas and common function words.
I stopped at step 39,504 and kept the checkpoints. This experiment did not
improve free text generation, so I am not treating its lower draft loss as a
successful result.