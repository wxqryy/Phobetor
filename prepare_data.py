"""V4 prose corpus: pinned sources, document split, exact dedup, atomic publication."""
import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import unicodedata
import numpy as np

SOURCES = (dict(name='pg19', repo='emozilla/pg19',
                revision='c021754c8e01c5b1cc83a1f549c1f97fbbb756b8', fraction=1.0,
                license='PG-19 Apache-2.0; Project Gutenberg book terms',
                files=['data/train-00000-of-00023-5263dc4323e881d2.parquet', 'data/train-00001-of-00023-2b13c68dbeba63b5.parquet', 'data/train-00002-of-00023-8e4dd6b70d51dfcf.parquet', 'data/train-00003-of-00023-e7f0ba0d3c2eb6c0.parquet', 'data/train-00004-of-00023-3f9e7e096fa07f4b.parquet', 'data/train-00005-of-00023-87dece6a0ab708e8.parquet', 'data/train-00006-of-00023-c50fea56d0518a87.parquet', 'data/train-00007-of-00023-e9fea5c158e172e2.parquet', 'data/train-00008-of-00023-275017526d706c9d.parquet', 'data/train-00009-of-00023-f07a29af90b71085.parquet', 'data/train-00010-of-00023-dd5d8583a9f6a549.parquet', 'data/train-00011-of-00023-6c0bc91e6df2e661.parquet', 'data/train-00012-of-00023-78d820574bddc6bb.parquet', 'data/train-00013-of-00023-5b853732c713dded.parquet', 'data/train-00014-of-00023-54b567998cd5eb4b.parquet', 'data/train-00015-of-00023-7a78b1e66cfe52de.parquet', 'data/train-00016-of-00023-8269d10108995a28.parquet', 'data/train-00017-of-00023-e240b232c0daddca.parquet', 'data/train-00018-of-00023-f59d477d80180c78.parquet', 'data/train-00019-of-00023-198af082bd917d87.parquet', 'data/train-00020-of-00023-f9b058c5e9338514.parquet', 'data/train-00021-of-00023-e221306b7359a366.parquet', 'data/train-00022-of-00023-5a956eb2a5d6cab5.parquet']),)
SOURCES += ({'name': 'gutenberg_en', 'repo': 'manu/project_gutenberg', 'revision': '164853d214065df26a630ee1ab91a0c39e461caf', 'format': 'gutenberg', 'license': 'Project Gutenberg book terms', 'files': ['data/en-00000-of-00052-7cda8f63c262acf8.parquet', 'data/en-00001-of-00052-5c2b3fd5e60f0124.parquet', 'data/en-00002-of-00052-835bd07d97f52cbd.parquet', 'data/en-00003-of-00052-3827386b583e4d76.parquet', 'data/en-00004-of-00052-a2f24c4fe858fe0f.parquet', 'data/en-00005-of-00052-2a13fc98474cabed.parquet', 'data/en-00006-of-00052-81d2618caede0093.parquet', 'data/en-00007-of-00052-de0de442f370b789.parquet', 'data/en-00008-of-00052-a83055e8ef415c07.parquet', 'data/en-00009-of-00052-f2f126633fa25668.parquet', 'data/en-00010-of-00052-2226b3722aa696eb.parquet', 'data/en-00011-of-00052-6c9ae05ed451701f.parquet', 'data/en-00012-of-00052-2de5b14941be3266.parquet', 'data/en-00013-of-00052-a66a5e317603bb21.parquet', 'data/en-00014-of-00052-e976ff9fa7c0a4c2.parquet', 'data/en-00015-of-00052-9a9fd49be8a70a6c.parquet', 'data/en-00016-of-00052-5006e8c00e35ad72.parquet', 'data/en-00017-of-00052-c37121d3035604a6.parquet', 'data/en-00018-of-00052-76fc57ebfaac39a2.parquet', 'data/en-00019-of-00052-05068b06d8a4ffb7.parquet', 'data/en-00020-of-00052-31ef1cece5305678.parquet', 'data/en-00021-of-00052-82812d42cefe9a5b.parquet', 'data/en-00022-of-00052-061a44c5aeff4f98.parquet', 'data/en-00023-of-00052-380917a82781d4aa.parquet', 'data/en-00024-of-00052-fb2f9a960ee8c75e.parquet', 'data/en-00025-of-00052-9c00570871767dea.parquet', 'data/en-00026-of-00052-8719637a331ec653.parquet', 'data/en-00027-of-00052-9d8bfd7843e9718f.parquet', 'data/en-00028-of-00052-2e237a8d8b810dec.parquet', 'data/en-00029-of-00052-780d5f400fd85afa.parquet', 'data/en-00030-of-00052-cfdc6f381f17e852.parquet', 'data/en-00031-of-00052-e7f5d815a26b08d0.parquet', 'data/en-00032-of-00052-4ed9cbf89e3d13b5.parquet', 'data/en-00033-of-00052-4c525b2c5bfc7b3f.parquet', 'data/en-00034-of-00052-0f04bcad9d91ea41.parquet', 'data/en-00035-of-00052-617bd55ce8eaaf8c.parquet', 'data/en-00036-of-00052-dea698a178ee5475.parquet', 'data/en-00037-of-00052-80239e718491affb.parquet', 'data/en-00038-of-00052-671117f0cc621546.parquet', 'data/en-00039-of-00052-f766a27e24b911d6.parquet', 'data/en-00040-of-00052-5f304ed689a13135.parquet', 'data/en-00041-of-00052-6f04cd012627fa08.parquet', 'data/en-00042-of-00052-94e90b865e11f015.parquet', 'data/en-00043-of-00052-545c5249d7f68142.parquet', 'data/en-00044-of-00052-2f8a81e4b2cb26bd.parquet', 'data/en-00045-of-00052-bbb08eac7b16b553.parquet', 'data/en-00046-of-00052-fbf5b6f877101255.parquet', 'data/en-00047-of-00052-b53a9aba6fd08df1.parquet', 'data/en-00048-of-00052-c3e4665ddb21ff40.parquet', 'data/en-00049-of-00052-238a44ba6d899475.parquet', 'data/en-00050-of-00052-119c77546b4d5bc3.parquet', 'data/en-00051-of-00052-416e3a1d8d8e7d86.parquet']},)
FILTER_VERSION = 4


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def clean_prose(raw):
    """Reject technical/noisy documents, not just their individual symbols."""
    text = unicodedata.normalize('NFKC', html.unescape(raw)).strip()
    if re.search(r'```|`|\\[A-Za-z]+|\$|[{}=<>^|\[\]]|[∑∫√≤≥≈±×÷]', text):
        return None, 'code_or_formula'
    if re.search(r'https?://|www\.|\b(?:import\s+\w+|def\s+\w+\(|print\()', text):
        return None, 'link_or_code'
    if re.search(r'click here|subscribe|all rights reserved|cookie policy|privacy policy|'
                 r'add to cart|sign up|refer slide time|as an ai|language model', text, re.I):
        return None, 'boilerplate'
    if len(re.findall(r'\b\d{1,2}:\d{2}\b', text)) >= 2:
        return None, 'timestamps'
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or re.match(r'^#{1,6}\s', line):
            continue
        plain_line = line.strip('* ')
        if re.fullmatch(r'-{3,}', plain_line):
            continue
        if re.match(r'^(?:unit title|title|key term\s*\d*|thought experiment|interactive element|'
                    r'learning objectives?|section|chapter|summary)\s*:', plain_line, re.I):
            if len(plain_line.split()) <= 25 and not plain_line.endswith(('.', '?', '!')):
                continue
        if re.fullmatch(r'\*{1,2}[^*]+\*{1,2}', line) and len(line.split()) < 15:
            continue
        if re.match(r'^(?:[-*•]|\d+[.)])\s', line):
            return None, 'list'
        lines.append(re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', line))
    if not lines:
        return None, 'empty'
    if len(lines) >= 6 and sum(len(line.split()) < 5 for line in lines)/len(lines) > 0.3:
        return None, 'short_lines'
    text = re.sub(r'\s+', ' ', ' '.join(lines)).strip()
    if any(c in text for c in '#_*\\'):
        return None, 'remaining_markup'
    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text)
    if not 60 <= len(words) <= 3000:
        return None, 'length'
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters)/len(letters) < 0.98:
        return None, 'non_english_script'
    if len(letters)/max(1, len(re.sub(r'\s', '', text))) < 0.80:
        return None, 'symbol_density'
    sentences = [s.strip().casefold() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) < 4:
        return None, 'few_sentences'
    if len(sentences) >= 5 and len(set(sentences))/len(sentences) < 0.8:
        return None, 'repetition'
    if len(set(w.lower() for w in words))/len(words) < 0.12:
        return None, 'low_diversity'
    return text, 'accepted'


def document_identity(text):
    return hashlib.sha256(text.casefold().encode('utf-8')).hexdigest()


def document_split(identity):
    return 'val' if int(identity[:16], 16) % 100 == 0 else 'train'


def book_chunks(raw, max_words=1500):
    """Keep book prose; unwrap printing lines, remove front matter/footnote markup."""
    raw = re.split(r'(?im)^\s*End of (?:the )?Project Gutenberg', raw, maxsplit=1)[0]
    paragraphs, size = [], 0
    for paragraph in re.split(r'\n\s*\n', raw):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if re.search(r'project gutenberg|distributed proofread|produced by|transcriber', paragraph, re.I):
            continue
        if re.search(r'(?im)^\s*\[?footnote\b', paragraph):
            continue
        paragraph = re.sub(r'\[(?:\d+|[a-zA-Z])\]', '', paragraph)
        paragraph = re.sub(r'_([^_]+)_', r'\1', paragraph)
        if len(paragraph.split()) < 8 and not re.search(r'[.!?][\"”\')]*$', paragraph):
            continue
        n = len(paragraph.split())
        if paragraphs and size+n > max_words:
            yield '\n'.join(paragraphs)
            paragraphs, size = [], 0
        paragraphs.append(paragraph)
        size += n
    if paragraphs:
        yield '\n'.join(paragraphs)


def gutenberg_id(value):
    match = re.search(r'(?:^|/)([0-9]+)(?:[-./]|$)', str(value))
    if not match:
        raise ValueError(f'Unrecognized Gutenberg book ID: {value!r}')
    return match.group(1)


def strip_gutenberg_wrapper(text):
    start = re.search(r'(?im)^\s*\*{3}\s*START OF[^\n]*\*{3}[^\n]*\n?', text)
    if start:
        text = text[start.end():]
    end = re.search(r'(?im)^\s*\*{3}\s*END OF[^\n]*', text)
    return text[:end.start()] if end else text


def rows(source, start_shard=0, book_ids=None):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    book_ids = set() if book_ids is None else book_ids
    for shard, filename in enumerate(source['files']):
        if shard < start_shard:
            continue
        print(f'Downloading/reading HF shard {shard+1}/{len(source["files"])}: {filename}', flush=True)
        path = hf_hub_download(source['repo'], filename, repo_type='dataset', revision=source['revision'])
        skipped_ids = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                if source.get('format') == 'gutenberg':
                    try:
                        book_id = gutenberg_id(row['id'])
                    except ValueError:
                        skipped_ids += 1
                        continue
                    if book_id in book_ids:
                        continue
                    book_ids.add(book_id)
                    raw = strip_gutenberg_wrapper(row['text'])
                    group = document_identity('gutenberg:'+book_id)
                    title, url = 'Gutenberg '+book_id, 'https://www.gutenberg.org/ebooks/'+book_id
                else:
                    raw = row['text']
                    group = document_identity(re.sub(r'\s+', ' ', raw).strip())
                    title, url = row['short_book_title'], row['url']
                    book_ids.add(gutenberg_id(url))
                for text in book_chunks(raw):
                    yield dict(text=text, group=group, url=url, title=title)
        if skipped_ids:
            print(f'Skipped {skipped_ids} records without a numeric Gutenberg book ID.', flush=True)
        yield dict(shard_done=shard, unrecognized_book_ids=skipped_ids)


def build(output, tokenizer_path, train_target, val_target, restart=False):
    from transformers import AutoTokenizer
    output = Path(output)
    if output.exists():
        manifest_path = output/'manifest.json'
        if not manifest_path.exists():
            raise RuntimeError(f'{output} exists without a completed manifest; refusing to overwrite')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('filter_version') != FILTER_VERSION:
            raise ValueError('Existing corpus uses an older filter; build into a different directory')
        if manifest['requested_tokens'] != dict(train=train_target, val=val_target):
            raise ValueError('Existing corpus has different token targets; choose another output directory')
        if manifest['tokenizer_sha256'] != file_hash(Path(tokenizer_path)/'tokenizer.json'):
            raise ValueError('Existing corpus uses a different tokenizer')
        for info in manifest['splits'].values():
            if file_hash(output/info['file']) != info['sha256']:
                raise ValueError('Existing corpus checksum differs')
        print(f'Complete corpus already verified: {output}', flush=True)
        return manifest
    staging = output.with_name(output.name+'.building')
    if staging.exists() and restart:
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if len(tokenizer) > 65536 or tokenizer.eos_token_id is None or tokenizer.mask_token_id is None:
        raise ValueError('Expected uint16 tokenizer with EOS and MASK')
    tokenizer.model_max_length = 10**9
    signature = dict(sources=SOURCES, filter_version=FILTER_VERSION, train=train_target, val=val_target,
                     tokenizer_sha256=file_hash(Path(tokenizer_path)/'tokenizer.json'))
    signature = json.loads(json.dumps(signature))
    progress_path = staging/'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if progress and progress['signature'] != signature:
        old = progress['signature']
        prefix = len(old['sources'])
        if ({k:v for k,v in old.items() if k != 'sources'} !=
                {k:v for k,v in signature.items() if k != 'sources'} or
                old['sources'] != signature['sources'][:prefix]):
            raise ValueError('Incomplete build configuration differs; use another directory or --restart')
        print('Extending source list; preserving prepared tokens and saved shard position.', flush=True)
    totals = progress.get('totals', dict(train=0, val=0))
    seen = set(progress.get('seen', []))
    source_reports = progress.get('source_reports', [])
    book_ids = set(progress.get('book_ids', []))
    if progress and not book_ids:
        with (staging/'documents.jsonl').open('rb') as old_index:
            while old_index.tell() < progress['index_bytes']:
                entry = json.loads(old_index.readline())
                book_ids.add(gutenberg_id(entry['url']))
        print(f'Excluding {len(book_ids):,} previously prepared Gutenberg book IDs.', flush=True)
    def resume_file(path, size):
        f = path.open('r+b' if path.exists() else 'w+b')
        if path.stat().st_size < size:
            f.close()
            raise ValueError(f'Incomplete staging file: {path}')
        f.truncate(size); f.seek(size)
        return f
    handles = {s: resume_file(staging/f'{s}_tokens.bin', totals[s]*2) for s in totals}
    index = resume_file(staging/'documents.jsonl', progress.get('index_bytes', 0))
    preview = resume_file(staging/'preview.txt', progress.get('preview_bytes', 0))
    try:
        for source_index in range(progress.get('source_index', 0), len(SOURCES)):
            source = SOURCES[source_index]
            counts = progress.get('source_tokens', totals.copy() if source_index == 0 else dict(train=0,val=0))
            stats = Counter(progress.get('stats', {}))
            targets = dict(train=train_target, val=val_target)
            examples, pending = progress.get('examples', 0), []

            def save_progress(next_shard, next_source=source_index):
                for f in (*handles.values(), index, preview):
                    f.flush(); os.fsync(f.fileno())
                state = dict(signature=signature, totals=totals, seen=sorted(seen), book_ids=sorted(book_ids),
                             stats=dict(stats) if next_source == source_index else {},
                             source_tokens=counts if next_source == source_index else dict(train=0,val=0),
                             examples=examples if next_source == source_index else 0,
                             next_shard=next_shard, source_index=next_source, source_reports=source_reports,
                             index_bytes=index.tell(), preview_bytes=preview.tell())
                temporary = staging/'progress.tmp'
                temporary.write_text(json.dumps(state))
                with temporary.open('rb') as f: os.fsync(f.fileno())
                os.replace(temporary, progress_path)
                print(f'Shard saved; tokens={totals}', flush=True)


            def consume():
                nonlocal examples
                if not pending:
                    return
                encoded = tokenizer([item[0] for item in pending], add_special_tokens=False)['input_ids']
                for (text, identity, split, provenance), ids in zip(pending, encoded):
                    if totals[split] >= targets[split]:
                        continue
                    if any(i < 0 or i >= len(tokenizer) or i in (tokenizer.mask_token_id, tokenizer.eos_token_id) for i in ids):
                        stats['special_token_in_text'] += 1
                        continue
                    ids.append(tokenizer.eos_token_id)
                    offset = totals[split]
                    handles[split].write(np.asarray(ids, dtype=np.uint16).tobytes())
                    totals[split] += len(ids); counts[split] += len(ids)
                    stats['accepted_'+split] += 1
                    index.write((json.dumps(dict(hash=identity, source=source['name'], split=split,
                                                offset=offset, tokens=len(ids), **provenance))+'\n').encode())
                    if examples < 12:
                        preview.write(f"SOURCE {source['name']} | {split} | {identity}\n{text}\n\n".encode())
                        examples += 1
                pending.clear()

            for row in rows(source, start_shard=progress.get('next_shard', 0), book_ids=book_ids):
                if 'shard_done' in row:
                    consume()
                    stats['unrecognized_book_ids'] += row.get('unrecognized_book_ids', 0)
                    save_progress(row['shard_done']+1)
                    continue
                stats['read'] += 1
                raw = row['text']
                text, reason = clean_prose(raw)
                if text is None:
                    stats[reason] += 1
                    continue
                identity = document_identity(text)
                if identity in seen:
                    stats['duplicate'] += 1
                    continue
                seen.add(identity)
                split = document_split(row.get('group', identity))
                if totals[split] < targets[split]:
                    provenance = {k: row[k] for k in ('url', 'title') if k in row}
                    pending.append((text, identity, split, provenance))
                if len(pending) >= 256:
                    consume()
                if stats['read'] % 10000 == 0:
                    consume()
                    print(f"{source['name']}: rows={stats['read']:,}, tokens={totals}", flush=True)
                if all(totals[s] >= targets[s] for s in totals):
                    break
            consume()
            source_reports.append({**source, 'tokens': counts.copy(), 'filter_counts': dict(stats)})
            save_progress(0, source_index+1)
            progress = {}
            if all(totals[s] >= targets[s] for s in totals):
                break
        if totals['train'] < train_target or totals['val'] < val_target:
            raise RuntimeError(f'All sources exhausted: {totals}; prepared data saved, no corpus published')
        for f in (*handles.values(), index, preview):
            f.flush(); os.fsync(f.fileno())
    finally:
        for f in (*handles.values(), index, preview):
            f.close()
    manifest = dict(version=4, name='phobetor_v4_plain_english', filter_version=FILTER_VERSION,
                    dtype='uint16', vocab_size=len(tokenizer), eos_token_id=tokenizer.eos_token_id,
                    mask_token_id=tokenizer.mask_token_id,
                    tokenizer_sha256=file_hash(Path(tokenizer_path)/'tokenizer.json'),
                    requested_tokens=dict(train=train_target, val=val_target), sources=source_reports,
                    split_method='PG19: whole-book hash; supplement: Gutenberg ID hash; modulo 100; book-ID exclusion and fragment exact dedup',
                    splits={s:dict(file=f'{s}_tokens.bin', tokens=totals[s], sha256=file_hash(staging/f'{s}_tokens.bin')) for s in totals})
    manifest['corpus_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (staging/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    for name in ('progress.json', 'progress.tmp'):
        (staging/name).unlink(missing_ok=True)
    os.replace(staging, output)
    print(f'Ready: {output} | {totals} | corpus {manifest["corpus_id"]}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='./data/v4_books')
    parser.add_argument('--tokenizer', default='./tokenizer')
    parser.add_argument('--train-tokens', type=int, default=2_000_000_000)
    parser.add_argument('--val-tokens', type=int, default=10_000_000)
    parser.add_argument('--restart', action='store_true', help='Rebuild only an incomplete staging directory')
    args = parser.parse_args()
    if args.train_tokens < 1024 or args.val_tokens < 1024:
        parser.error('Both splits must have at least 1024 tokens')
    build(args.output, args.tokenizer, args.train_tokens, args.val_tokens, args.restart)


if __name__ == '__main__':
    main()
