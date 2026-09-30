import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from characters import VOCAB_SIZE, encode


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while part := stream.read(8 * 1024 * 1024):
            value.update(part)
    return value.hexdigest()


def convert_split(source, destination, tokenizer, source_id, split):
    input_ids = np.memmap(source, dtype=np.uint16, mode='r')
    state_path = destination.with_suffix('.state.json')
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'source_id': source_id, 'source_tokens': len(input_ids),
        'source_offset': 0, 'characters': 0, 'documents': 0, 'complete': False,
    }
    if state['source_id'] != source_id or state['source_tokens'] != len(input_ids):
        raise ValueError(f'{split}: source corpus changed after conversion started')
    if destination.exists() and not state_path.exists():
        raise ValueError(f'{split}: output exists without a progress file')
    if state['complete']:
        if destination.stat().st_size != state['characters']:
            raise ValueError(f'{split}: completed output has the wrong size')
        return state
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = []
    cursor = state['source_offset']
    last_saved = cursor
    with destination.open('r+b' if destination.exists() else 'w+b') as output:
        output.truncate(state['characters'])
        output.seek(state['characters'])
        while cursor < len(input_ids):
            end = min(cursor + 16384, len(input_ids))
            window = input_ids[cursor:end]
            last = 0
            for position in np.flatnonzero(window == tokenizer.eos_token_id):
                pending.append(window[last:position])
                ids = np.concatenate(pending).astype(np.int64).tolist()
                text = tokenizer.decode(ids, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                encoded = bytes(encode(text + '\n\n'))
                output.write(encoded)
                state['characters'] += len(encoded)
                state['documents'] += 1
                state['source_offset'] = cursor + int(position) + 1
                pending.clear()
                last = int(position) + 1
            if last < len(window):
                pending.append(window[last:])
            cursor = end
            if state['source_offset'] - last_saved >= 1_000_000:
                output.flush()
                os.fsync(output.fileno())
                save_json(state_path, state)
                last_saved = state['source_offset']
                print(f'{split}: {cursor:,}/{len(input_ids):,} source tokens; '
                      f'{state["characters"]:,} characters', flush=True)
        if pending:
            ids = np.concatenate(pending).astype(np.int64).tolist()
            encoded = bytes(encode(tokenizer.decode(ids, skip_special_tokens=False,
                                                     clean_up_tokenization_spaces=False)))
            output.write(encoded)
            state['characters'] += len(encoded)
        output.flush()
        os.fsync(output.fileno())
    state['source_offset'] = len(input_ids)
    state['complete'] = True
    state['sha256'] = digest(destination)
    save_json(state_path, state)
    print(f'{split}: complete; {state["characters"]:,} characters', flush=True)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/v4_books')
    parser.add_argument('--tokenizer', default='tokenizer')
    parser.add_argument('--output', default='data/v5_characters')
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    published = output / 'manifest.json'
    if published.is_file():
        existing = json.loads(published.read_text())
        if existing.get('version') != 5 or existing.get('vocab_size') != VOCAB_SIZE:
            raise ValueError('Existing character corpus manifest is incompatible')
        for split in ('train', 'val'):
            info = existing['splits'][split]
            path = output / info['file']
            if path.stat().st_size != info['characters'] or digest(path) != info['sha256']:
                raise ValueError(f'{split}: published character corpus differs from manifest')
        print(f'Ready: {output} | corpus {existing["corpus_id"]}', flush=True)
        return
    manifest = json.loads((source / 'manifest.json').read_text())
    tokenizer_path = Path(args.tokenizer)
    if digest(tokenizer_path / 'tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer does not match the V4 source corpus')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.eos_token_id != manifest['eos_token_id']:
        raise ValueError('EOS token differs from corpus metadata')
    results = {}
    for split in ('train', 'val'):
        info = manifest['splits'][split]
        source_file = source / info['file']
        if source_file.stat().st_size != info['tokens'] * 2:
            raise ValueError(f'{split}: source corpus size differs from manifest')
        results[split] = convert_split(source_file, output / f'{split}_chars.bin', tokenizer,
                                       manifest['corpus_id'], split)
    content = {'version': 5, 'name': 'phobetor_v5_ascii_character_books',
               'source_corpus_id': manifest['corpus_id'], 'vocab_size': VOCAB_SIZE,
               'alphabet': 'newline_plus_printable_ascii_32_through_126',
               'splits': {split: {'file': f'{split}_chars.bin',
                                  'characters': result['characters'],
                                  'sha256': result['sha256'],
                                  'documents': result['documents']}
                          for split, result in results.items()}}
    content['corpus_id'] = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    save_json(output / 'manifest.json', content)
    print(f'Ready: {output} | corpus {content["corpus_id"]}', flush=True)


if __name__ == '__main__':
    main()
