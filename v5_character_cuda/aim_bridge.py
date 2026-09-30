import json
import re
import time
from pathlib import Path

from aim import Run, Text


ROOT = Path('/run')
STATE_PATH = Path('/repo/bridge_state_v5.json')
TRAIN_LOG = ROOT / 'train.log'
SAMPLE_LOG = ROOT / 'latest_samples_v5_cuda.log'
STEP_PATTERN = re.compile(
    r'^Step (\d+)/(\d+) \| Loss ([\d.eE+-]+) \| LR ([\d.eE+-]+) '
    r'\| Grad ([\d.eE+-]+) \| ([\d.]+) char/s \| ([\d.]+) s / [\d.]+ s '
    r'\| RAM ([\d.]+)% \| VRAM peak ([\d.]+) GiB'
)
VAL_PATTERN = re.compile(
    r'^VAL (\d+) \| full-noise ([\d.eE+-]+) \| low-noise ([\d.eE+-]+) '
    r'\| overlap-new ([\d.eE+-]+)'
)
PREP_PATTERN = re.compile(r'^(train|val): ([\d,]+)/([\d,]+) source tokens; ([\d,]+) characters')


def save_state(state):
    temporary = STATE_PATH.with_suffix('.tmp')
    temporary.write_text(json.dumps(state), encoding='utf-8')
    temporary.replace(STATE_PATH)


def track_line(run, line):
    print(line, flush=True)
    match = STEP_PATTERN.match(line)
    if match:
        step = int(match.group(1))
        values = {
            'train_loss': float(match.group(3)),
            'learning_rate': float(match.group(4)),
            'grad_norm': float(match.group(5)),
            'characters_per_second': float(match.group(6)),
            'step_seconds': float(match.group(7)),
            'ram_percent': float(match.group(8)),
            'vram_peak_gib': float(match.group(9)),
        }
        for name, value in values.items():
            run.track(value, name=name, step=step)
        return
    match = VAL_PATTERN.match(line)
    if match:
        step = int(match.group(1))
        for name, value in zip(
            ('val_full_noise_loss', 'val_low_noise_loss', 'val_overlap_loss'),
            match.groups()[1:],
        ):
            run.track(float(value), name=name, step=step)
        return
    match = PREP_PATTERN.match(line)
    if match:
        split = match.group(1)
        current = int(match.group(2).replace(',', ''))
        total = int(match.group(3).replace(',', ''))
        step = current // 1_000_000
        run.track(100.0 * current / total, name='preparation_percent',
                  step=step, context={'split': split})


def track_sample(run, line):
    data = json.loads(line)
    for label, prompt, continuation in data.get('samples', []):
        run.track(Text(prompt + ' >>> ' + continuation), name='sample',
                  step=data['step'], context={'prompt': label})


def consume(path, state, key, callback):
    if not path.exists():
        return False
    offset = state.get(key, 0)
    if path.stat().st_size < offset:
        offset = 0
    changed = False
    with path.open('r', encoding='utf-8', errors='replace') as stream:
        stream.seek(offset)
        while line := stream.readline():
            if not line.endswith('\n'):
                break
            callback(line.rstrip('\r\n'))
            state[key] = stream.tell()
            save_state(state)
            changed = True
    return changed


def main():
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    options = {'repo': '/repo', 'system_tracking_interval': None,
               'capture_terminal_logs': True, 'force_resume': True}
    if state.get('run_hash'):
        run = Run(run_hash=state['run_hash'], **options)
    else:
        run = Run(experiment='phobetor_v5_character_cuda', **options)
        run['model'] = {'context_window': 1024, 'micro_batch': 16,
                        'effective_batch': 16, 'device': 'RTX 3070 Ti 8GB'}
        state['run_hash'] = run.hash
        save_state(state)
    print(f'Aim bridge ready: {run.hash}', flush=True)
    if TRAIN_LOG.exists():
        with TRAIN_LOG.open('r', encoding='utf-8', errors='replace') as stream:
            recent = stream.readlines()[-80:]
        print('Recent training log before Aim bridge restart:', flush=True)
        for line in recent:
            print(line.rstrip('\r\n'), flush=True)
    try:
        while True:
            changed = consume(TRAIN_LOG, state, 'train_offset', lambda line: track_line(run, line))
            changed |= consume(SAMPLE_LOG, state, 'sample_offset', lambda line: track_sample(run, line))
            if not changed:
                time.sleep(2)
    finally:
        run.close()


if __name__ == '__main__':
    main()
