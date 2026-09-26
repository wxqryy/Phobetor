#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel packaging ninja
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install 'torch==2.7.1+cu118' --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install --no-deps --no-build-isolation 'git+https://github.com/state-spaces/mamba.git@e9594ce1c732d97440f0332fdc43170a2294dbfa'
.venv/bin/python - <<'PY'
from pathlib import Path
import site

package = next(Path(path) / 'mamba_ssm' / '__init__.py' for path in site.getsitepackages() if (Path(path) / 'mamba_ssm' / '__init__.py').is_file())
package.write_text('__version__ = "2.3.2.post1"\n')
PY
.venv/bin/python -c 'import torch; from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; print("CUDA:", torch.cuda.is_available(), "SSD:", mamba_chunk_scan_combined.__name__)'
