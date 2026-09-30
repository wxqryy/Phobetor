#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install 'torch==2.7.1+cu118' --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name() if torch.cuda.is_available() else "CUDA unavailable")'
