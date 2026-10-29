#!/usr/bin/env bash
# One-command setup:   ./install.sh          (NVIDIA GPU, default)
#                      ./install.sh --cpu    (CPU only)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

MODE=gpu
[[ "${1:-}" == "--cpu" ]] && MODE=cpu

# numpy<2 / onnxruntime wheels need Python 3.10-3.12
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
    for c in python3.11 python3.12 python3.10; do
        command -v "$c" >/dev/null && { PY="$c"; break; }
    done
fi
[[ -n "$PY" ]] || { echo "Python 3.10-3.12 not found (try: PYTHON=/path/to/python3.11 ./install.sh)"; exit 1; }
echo "==> using $($PY --version) ($PY), mode: $MODE"

[[ -d venv ]] || "$PY" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip >/dev/null
PIP=(pip install --default-timeout=120 --retries 5)

if [[ "$MODE" == gpu ]]; then
    "${PIP[@]}" -r requirements.txt
else
    "${PIP[@]}" -r requirements-base.txt -r requirements-cpu.txt
fi
"${PIP[@]}" --no-deps -e .

echo
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found - needed for video files (pacman -S ffmpeg / apt install ffmpeg)"
[[ "$MODE" == gpu ]] && ! command -v nvidia-smi >/dev/null && echo "WARNING: nvidia-smi not found - is the NVIDIA driver installed?"
mkdir -p models
ls models/*.dfm >/dev/null 2>&1 || echo "NOTE: put your .dfm models into ./models/"
echo "==> done. Start with:  ./dfl.sh gui     (or: source venv/bin/activate && dfl --help)"
