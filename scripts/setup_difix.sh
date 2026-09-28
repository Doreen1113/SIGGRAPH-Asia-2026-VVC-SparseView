#!/bin/bash
# Isolated env for Difix3D+ (old diffusers pins would break the ftgspp venv). Weights: nvidia/difix_ref.
set -x
cd /home/intern_2603055/vvc
[ -d /home/intern_2603055/projects/Difix3D ] || git clone --depth 1 https://github.com/nv-tlabs/Difix3D /home/intern_2603055/projects/Difix3D
~/.local/bin/uv venv --python 3.10 /home/intern_2603055/vvc/envs/difix
export VIRTUAL_ENV=/home/intern_2603055/vvc/envs/difix
~/.local/bin/uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
~/.local/bin/uv pip install "diffusers==0.25.1" "transformers==4.38.0" "peft==0.9.0" accelerate safetensors huggingface_hub pillow numpy opencv-python-headless einops lpips torchmetrics scipy
export HF_HOME=/home/intern_2603055/.cache/huggingface
/home/intern_2603055/vvc/envs/difix/bin/python -c "
from huggingface_hub import snapshot_download
for r in ['nvidia/difix_ref','nvidia/difix']:
    p=snapshot_download(r); print('downloaded', r, '->', p)"
/home/intern_2603055/vvc/envs/difix/bin/python -c "import torch,diffusers,transformers; print('env ok', torch.__version__, torch.cuda.is_available(), diffusers.__version__)"
echo "DIFIX SETUP DONE $(date -u +%H:%M)"
