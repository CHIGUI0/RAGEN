#!/bin/bash
# Setup script for Vast.ai B200/B100 (Blackwell sm_100) machines
# Search benchmark only. For H100/A100 use setup_vast.sh instead.
#
# Installs: conda env, PyTorch 2.8 (cu128), flash-attn 2.8.1, vllm 0.11,
#           verl (submodule), ragen (editable), retrieval server deps, wandb

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_step() { echo -e "${BLUE}[Step] ${1}${NC}"; }
print_ok()   { echo -e "${GREEN}[OK]   ${1}${NC}"; }
print_err()  { echo -e "${RED}[ERR]  ${1}${NC}"; }

# ---------- Ensure we are in the repo root ----------
if [ ! -f "train.py" ]; then
    echo "Please run this script from the RAGEN repo root (where train.py is)."
    exit 1
fi

# ---------- Install conda if missing ----------
install_conda() {
    if command -v conda &>/dev/null; then
        print_ok "Conda already installed"
        return
    fi
    print_step "Installing Miniconda..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    print_ok "Miniconda installed"
}

install_conda

# Source conda for this script session
eval "$(conda shell.bash hook)"

# ---------- Create conda env ----------
if ! conda env list | grep -q "ragen"; then
    print_step "Creating conda environment 'ragen' (Python 3.12)..."
    conda create -n ragen python=3.12 -y
else
    print_ok "Conda env 'ragen' already exists"
fi
conda activate ragen

# ---------- Basic build tools ----------
print_step "Installing pip / setuptools / wheel..."
pip install -U pip "setuptools<70.0.0" wheel
pip install numpy ninja packaging psutil

# ---------- verl submodule ----------
print_step "Initializing verl submodule..."
git submodule update --init verl

# ---------- PyTorch 2.8 + CUDA 12.8 (B200/B100 support) ----------
print_step "Installing PyTorch 2.8.0 (cu128) for Blackwell GPUs..."
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# ---------- vLLM 0.11 (B200 compatible) ----------
print_step "Installing vLLM 0.11.0..."
pip install --no-cache-dir "vllm==0.11.0"

# ---------- flash-attn 2.8.1 (prebuilt wheel, no compilation needed) ----------
print_step "Installing flash-attn 2.8.1 (prebuilt wheel)..."
FLASH_ATTN_WHL="flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
if [ ! -f "$FLASH_ATTN_WHL" ]; then
    wget -nv "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/${FLASH_ATTN_WHL}"
fi
pip install --no-cache-dir "$FLASH_ATTN_WHL"
rm -f "$FLASH_ATTN_WHL"
print_ok "flash-attn 2.8.1 installed (prebuilt, no compilation)"

# ---------- verl + ragen ----------
print_step "Installing verl..."
cd verl && pip install -e . --no-deps && cd ..

print_step "Installing ragen package..."
pip install -e . --no-deps

# ---------- Python requirements (search-relevant only) ----------
print_step "Installing Python requirements (search-relevant only)..."
pip install \
    "transformers>=4.48.2" \
    accelerate datasets peft \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    "tensordict>=0.8.0,<0.9.0" torchdata \
    "ray[default]>=2.10" \
    codetiming hydra-core pylatexenc \
    dill pybind11 \
    IPython matplotlib \
    gym gym_sokoban \
    gymnasium "gymnasium[toy-text]" \
    debugpy together anthropic

# ---------- Retrieval server deps ----------
print_step "Installing retrieval server dependencies..."
pip install flask sentence-transformers faiss-cpu==1.11.0 wandb

# ---------- Verify ----------
print_step "Verifying installation..."
python -c "
import torch
print(f'  torch {torch.__version__}')
assert 'sm_100' in str(torch.cuda.get_arch_list()) or any('100' in a for a in torch.cuda.get_arch_list()), \
    'PyTorch does not support sm_100 (B200/B100). Check torch version.'
import vllm; print(f'  vllm {vllm.__version__}')
import flash_attn; print(f'  flash_attn {flash_attn.__version__}')
import transformers; print(f'  transformers {transformers.__version__}')
import flask; print(f'  flask OK')
import wandb; print(f'  wandb {wandb.__version__}')
import verl; print(f'  verl OK')
t = torch.zeros(1).cuda(); print(f'  GPU test: OK')
print('All checks passed!')
"

# ---------- Done ----------
echo ""
print_ok "Setup complete! (B200/B100 Blackwell)"
echo "To activate:  conda activate ragen"
echo "Next step:    bash scripts/vast/prepare_data.sh"
