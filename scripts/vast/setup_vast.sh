#!/bin/bash
# Setup script for Vast.ai 8×H200 machine (Search benchmark only)
# Installs conda env, PyTorch+CUDA, flash-attn, vllm, verl, ragen, retrieval deps

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

# ---------- CUDA toolkit ----------
print_step "Checking CUDA..."
if command -v nvcc &>/dev/null; then
    nvcc_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    nvcc_major=$(echo "$nvcc_version" | cut -d. -f1)
    nvcc_minor=$(echo "$nvcc_version" | cut -d. -f2)
    print_ok "Found NVCC $nvcc_version"
    if [[ "$nvcc_major" -gt 12 || ("$nvcc_major" -eq 12 && "$nvcc_minor" -ge 1) ]]; then
        export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}
    else
        print_step "CUDA < 12.4, installing toolkit via conda..."
        conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
        export CUDA_HOME=$CONDA_PREFIX
    fi
else
    print_step "NVCC not found, installing CUDA toolkit 12.4 via conda..."
    conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
    export CUDA_HOME=$CONDA_PREFIX
fi

# ---------- PyTorch ----------
print_step "Installing PyTorch 2.5.0 (cu124)..."
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# ---------- flash-attn ----------
print_step "Installing flash-attn..."
pip install flash-attn==2.7.4.post1 --no-build-isolation

# ---------- verl + ragen ----------
print_step "Installing verl submodule..."
git submodule init && git submodule update
cd verl && pip install -e . --no-dependencies && cd ..

print_step "Installing ragen package..."
pip install -e .

# ---------- Python requirements (skip webshop) ----------
print_step "Installing Python requirements (search-relevant only)..."
# Install everything from requirements.txt except the webshop line and kimina-client
grep -v "webshop" requirements.txt | grep -v "kimina" | pip install -r /dev/stdin

# Pin transformers version used by the project
pip install transformers==4.48.2

# ---------- Retrieval server deps ----------
print_step "Installing retrieval server dependencies..."
pip install flask sentence-transformers faiss-cpu==1.11.0

# ---------- vLLM (already in requirements.txt, ensure correct version) ----------
print_step "Verifying vllm installation..."
python -c "import vllm; print(f'vllm {vllm.__version__} OK')" || {
    print_err "vllm import failed, reinstalling..."
    pip install vllm==0.8.2
}

# ---------- Done ----------
echo ""
print_ok "Setup complete!"
echo "To activate:  conda activate ragen"
echo "Next step:    bash scripts/vast/prepare_data.sh"
