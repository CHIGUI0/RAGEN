#!/bin/bash

# Exit on error
set -e

echo "Setting up webshop..."
echo "NOTE: please run scripts/setup_ragen.sh before running this script"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

# Main installation process
# Check if conda is available
if command -v conda &> /dev/null; then
    # Need to source conda for script environment
    eval "$(conda shell.bash hook)"
    print_step "Activating conda environment 'ragen'..."
    conda activate ragen
else
    echo -e "${GREEN}Conda not found, assuming python environment is already set up...${NC}"
fi


# Install remaining requirements
print_step "Installing additional requirements..."
# We explicitly install requirements here but skip the webshop recursion in favor of manual handling below
pip install -r requirements.txt

# Install webshop requirements
print_step "Installing webshop minimal requirements..."
# Pyserini installs torch by default which overrides our optimized torch.
# We install it without dependencies, then install its non-conflicting dependencies manually.
pip install pyserini --no-dependencies

# Install other requirements for webshop, excluding pyserini which we just installed
if [ -f "external/webshop-minimal/requirements.txt" ]; then
    grep -v "pyserini" external/webshop-minimal/requirements.txt | pip install -r /dev/stdin
else
    echo "Warning: external/webshop-minimal/requirements.txt not found!"
fi

# webshop installation, model loading
pip install -e external/webshop-minimal/ --no-dependencies
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

print_step "Downloading data..."
python scripts/download_data.py

# Optional: download full data set
print_step "Downloading full data set..."
conda install conda-forge::gdown
mkdir -p external/webshop-minimal/webshop_minimal/data/full
cd external/webshop-minimal/webshop_minimal/data/full
gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB # items_shuffle
gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi # items_ins_v2
cd ../../../../..

echo -e "${GREEN}Installation completed successfully!${NC}"
echo "To activate the environment, run: conda activate ragen"

