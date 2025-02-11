#!/usr/bin/env bash
set -euo pipefail

########################################
# Change to the script’s directory
########################################
cd "$(dirname "${BASH_SOURCE[0]}")"

########################################
# Check if the current path has spaces.
# (Miniconda cannot be silently installed in paths with spaces.)
########################################
if [[ "$(pwd)" =~ " " ]]; then
    echo "This script relies on Miniconda which cannot be silently installed under a path with spaces."
    exit 1
fi

########################################
# Install system-level build dependencies.
# These are required for building packages like sentencepiece and external repos.
########################################
echo "Installing system-level build dependencies: build-essential, cmake, pkg-config, ninja-build..."
apt-get update
apt-get install -y build-essential cmake pkg-config ninja-build

########################################
# Deactivate any active conda environments to avoid conflicts.
########################################
{ conda deactivate && conda deactivate && conda deactivate; } 2>/dev/null

########################################
# Determine system architecture.
########################################
OS_ARCH=$(uname -m)
case "${OS_ARCH}" in
    x86_64*)    OS_ARCH="x86_64" ;;
    arm64*|aarch64*)     OS_ARCH="aarch64" ;;
    *)          echo "Unknown system architecture: $OS_ARCH! This script runs only on x86_64 or aarch64." && exit 1 ;;
esac

########################################
# Configuration variables.
########################################
INSTALL_DIR="$(pwd)/installer_files"
CONDA_ROOT_PREFIX="$(pwd)/installer_files/conda"
INSTALL_ENV_DIR="$(pwd)/installer_files/env"
MINICONDA_DOWNLOAD_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${OS_ARCH}.sh"
conda_exists="F"

########################################
# Check if Conda is already installed in our target directory.
########################################
if "$CONDA_ROOT_PREFIX/bin/conda" --version &>/dev/null; then
    conda_exists="T"
fi

########################################
# If Conda is not installed, download and install Miniconda.
########################################
if [ "$conda_exists" == "F" ]; then
    echo "Downloading Miniconda from $MINICONDA_DOWNLOAD_URL to $INSTALL_DIR/miniconda_installer.sh"
    mkdir -p "$INSTALL_DIR"
    curl -L "$MINICONDA_DOWNLOAD_URL" -o "$INSTALL_DIR/miniconda_installer.sh"
    chmod +x "$INSTALL_DIR/miniconda_installer.sh"
    bash "$INSTALL_DIR/miniconda_installer.sh" -b -p "$CONDA_ROOT_PREFIX"
    echo "Miniconda version:"
    "$CONDA_ROOT_PREFIX/bin/conda" --version
    rm "$INSTALL_DIR/miniconda_installer.sh"
fi

########################################
# Create the Conda environment for SpongeQuant if it doesn’t already exist.
########################################
if [ ! -e "$INSTALL_ENV_DIR" ]; then
    echo "Creating Conda environment in $INSTALL_ENV_DIR"
    "$CONDA_ROOT_PREFIX/bin/conda" create -y -k --prefix "$INSTALL_ENV_DIR" python=3.11
fi

########################################
# Check if the Conda environment was successfully created.
########################################
if [ ! -e "$INSTALL_ENV_DIR/bin/python" ]; then
    echo "Conda environment is empty."
    exit 1
fi

########################################
# Set environment isolation variables.
########################################
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export CUDA_PATH="$INSTALL_ENV_DIR"
export CUDA_HOME="$CUDA_PATH"

########################################
# Activate the Conda environment.
########################################
source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh"
conda activate "$INSTALL_ENV_DIR"

########################################
# Dynamically install Python dependencies based on GPU availability.
########################################
if command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected: installing GPU CUDA dependencies..."
    pip install -r src/requirements.gpu-cuda.txt
else
    echo "No NVIDIA GPU detected: installing CPU-only dependencies..."
    pip install -r src/requirements.cpu.txt
fi

########################################
# Determine if a GPU is present (NVIDIA or AMD).
########################################
GPU_FOUND=0
if command -v nvidia-smi &>/dev/null; then
    GPU_FOUND=1
elif command -v rocm-smi &>/dev/null; then
    GPU_FOUND=1
fi

########################################
# Clone and build external repositories.
########################################

# --- Clone and build llama.cpp (always needed) ---
if [ ! -d "llama_cpp" ]; then
    echo "Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp.git llama_cpp
fi
cd llama_cpp
if [ ! -d "build" ]; then
    echo "Building llama.cpp binaries..."
    mkdir build
    cd build
    cmake -G Ninja ..
    ninja
else
    echo "llama.cpp binaries already built."
    cd build
fi
cd ../..

# --- Clone and install AutoAWQ (only if a GPU is found) ---
if [ "$GPU_FOUND" -eq 1 ]; then
    if [ ! -d "AutoAWQ" ]; then
        echo "Cloning AutoAWQ repository..."
        git clone https://github.com/casper-hansen/AutoAWQ.git AutoAWQ
        cd AutoAWQ
        git checkout v0.2.4
        pip install -e .
        cd ..
    else
        echo "AutoAWQ repository already present."
    fi
else
    echo "No GPU detected; skipping AutoAWQ installation."
fi

# --- Clone and install exllamav2 (only if a GPU is found) ---
if [ "$GPU_FOUND" -eq 1 ]; then
    if [ ! -d "exllamav2" ]; then
        echo "Cloning exllamav2 repository..."
        git clone https://github.com/turboderp-org/exllamav2.git exllamav2
        cd exllamav2
        pip install -e .
        cd ..
    else
        echo "exllamav2 repository already present."
    fi
else
    echo "No GPU detected; skipping exllamav2 installation."
fi

########################################
# Run the SpongeQuant application.
# (Assuming the main script is at src/app.py)
########################################
python src/app.py "$@"

########################################
# Pause before exit.
########################################
read -p "Press any key to exit..." -n1 -s
echo