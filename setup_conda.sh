#!/usr/bin/env bash

ENV_NAME="conv_gan"

echo "Setting up conda environment..."
if ! command -v python3 &> /dev/null || ! command -v conda &> /dev/null; then
    echo "Python 3.10 or Conda is not installed. Installing might require sudo access."
    echo "Installing Python 3.10..."
    sudo apt install python3.10
    echo "Downloading Conda..."
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O miniconda.sh
    echo "Installing Conda..."
    bash miniconda.sh -b -p "$HOME/miniconda"
    echo "Initialising Conda..."
    source "$HOME/miniconda/bin/activate"
    conda init
fi

# Update conda
conda update -n base -c defaults conda
# Create environment
conda env create -f envs/env.yml -n $ENV_NAME
# Activate environment
conda activate $ENV_NAME
## We include the pre-processed data in the repository, so we don't need to download it anymore
## Download & pre-process data
#chmod +x pre-processing.sh
#chmod +x download_geolife.sh
#./download_geolife.sh