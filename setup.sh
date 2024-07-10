#!/usr/bin/env bash

# Option 1: Create venv in home directory
VENV_PATH="${HOME}/.virtualenvs/conv_gan"
# Option 2: Create venv in project directory
# VENV_PATH="$(dirname ${0})/venv"

echo "Setting up environment..."
if ! command -v python3 &> /dev/null || ! python3 -c "import venv" &> /dev/null; then
    echo "Either Python 3 is not installed or the venv module is missing. Installing them requires sudo access."
    sudo apt install python3.10 python3.10-venv
fi

# Create virtual environment (if it doesn't exist)
if [ ! -d "${VENV_BASE_PATH}${VENV_NAME}" ]; then
    python3 -m venv "${VENV_PATH}"
else
    echo "Virtual environment already exists. Skipping creation."
fi
source "${VENV_PATH}/bin/activate"

# Install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

## We include the pre-processed data in the repository, so we don't need to download it anymore
## Make the data processing scripts executable, checking for existence first
## Determine the script's directory to locate related scripts
#SCRIPT_DIR="$(dirname "${0}")"
#if [ -f "${SCRIPT_DIR}/pre-processing.sh" ]; then
#    chmod +x "${SCRIPT_DIR}/pre-processing.sh"
#else
#    echo "pre-processing.sh not found."
#fi
#
#if [ -f "${SCRIPT_DIR}/download_geolife.sh" ]; then
#    chmod +x "${SCRIPT_DIR}/download_geolife.sh"
#else
#    echo "download_geolife.sh not found."
#fi
#
## Execute the download script if it exists
#if [ -x "${SCRIPT_DIR}/download_geolife.sh" ]; then
#    "${SCRIPT_DIR}/download_geolife.sh"
#else
#    echo "download_geolife.sh is not executable or missing."
#fi
