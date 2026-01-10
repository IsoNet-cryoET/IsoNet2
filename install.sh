#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
mkdir -p ${SCRIPT_DIR}/build

# Checking if Mamba exists, and using that instead
if command -v mamba &> /dev/null; then
    CONDA_SOLVER="mamba"
    echo "Using mamba for environment creation."
else
    CONDA_SOLVER="conda"
    echo "Using conda for environment creation."
fi

# Install Conda env
${CONDA_SOLVER} env create -p "${SCRIPT_DIR}/build/conda_env" -f "${SCRIPT_DIR}/isonet2_environment.yml"

# Intall IsoNet via pip
cd ${SCRIPT_DIR}
"${SCRIPT_DIR}/build/conda_env/bin/pip" install .

# Download GUI
curl -L "https://github.com/IsoNet-cryoET/IsoNet2/releases/download/IsoNet-2.0.0-beta/isoapp-1.0.0.AppImage" -o ${SCRIPT_DIR}/build/isoapp-1.0.0.AppImage
chmod +x ${SCRIPT_DIR}/build/isoapp-1.0.0.AppImage

# Source bashrc file
source "${SCRIPT_DIR}/isonet2.bashrc"
