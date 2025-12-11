#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
conda env create -f "${SCRIPT_DIR}/isonet2_environment.yml"
source "${SCRIPT_DIR}/isonet2.bashrc"