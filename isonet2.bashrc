#!/bin/bash
ISONET_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PATH="$ISONET_DIR/IsoNet/bin:$PATH"
export PYTHONPATH="$ISONET_DIR:$PYTHONPATH"