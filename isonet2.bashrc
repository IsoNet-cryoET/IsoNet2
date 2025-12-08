#!/bin/bash
ISONET_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PATH="$ISONET_DIR/IsoNet/bin:$PATH"
export PYTHONPATH="$ISONET_DIR:$PYTHONPATH"

if [ -x "$ISONET_DIR/isoapp-1.0.0.AppImage" ]; then
    alias IsoNet2="$ISONET_DIR/isoapp-1.0.0.AppImage"
elif [ -x "$ISONET_DIR/../isoapp-1.0.0.AppImage" ]; then
    alias IsoNet2="$ISONET_DIR/../isoapp-1.0.0.AppImage"
elif [ -f "$ISONET_DIR/isoapp-1.0.0.AppImage" ]; then
    chmod +x "$ISONET_DIR/isoapp-1.0.0.AppImage" 2>/dev/null || true
    alias IsoNet2="$ISONET_DIR/isoapp-1.0.0.AppImage"
elif [ -f "$ISONET_DIR/../isoapp-1.0.0.AppImage" ]; then
    chmod +x "$ISONET_DIR/../isoapp-1.0.0.AppImage" 2>/dev/null || true
    alias IsoNet2="$ISONET_DIR/../isoapp-1.0.0.AppImage"
else
    alias IsoNet2='echo "IsoApp AppImage not found: expected isoapp-1.0.0.AppImage or ../isoapp-1.0.0.AppImage"'
fi

conda activate isonet2_environment
