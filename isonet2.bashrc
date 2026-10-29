#!/usr/bin/env bash
ISONET_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PATH="$ISONET_DIR/build/conda_env/bin:$PATH"

if [ -x "$ISONET_DIR/isoapp.AppImage" ]; then
    alias IsoNet2="$ISONET_DIR/isoapp.AppImage"
elif [ -x "$ISONET_DIR/build/isoapp.AppImage" ]; then
    alias IsoNet2="$ISONET_DIR/build/isoapp.AppImage"
elif [ -x "$ISONET_DIR/../isoapp.AppImage" ]; then
    alias IsoNet2="$ISONET_DIR/../isoapp.AppImage"
elif [ -f "$ISONET_DIR/isoapp.AppImage" ]; then
    chmod +x "$ISONET_DIR/isoapp.AppImage" 2>/dev/null || true
    alias IsoNet2="$ISONET_DIR/isoapp.AppImage"
elif [ -f "$ISONET_DIR/build/isoapp.AppImage" ]; then
    chmod +x "$ISONET_DIR/build/isoapp.AppImage" 2>/dev/null || true
    alias IsoNet2="$ISONET_DIR/build/isoapp.AppImage"
elif [ -f "$ISONET_DIR/../isoapp.AppImage" ]; then
    chmod +x "$ISONET_DIR/../isoapp.AppImage" 2>/dev/null || true
    alias IsoNet2="$ISONET_DIR/../isoapp.AppImage"
else
    alias IsoNet2='echo "IsoApp AppImage not found: expected isoapp.AppImage or ../isoapp.AppImage"'
fi
