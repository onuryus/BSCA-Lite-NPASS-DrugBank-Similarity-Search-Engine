#!/bin/bash
# prepare_databases.sh
# Builds NPASS and DrugBank binary indexes from raw data.
# Run once before the first search session.

set -e

BUILD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONF="$BUILD/config.sh"
if [ ! -f "$CONF" ]; then
    echo "ERROR: $CONF not found."
    echo "  cp $BUILD/config.sh.example $BUILD/config.sh"
    echo "  then edit config.sh and fill in your paths."
    exit 1
fi
source "$CONF"

if [ -z "$IDIR" ]; then
    echo "ERROR: IDIR is not set. Edit config.sh."
    exit 1
fi

mkdir -p "$IDIR/npass" "$IDIR/drugbank"

cd "$BUILD"

# ── Step 1: NPASS ─────────────────────────────────────────────────────────────
if [ -z "$NPASS_RAW" ]; then
    echo "[SKIP] NPASS_RAW not set — skipping NPASS"
elif [ ! -d "$NPASS_RAW" ]; then
    echo "ERROR: NPASS_RAW directory not found: $NPASS_RAW"
    exit 1
else
    echo "=== Step 1: Building NPASS index ==="
    ./prepare_npass --npass-dir "$NPASS_RAW" --outdir "$IDIR/npass"
    echo ""
fi

# ── Step 2: DrugBank ──────────────────────────────────────────────────────────
if [ -z "$DRUGBANK_XML" ]; then
    echo "[SKIP] DRUGBANK_XML not set — skipping DrugBank"
elif [ ! -f "$DRUGBANK_XML" ]; then
    echo "ERROR: DrugBank XML not found: $DRUGBANK_XML"
    exit 1
else
    echo "=== Step 2: Building DrugBank index ==="
    ./prepare_drugbank --xml "$DRUGBANK_XML" --outdir "$IDIR/drugbank"
    echo ""
fi

echo "=== Preparation complete ==="
echo "  NPASS index  : $IDIR/npass"
echo "  DrugBank bin : $IDIR/drugbank/drugbank.bin"
echo ""
echo "Now run: bash search_lite.sh"
