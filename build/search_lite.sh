#!/bin/bash
# search_lite.sh
# Interactive similarity search against NPASS and DrugBank.
# Databases are loaded once; queries are entered interactively until 'quit'.

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

DRUGBANK_BIN="$IDIR/drugbank/drugbank.bin"

if [ ! -f "$DRUGBANK_BIN" ]; then
    echo "ERROR: DrugBank binary not found: $DRUGBANK_BIN"
    echo "Run: bash prepare_databases.sh"
    exit 1
fi

if [ -z "$NPASS_RAW" ] || [ ! -d "$NPASS_RAW" ]; then
    echo "ERROR: NPASS_RAW directory not found: $NPASS_RAW"
    echo "Set NPASS_RAW in config.sh and run: bash prepare_databases.sh"
    exit 1
fi

cd "$BUILD"

./search_lite \
    --npass-index  "$NPASS_RAW" \
    --npass-fps    "$IDIR/npass/npass_fps.bin" \
    --drugbank-bin "$DRUGBANK_BIN" \
    --databases    npass,drugbank \
    --metric       tanimoto \
    --top          50 \
    --out-mode     both \
    --out          "$IDIR/results.tsv" \
    --no-gpu
