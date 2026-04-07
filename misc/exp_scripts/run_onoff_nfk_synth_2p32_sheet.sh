#!/usr/bin/env bash
set -euo pipefail

# Runs OnOff-NFK/OblRadix on the synthetic 2^32 dataset using the (r,b) rows
# from the provided sheet image.
#
# Parameters:
# - r = NUM_RADIX_BITS
# - b = BINS_PER_PART
#
# Output policy:
# - The output file contains ONLY stdout/stderr produced by ./OblRadix.
# - Build/configure output is suppressed (but failures still stop the script).
#
# Usage:
#   ./run_onoff_nfk_synth_2p32_sheet.sh [threads] [output_file]
#
# Example:
#   ./run_onoff_nfk_synth_2p32_sheet.sh 32 runs/synth_2p32_sheet.txt
#
# Optional env vars:
#   NUM_PASSES=2   (default: 2)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
ONOFF_DIR="$REPO_ROOT/OnOff-NFK"
BUILD_DIR="$ONOFF_DIR/build"

THREADS="${1:-32}"
OUT_FILE="${2:-$REPO_ROOT/runs/onoff_nfk_synth_2p32_sheet_$(date -u +%Y%m%dT%H%M%SZ).txt}"
NUM_PASSES="${NUM_PASSES:-2}"

DATASET="$REPO_ROOT/datasets/join_input_1x1_2power_32.txt"

mkdir -p -- "$(dirname -- "$OUT_FILE")"
: >>"$OUT_FILE"

if [[ ! -d "$ONOFF_DIR" ]]; then
  echo "Error: missing directory: $ONOFF_DIR" >&2
  exit 1
fi

if [[ ! -f "$DATASET" ]]; then
  echo "Error: dataset not found: $DATASET" >&2
  exit 1
fi
DATASET_ABS="$(realpath -m -- "$DATASET")"

mkdir -p "$BUILD_DIR"
if [[ "$(cd "$BUILD_DIR" && pwd)" != "$ONOFF_DIR/build" ]]; then
  echo "Error: refusing to use unexpected build dir: $BUILD_DIR" >&2
  exit 1
fi

# Sheet rows for Synthetic-1 2^32 — each executed once.
cases=(
  "13 128"
  "13 256"
  "13 512"
  "14 128"
  "14 256"
  "14 512"
)

for case in "${cases[@]}"; do
  read -r r b <<<"$case"

  rm -rf "$BUILD_DIR"/*
  cmake -S "$ONOFF_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNUM_PASSES="$NUM_PASSES" \
    -DNUM_RADIX_BITS="$r" \
    -DBINS_PER_PART="$b" >/dev/null
  cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null

  (cd "$BUILD_DIR" && ./OblRadix "$THREADS" "$DATASET_ABS") >>"$OUT_FILE" 2>&1
done

