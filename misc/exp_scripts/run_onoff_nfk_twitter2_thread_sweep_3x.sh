#!/usr/bin/env bash
set -euo pipefail

# Runs OnOff-NFK/OblRadix on Twitter-2 with fixed parameters:
#   NUM_RADIX_BITS=12, BINS_PER_PART=64
# Sweeps thread counts: 2, 4, 16, 32, 64
# Each thread count is executed 3 times.
#
# Output policy:
# - The output file contains ONLY stdout/stderr produced by ./OblRadix.
# - Build/configure output is suppressed (but failures still stop the script).
#
# Usage:
#   ./run_onoff_nfk_twitter2_thread_sweep_3x.sh [output_file] [dataset_path]
#
# Examples:
#   ./run_onoff_nfk_twitter2_thread_sweep_3x.sh
#   ./run_onoff_nfk_twitter2_thread_sweep_3x.sh runs/twitter2_threads_3x.txt
#
# Optional env vars:
#   NUM_PASSES=2   (default: 2)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
ONOFF_NFK_DIR="$REPO_ROOT/OnOff-NFK"
BUILD_DIR="$ONOFF_NFK_DIR/build"

OUT_FILE="${1:-$REPO_ROOT/runs/twitter2_threads_3x_$(date -u +%Y%m%dT%H%M%SZ).txt}"
DATASET_ARG="${2:-$REPO_ROOT/datasets/real/twitter/twitter_2.txt}"
NUM_PASSES="${NUM_PASSES:-2}"

mkdir -p -- "$(dirname -- "$OUT_FILE")"
: >>"$OUT_FILE"

DATASET=""
if [[ -f "$DATASET_ARG" ]]; then
  DATASET="$(realpath -m -- "$DATASET_ARG")"
elif [[ -f "$REPO_ROOT/$DATASET_ARG" ]]; then
  DATASET="$(realpath -m -- "$REPO_ROOT/$DATASET_ARG")"
else
  echo "Error: dataset not found: $DATASET_ARG" >&2
  exit 1
fi

if [[ ! -d "$ONOFF_NFK_DIR" ]]; then
  echo "Error: missing directory: $ONOFF_NFK_DIR" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
if [[ "$(cd "$BUILD_DIR" && pwd)" != "$ONOFF_NFK_DIR/build" ]]; then
  echo "Error: refusing to use unexpected build dir: $BUILD_DIR" >&2
  exit 1
fi

# Fixed params from the sheet.
R=12
B=64

rm -rf "$BUILD_DIR"/*
cmake -S "$ONOFF_NFK_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNUM_PASSES="$NUM_PASSES" \
  -DNUM_RADIX_BITS="$R" \
  -DBINS_PER_PART="$B" >/dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null

# thread_counts=(2 4 16 32 64)
thread_counts=(8)

pushd "$BUILD_DIR" >/dev/null
for t in "${thread_counts[@]}"; do
  for _ in 1 2 3; do
    ./OblRadix "$t" "$DATASET" >>"$OUT_FILE" 2>&1
  done
done
popd >/dev/null

