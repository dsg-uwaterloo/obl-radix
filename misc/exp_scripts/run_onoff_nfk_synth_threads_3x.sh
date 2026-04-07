#!/usr/bin/env bash
set -euo pipefail

# Runs OnOff-NFK/OblRadix on selected synthetic datasets:
#   join_input_1x1_2power_{26,28,30}.txt
# using the (r,b) parameters from the sheet, and sweeps thread counts:
#   2, 4, 8, 16, 64
# Each thread count is executed 3 times.
#
# Output policy:
# - The output file contains ONLY stdout/stderr produced by ./OblRadix.
# - Build/configure output is suppressed (but failures still stop the script).
#
# Usage:
#   ./run_onoff_nfk_synth_threads_3x.sh [output_file]
#
# Example:
#   ./run_onoff_nfk_synth_threads_3x.sh runs/synth_threads_3x.txt
#
# Optional env vars:
#   NUM_PASSES=2   (default: 2)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
ONOFF_NFK_DIR="$REPO_ROOT/OnOff-NFK"
BUILD_DIR="$ONOFF_NFK_DIR/build"

OUT_FILE="${1:-$REPO_ROOT/runs/synth_threads_3x_$(date -u +%Y%m%dT%H%M%SZ).txt}"
NUM_PASSES="${NUM_PASSES:-2}"

mkdir -p -- "$(dirname -- "$OUT_FILE")"
: >>"$OUT_FILE"

if [[ ! -d "$ONOFF_NFK_DIR" ]]; then
  echo "Error: missing directory: $ONOFF_NFK_DIR" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
if [[ "$(cd "$BUILD_DIR" && pwd)" != "$ONOFF_NFK_DIR/build" ]]; then
  echo "Error: refusing to use unexpected build dir: $BUILD_DIR" >&2
  exit 1
fi

thread_counts=(2 4 8 16 64)

datasets=(
  "Synthetic-1_2p26"
  "Synthetic-1_2p28"
  "Synthetic-1_2p30"
)

declare -A dataset_path=(
  ["Synthetic-1_2p26"]="$REPO_ROOT/datasets/join_input_1x1_2power_26.txt"
  ["Synthetic-1_2p28"]="$REPO_ROOT/datasets/join_input_1x1_2power_28.txt"
  ["Synthetic-1_2p30"]="$REPO_ROOT/datasets/join_input_1x1_2power_30.txt"
)

# (r,b) from the sheet.
declare -A dataset_r=(
  ["Synthetic-1_2p26"]=10
  ["Synthetic-1_2p28"]=12
  ["Synthetic-1_2p30"]=13
)

declare -A dataset_b=(
  ["Synthetic-1_2p26"]=128 #256
  ["Synthetic-1_2p28"]=128
  ["Synthetic-1_2p30"]=256
)

for name in "${datasets[@]}"; do
  input="${dataset_path[$name]}"
  if [[ ! -f "$input" ]]; then
    echo "Error: dataset not found: $input" >&2
    exit 1
  fi
  input_abs="$(realpath -m -- "$input")"

  r="${dataset_r[$name]}"
  b="${dataset_b[$name]}"

  rm -rf "$BUILD_DIR"/*
  cmake -S "$ONOFF_NFK_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNUM_PASSES="$NUM_PASSES" \
    -DNUM_RADIX_BITS="$r" \
    -DBINS_PER_PART="$b" >/dev/null
  cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null

  pushd "$BUILD_DIR" >/dev/null
  for t in "${thread_counts[@]}"; do
    for _ in 1 2 3; do
      ./OblRadix "$t" "$input_abs" >>"$OUT_FILE" 2>&1
    done
  done
  popd >/dev/null
done

