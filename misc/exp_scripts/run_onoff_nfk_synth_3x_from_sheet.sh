#!/usr/bin/env bash
set -euo pipefail

# Runs OnOff-NFK/OblRadix on synthetic datasets using the (r, b) parameters
# from the provided sheet image. Each dataset is executed 3 times.
#
# Output policy:
# - The output file contains ONLY stdout/stderr produced by ./OblRadix.
# - Build/configure output is suppressed (but failures still stop the script).
#
# Usage:
#   ./run_onoff_nfk_synth_3x_from_sheet.sh [threads] [output_file]
#
# Examples:
#   ./run_onoff_nfk_synth_3x_from_sheet.sh
#   ./run_onoff_nfk_synth_3x_from_sheet.sh 32 runs/synth_3x.txt
#
# Optional env vars:
#   NUM_PASSES=2   (default: 2)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
ONOFF_NFK_DIR="$REPO_ROOT/OnOff-NFK"
BUILD_DIR="$ONOFF_NFK_DIR/build"

THREADS="${1:-32}"
OUT_FILE="${2:-$REPO_ROOT/runs/synth_3x_$(date -u +%Y%m%dT%H%M%SZ).txt}"
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

# Synthetic datasets and (r,b) taken from the sheet.
datasets=(
  "Synthetic-1_2p22"
  "Synthetic-1_2p24"
  "Synthetic-1_2p26"
  "Synthetic-1_2p28"
  "Synthetic-1_2p30"
)

declare -A dataset_path=(
  ["Synthetic-1_2p22"]="$REPO_ROOT/datasets/join_input_1x1_2power_22.txt"
  ["Synthetic-1_2p24"]="$REPO_ROOT/datasets/join_input_1x1_2power_24.txt"
  ["Synthetic-1_2p26"]="$REPO_ROOT/datasets/join_input_1x1_2power_26.txt"
  ["Synthetic-1_2p28"]="$REPO_ROOT/datasets/join_input_1x1_2power_28.txt"
  ["Synthetic-1_2p30"]="$REPO_ROOT/datasets/join_input_1x1_2power_30.txt"
)

declare -A dataset_r=(
  ["Synthetic-1_2p22"]=10
  ["Synthetic-1_2p24"]=10
  ["Synthetic-1_2p26"]=10
  ["Synthetic-1_2p28"]=12
  ["Synthetic-1_2p30"]=13
)

declare -A dataset_b=(
  ["Synthetic-1_2p22"]=8 #16
  ["Synthetic-1_2p24"]=32 #64
  ["Synthetic-1_2p26"]=128 #256
  ["Synthetic-1_2p28"]=64 #128
  ["Synthetic-1_2p30"]=128 #256
)

for name in "${datasets[@]}"; do
  input="${dataset_path[$name]}"
  if [[ ! -f "$input" ]]; then
    echo "Error: dataset not found: $input" >&2
    echo "Hint: generate it or update dataset_path[] in this script." >&2
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
  for _ in 1 2 3; do
    ./OblRadix "$THREADS" "$input_abs" >>"$OUT_FILE" 2>&1
  done
  popd >/dev/null
done
