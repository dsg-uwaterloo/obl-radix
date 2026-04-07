#!/usr/bin/env bash
set -euo pipefail

# Runs OnOff-NFK/OblRadix on a fixed set of real datasets using the (r, b)
# parameters from the provided sheet image. Each dataset is executed 3 times.
#
# Output policy:
# - The output file contains ONLY stdout/stderr produced by ./OblRadix.
# - Build/configure output is suppressed (but failures still stop the script).
#
# Usage:
#   ./run_onoff_nfk_real_3x_from_sheet.sh [threads] [output_file]
#
# Examples:
#   ./run_onoff_nfk_real_3x_from_sheet.sh
#   ./run_onoff_nfk_real_3x_from_sheet.sh 32 runs/real_3x.txt
#
# Optional env vars:
#   NUM_PASSES=2   (default: 2)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
ONOFF_NFK_DIR="$REPO_ROOT/OnOff-NFK"
BUILD_DIR="$ONOFF_NFK_DIR/build"

THREADS="${1:-32}"
OUT_FILE="${2:-$REPO_ROOT/runs/real_3x_$(date -u +%Y%m%dT%H%M%SZ).txt}"
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

datasets=(
  "Amazon"
  "Jokes"
  "Slashdot"
  "Twitter-2"
  "Twitter-1"
)

declare -A dataset_path=(
  ["Amazon"]="$REPO_ROOT/datasets/real/amazon.txt"
  ["Jokes"]="$REPO_ROOT/datasets/real/jokes/jokes.txt"
  ["Slashdot"]="$REPO_ROOT/datasets/real/slashdot.txt"
  ["Twitter-2"]="$REPO_ROOT/datasets/real/twitter/twitter_2.txt"
  ["Twitter-1"]="$REPO_ROOT/datasets/real/twitter/twitter_1.txt"
)

# r = NUM_RADIX_BITS, b = BINS_PER_PART (as shown in the sheet).
declare -A dataset_r=(
  ["Amazon"]=10
  ["Jokes"]=4
  ["Slashdot"]=5
  ["Twitter-2"]=12
  ["Twitter-1"]=10
)

declare -A dataset_b=(
  ["Amazon"]=8 #16
  ["Jokes"]=32 #64
  ["Slashdot"]=128 #256
  ["Twitter-2"]=32 #64
  ["Twitter-1"]=64 #128
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
  for _ in 1 2 3; do
    ./OblRadix "$THREADS" "$input_abs" >>"$OUT_FILE" 2>&1
  done
  popd >/dev/null
done

