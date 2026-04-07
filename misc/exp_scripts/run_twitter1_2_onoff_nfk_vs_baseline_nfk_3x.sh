#!/usr/bin/env bash
set -euo pipefail

# Runs Twitter-2 (r=12,b=64) and Twitter-1 (r=10,b=64) multiple times each using:
# - OnOff-NFK/OblRadix (reconfigured/rebuilt per dataset config)
# - baselines/obliviatorNFK-TDX/standalone_join
# All runs use 32 threads.
#
# Output policy:
# - The output file contains ONLY stdout/stderr produced by the executables.
# - Build/configure output is suppressed (but failures still stop the script).
#
# Usage:
#   ./run_twitter1_2_onoff_nfk_vs_baseline_nfk_3x.sh [output_file]
#
# Example:
#   RUNS=6 ./run_twitter1_2_onoff_nfk_vs_baseline_nfk_3x.sh runs/twitter12_compare_6x.txt
#
# Optional env vars:
#   RUNS=6         (default: 6)
#   NUM_PASSES=2   (default: 2)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

THREADS=32
RUNS="${RUNS:-6}"
OUT_FILE="${1:-$REPO_ROOT/runs/twitter1_2_onoff_nfk_vs_baseline_nfk_${RUNS}x_$(date -u +%Y%m%dT%H%M%SZ).txt}"
NUM_PASSES="${NUM_PASSES:-2}"

mkdir -p -- "$(dirname -- "$OUT_FILE")"
: >>"$OUT_FILE"

declare -A dataset_path=(
  ["Twitter-2"]="$REPO_ROOT/datasets/real/twitter/twitter_2.txt"
  ["Twitter-1"]="$REPO_ROOT/datasets/real/twitter/twitter_1.txt"
)

declare -A dataset_r=(
  ["Twitter-2"]=12
  ["Twitter-1"]=10
)

declare -A dataset_b=(
  ["Twitter-2"]=64
  ["Twitter-1"]=64
)

datasets=( "Twitter-2" "Twitter-1" )

# ---------------- Baseline NFK build ----------------
BASELINE_DIR="$REPO_ROOT/baselines/obliviatorNFK-TDX"
BASELINE_BIN="$BASELINE_DIR/standalone_join"

if [[ ! -d "$BASELINE_DIR" ]]; then
  echo "Error: missing directory: $BASELINE_DIR" >&2
  exit 1
fi

if [[ ! -x "$BASELINE_BIN" ]]; then
  (cd "$BASELINE_DIR" && make -f Makefile.standalone clean >/dev/null && make -f Makefile.standalone >/dev/null)
fi

# ---------------- OnOff-NFK build dir ----------------
ONOFF_DIR="$REPO_ROOT/OnOff-NFK"
BUILD_DIR="$ONOFF_DIR/build"

if [[ ! -d "$ONOFF_DIR" ]]; then
  echo "Error: missing directory: $ONOFF_DIR" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
if [[ "$(cd "$BUILD_DIR" && pwd)" != "$ONOFF_DIR/build" ]]; then
  echo "Error: refusing to use unexpected build dir: $BUILD_DIR" >&2
  exit 1
fi

pushd "$BASELINE_DIR" >/dev/null
for name in "${datasets[@]}"; do
  input="${dataset_path[$name]}"
  if [[ ! -f "$input" ]]; then
    echo "Error: dataset not found: $input" >&2
    exit 1
  fi
  input_abs="$(realpath -m -- "$input")"

  r="${dataset_r[$name]}"
  b="${dataset_b[$name]}"

  # OnOff-NFK: rebuild with dataset-specific (r,b), then run N times.
  find "$BUILD_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + >/dev/null 2>&1 || true
  cmake -S "$ONOFF_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNUM_PASSES="$NUM_PASSES" \
    -DNUM_RADIX_BITS="$r" \
    -DBINS_PER_PART="$b" >/dev/null
  cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null

  pushd "$BUILD_DIR" >/dev/null
  for ((run=1; run<=RUNS; run++)); do
    ./OblRadix "$THREADS" "$input_abs" >>"$OUT_FILE" 2>&1
  done
  popd >/dev/null

  # Baseline NFK: run N times.
  for ((run=1; run<=RUNS; run++)); do
    ./standalone_join "$THREADS" "$input_abs" >>"$OUT_FILE" 2>&1
  done
done
popd >/dev/null
