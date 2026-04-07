#!/usr/bin/env bash
set -euo pipefail

# Runs OnOff-NFK/OblRadix on the twitter-2 dataset over a parameter grid:
#   NUM_RADIX_BITS in {8,10,12}
#   BINS_PER_PART  in {2^6,2^8,2^10} = {64,256,1024}
#
# Usage:
#   ./run_onoff_nfk_twitter2_sweep.sh [threads] [dataset_path]
#
# Examples:
#   ./run_onoff_nfk_twitter2_sweep.sh
#   ./run_onoff_nfk_twitter2_sweep.sh 32 ../../datasets/real/twitter/twitter_2.txt

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
ONOFF_NFK_DIR="$REPO_ROOT/OnOff-NFK"
BUILD_DIR="$ONOFF_NFK_DIR/build"

THREADS="${1:-32}"
DATASET_ARG="${2:-$REPO_ROOT/datasets/real/twitter/twitter_2.txt}"
NUM_PASSES="${NUM_PASSES:-2}"

RADIX_BITS_LIST=(8 10 12)
BINS_EXP_LIST=(6 8 10)

if [[ ! -d "$ONOFF_NFK_DIR" ]]; then
  echo "Error: missing directory: $ONOFF_NFK_DIR" >&2
  exit 1
fi

DATASET=""
if [[ -f "$DATASET_ARG" ]]; then
  DATASET="$(realpath -m -- "$DATASET_ARG")"
elif [[ -f "$REPO_ROOT/$DATASET_ARG" ]]; then
  # Allow passing repo-relative paths like "datasets/real/twitter/twitter_2.txt".
  DATASET="$(realpath -m -- "$REPO_ROOT/$DATASET_ARG")"
else
  echo "Error: dataset not found: $DATASET_ARG" >&2
  echo "Hint: from the repo root, use: datasets/real/twitter/twitter_2.txt" >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/runs"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="$REPO_ROOT/runs/onoff-nfk_twitter2_${RUN_ID}"
mkdir -p "$OUT_ROOT"

OUT_FILE="$OUT_ROOT/output.txt"
touch "$OUT_FILE"

mkdir -p "$BUILD_DIR"
if [[ "$(cd "$BUILD_DIR" && pwd)" != "$ONOFF_NFK_DIR/build" ]]; then
  echo "Error: refusing to use unexpected build dir: $BUILD_DIR" >&2
  exit 1
fi

for RADIX_BITS in "${RADIX_BITS_LIST[@]}"; do
  for BEXP in "${BINS_EXP_LIST[@]}"; do
    # Skip (r=12, b=2^6) as requested.
    if [[ "$RADIX_BITS" == "12" && "$BEXP" == "6" ]]; then
      continue
    fi

    BINS_PER_PART=$((1 << BEXP))
    TAG="r${RADIX_BITS}_bins2p${BEXP}"

    # Clean in-between configs to avoid stale CMake cache/options.
    rm -rf "$BUILD_DIR"/*

    cmake -S "$ONOFF_NFK_DIR" -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DNUM_PASSES="$NUM_PASSES" \
      -DNUM_RADIX_BITS="$RADIX_BITS" \
      -DBINS_PER_PART="$BINS_PER_PART" >/dev/null

    cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null

    pushd "$BUILD_DIR" >/dev/null
      # Append ONLY stdout/stderr from OblRadix to the output file.
      for run in 1 2 3; do
        ./OblRadix "$THREADS" "$DATASET" 2>&1 | tee -a "$OUT_FILE"
        if [[ -f join.txt ]]; then
          mv -f join.txt "$OUT_ROOT/${TAG}.run${run}.join.txt"
        fi
      done
    popd >/dev/null
  done
done

printf '%s\n' "$OUT_FILE"
