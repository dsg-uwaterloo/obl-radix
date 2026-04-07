#!/usr/bin/env bash
set -euo pipefail

# Runs Twitter-3 once with 32 threads using:
# - baselines/obliviatorNFK-TDX/standalone_join
# - OnOff-NFK/OblRadix with NUM_RADIX_BITS=5 and BINS_PER_PART=4
#
# Output policy:
# - The output file contains ONLY stdout/stderr produced by the executables.
# - Build/configure output is suppressed (but failures still stop the script).
#
# Usage:
#   ./run_twitter3_once_baseline_nfk_vs_onoff_nfk.sh [output_file] [dataset_path]
#
# Example:
#   ./run_twitter3_once_baseline_nfk_vs_onoff_nfk.sh runs/twitter3_once_compare.txt datasets/real/twitter/twitter_3.txt
#
# Optional env vars:
#   NUM_PASSES=2   (default: 2)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
THREADS=32

OUT_FILE="${1:-$REPO_ROOT/runs/twitter3_once_baseline_nfk_vs_onoff_nfk_$(date -u +%Y%m%dT%H%M%SZ).txt}"
DATASET_ARG="${2:-$REPO_ROOT/datasets/real/twitter/twitter_3.txt}"
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

# ---------------- Baseline NFK ----------------
# BASELINE_DIR="$REPO_ROOT/baselines/obliviatorNFK-TDX"
# BASELINE_BIN="$BASELINE_DIR/standalone_join"

# if [[ ! -d "$BASELINE_DIR" ]]; then
#   echo "Error: missing directory: $BASELINE_DIR" >&2
#   exit 1
# fi

# if [[ ! -x "$BASELINE_BIN" ]]; then
#   (cd "$BASELINE_DIR" && make -f Makefile.standalone clean >/dev/null && make -f Makefile.standalone >/dev/null)
# fi

# (cd "$BASELINE_DIR" && ./standalone_join "$THREADS" "$DATASET") >>"$OUT_FILE" 2>&1

# ---------------- OnOff-NFK ----------------
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

rm -rf "$BUILD_DIR"/*
cmake -S "$ONOFF_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNUM_PASSES="$NUM_PASSES" \
  -DNUM_RADIX_BITS=5 \
  -DBINS_PER_PART=4 >/dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null

(cd "$BUILD_DIR" && ./OblRadix "$THREADS" "$DATASET") >>"$OUT_FILE" 2>&1

