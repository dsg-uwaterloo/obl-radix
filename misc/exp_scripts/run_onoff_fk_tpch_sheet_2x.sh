#!/usr/bin/env bash
set -euo pipefail

# Runs OnOff-FK/OblRadix on TPC-H datasets using the (r,b) rows from the sheet.
# Each (dataset,r,b) case is executed twice.
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
#   ./run_onoff_fk_tpch_sheet_2x.sh [threads] [output_file]
#
# Example:
#   ./run_onoff_fk_tpch_sheet_2x.sh 32 runs/tpch_sheet_2x.txt
#
# Optional env vars:
#   NUM_PASSES=2   (default: 2)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
ONOFF_FK_DIR="$REPO_ROOT/OnOff-FK"
BUILD_DIR="$ONOFF_FK_DIR/build"

THREADS="${1:-32}"
OUT_FILE="${2:-$REPO_ROOT/runs/onoff_fk_tpch_sheet_2x_$(date -u +%Y%m%dT%H%M%SZ).txt}"
NUM_PASSES="${NUM_PASSES:-2}"

mkdir -p -- "$(dirname -- "$OUT_FILE")"
: >>"$OUT_FILE"

if [[ ! -d "$ONOFF_FK_DIR" ]]; then
  echo "Error: missing directory: $ONOFF_FK_DIR" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
if [[ "$(cd "$BUILD_DIR" && pwd)" != "$ONOFF_FK_DIR/build" ]]; then
  echo "Error: refusing to use unexpected build dir: $BUILD_DIR" >&2
  exit 1
fi

declare -A dataset_path=(
  ["TPCH_sf100"]="$REPO_ROOT/datasets/TPC-H/sf100/tpch_fk_swapped.txt"
  ["TPCH_sf10"]="$REPO_ROOT/datasets/TPC-H/sf10/tpch_fk_swapped.txt"
)

# Sheet rows (dataset, r, b) — each executed twice.
cases=(
  "TPCH_sf100 11 256"
  "TPCH_sf100 11 512"
  "TPCH_sf100 12 128"
  "TPCH_sf100 12 256"
  "TPCH_sf100 12 512"
  "TPCH_sf10 12 16"
  "TPCH_sf10 10 64"
  "TPCH_sf10 11 32"
)

for case in "${cases[@]}"; do
  read -r ds r b <<<"$case"
  input="${dataset_path[$ds]:-}"
  if [[ -z "$input" ]]; then
    echo "Error: unknown dataset key: $ds" >&2
    exit 1
  fi
  if [[ ! -f "$input" ]]; then
    echo "Error: dataset not found: $input" >&2
    exit 1
  fi
  input_abs="$(realpath -m -- "$input")"

  rm -rf "$BUILD_DIR"/*
  cmake -S "$ONOFF_FK_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNUM_PASSES="$NUM_PASSES" \
    -DNUM_RADIX_BITS="$r" \
    -DBINS_PER_PART="$b" >/dev/null
  cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null

  for _ in 1 2; do
    (cd "$BUILD_DIR" && ./OblRadix "$THREADS" "$input_abs") >>"$OUT_FILE" 2>&1
  done
done

