#!/usr/bin/env bash
set -euo pipefail

# Runs baselines/obliviatorNFK-TDX/standalone_join on real datasets with 32
# threads. Each dataset is executed 3 times.
#
# Output policy:
# - The output file contains ONLY stdout/stderr produced by ./standalone_join.
# - Build output is suppressed (but failures still stop the script).
#
# Usage:
#   ./run_baseline_obliviator_nfk_real_3x.sh [output_file]
#
# Example:
#   ./run_baseline_obliviator_nfk_real_3x.sh runs/baseline_nfk_real_3x.txt

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
BASELINE_DIR="$REPO_ROOT/baselines/obliviatorNFK-TDX"
BIN="$BASELINE_DIR/standalone_join"
THREADS=32

OUT_FILE="${1:-$REPO_ROOT/runs/baseline_nfk_real_3x_$(date -u +%Y%m%dT%H%M%SZ).txt}"
mkdir -p -- "$(dirname -- "$OUT_FILE")"
: >>"$OUT_FILE"

if [[ ! -d "$BASELINE_DIR" ]]; then
  echo "Error: missing directory: $BASELINE_DIR" >&2
  exit 1
fi

# Build if needed (suppressed output).
if [[ ! -x "$BIN" ]]; then
  (cd "$BASELINE_DIR" && make -f Makefile.standalone clean >/dev/null && make -f Makefile.standalone >/dev/null)
fi

datasets=(
  "Jokes"
  "Amazon"
  "Slashdot"
  "Twitter-1"
  "Twitter-2"
  "Twitter-3"
)

declare -A dataset_path=(
  ["Jokes"]="$REPO_ROOT/datasets/real/jokes/jokes.txt"
  ["Amazon"]="$REPO_ROOT/datasets/real/amazon.txt"
  ["Slashdot"]="$REPO_ROOT/datasets/real/slashdot.txt"
  ["Twitter-1"]="$REPO_ROOT/datasets/real/twitter/twitter_1.txt"
  ["Twitter-2"]="$REPO_ROOT/datasets/real/twitter/twitter_2.txt"
  ["Twitter-3"]="$REPO_ROOT/datasets/real/twitter/twitter_3.txt"
)

for name in "${datasets[@]}"; do
  input="${dataset_path[$name]}"
  if [[ ! -f "$input" ]]; then
    echo "Error: dataset not found: $input" >&2
    exit 1
  fi
  input_abs="$(realpath -m -- "$input")"

  for _ in 1 2 3; do
    (cd "$BASELINE_DIR" && ./standalone_join "$THREADS" "$input_abs") >>"$OUT_FILE" 2>&1
  done
done

