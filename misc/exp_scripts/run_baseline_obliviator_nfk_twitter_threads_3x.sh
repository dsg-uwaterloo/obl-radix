#!/usr/bin/env bash
set -euo pipefail

# Runs baselines/obliviatorNFK-TDX/standalone_join on Twitter-2
# sweeping thread counts: 2, 4, 8, 16, 64. Each thread count is executed 3
# times per dataset.
#
# Output policy:
# - The output file contains ONLY stdout/stderr produced by ./standalone_join.
# - Build output is suppressed (but failures still stop the script).
#
# Usage:
#   ./run_baseline_obliviator_nfk_twitter_threads_3x.sh [output_file]
#
# Example:
#   ./run_baseline_obliviator_nfk_twitter_threads_3x.sh runs/baseline_nfk_twitter_threads_3x.txt

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
BASELINE_DIR="$REPO_ROOT/baselines/obliviatorNFK-TDX"
BIN="$BASELINE_DIR/standalone_join"

OUT_FILE="${1:-$REPO_ROOT/runs/baseline_nfk_twitter_threads_3x_$(date -u +%Y%m%dT%H%M%SZ).txt}"
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
  "Twitter-2"
  # "Twitter-3"
)

declare -A dataset_path=(
  ["Twitter-2"]="$REPO_ROOT/datasets/real/twitter/twitter_2.txt"
  # ["Twitter-3"]="$REPO_ROOT/datasets/real/twitter/twitter_3.txt"
)

# thread_counts=(2 4 8 16 64)
thread_counts=(64)

for name in "${datasets[@]}"; do
  input="${dataset_path[$name]}"
  if [[ ! -f "$input" ]]; then
    echo "Error: dataset not found: $input" >&2
    exit 1
  fi
  input_abs="$(realpath -m -- "$input")"

  for t in "${thread_counts[@]}"; do
    for _ in 1 2 3; do
      (cd "$BASELINE_DIR" && ./standalone_join "$t" "$input_abs") >>"$OUT_FILE" 2>&1
    done
  done
done
