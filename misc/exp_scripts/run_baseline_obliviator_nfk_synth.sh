#!/usr/bin/env bash
set -euo pipefail

# Runs baselines/obliviatorNFK-TDX/standalone_join on synthetic datasets:
#   join_input_1x1_2power_{22,24,26,28,30,32}.txt
# with 32 threads.
#
# Repetitions:
# - 2^22, 2^24, 2^26, 2^28: run 3 times
# - 2^30, 2^32: run 1 time
#
# Output policy:
# - The output file contains ONLY stdout/stderr produced by ./standalone_join.
# - Build output is suppressed (but failures still stop the script).
#
# Usage:
#   ./run_baseline_obliviator_nfk_synth.sh [output_file]
#
# Example:
#   ./run_baseline_obliviator_nfk_synth.sh runs/baseline_nfk_synth.txt

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
BASELINE_DIR="$REPO_ROOT/baselines/obliviatorNFK-TDX"
BIN="$BASELINE_DIR/standalone_join"
THREADS=32

OUT_FILE="${1:-$REPO_ROOT/runs/baseline_nfk_synth_$(date -u +%Y%m%dT%H%M%SZ).txt}"
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

#datasets=(22 24 26 28 30 32)
datasets=(32)

for exp in "${datasets[@]}"; do
  input="$REPO_ROOT/datasets/join_input_1x1_2power_${exp}.txt"
  if [[ ! -f "$input" ]]; then
    echo "Error: dataset not found: $input" >&2
    exit 1
  fi
  input_abs="$(realpath -m -- "$input")"

  reps=3
  if [[ "$exp" == "30" || "$exp" == "32" ]]; then
    reps=1
  fi

  for ((i=0; i<reps; i++)); do
    (cd "$BASELINE_DIR" && ./standalone_join "$THREADS" "$input_abs") >>"$OUT_FILE" 2>&1
  done
done

