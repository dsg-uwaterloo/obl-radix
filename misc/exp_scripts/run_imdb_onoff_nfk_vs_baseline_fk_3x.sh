#!/usr/bin/env bash
set -euo pipefail

# Runs:
# - OnOff-FK/OblRadix with NUM_RADIX_BITS=10 and BINS_PER_PART=32
# - baselines/obliviatorFK-TDX/standalone_join
# on the IMDb dataset with 32 threads. Each program is executed 3 times.
#
# Output policy:
# - The output file contains ONLY stdout/stderr produced by the executables.
# - Build/configure output is suppressed (but failures still stop the script).
#
# Usage:
#   ./run_imdb_onoff_nfk_vs_baseline_fk_3x.sh [output_file] [imdb_path]
#
# Example:
#   ./run_imdb_onoff_nfk_vs_baseline_fk_3x.sh runs/imdb_compare_3x.txt datasets/real/imdb/imdb.txt
#
# Optional env vars:
#   NUM_PASSES=2   (default: 2)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

THREADS=32
OUT_FILE="${1:-$REPO_ROOT/runs/imdb_onoff_fk_vs_baseline_fk_3x_$(date -u +%Y%m%dT%H%M%SZ).txt}"
IMDB_ARG="${2:-$REPO_ROOT/datasets/real/imdb/imdb.txt}"
NUM_PASSES="${NUM_PASSES:-2}"

mkdir -p -- "$(dirname -- "$OUT_FILE")"
: >>"$OUT_FILE"

IMDB=""
if [[ -f "$IMDB_ARG" ]]; then
  IMDB="$(realpath -m -- "$IMDB_ARG")"
elif [[ -f "$REPO_ROOT/$IMDB_ARG" ]]; then
  IMDB="$(realpath -m -- "$REPO_ROOT/$IMDB_ARG")"
else
  echo "Error: dataset not found: $IMDB_ARG" >&2
  exit 1
fi

# ---------------- OnOff-FK build ----------------
ONOFF_FK_DIR="$REPO_ROOT/OnOff-FK"
ONOFF_BUILD="$ONOFF_FK_DIR/build"

if [[ ! -d "$ONOFF_FK_DIR" ]]; then
  echo "Error: missing directory: $ONOFF_FK_DIR" >&2
  exit 1
fi

mkdir -p "$ONOFF_BUILD"
if [[ "$(cd "$ONOFF_BUILD" && pwd)" != "$ONOFF_FK_DIR/build" ]]; then
  echo "Error: refusing to use unexpected build dir: $ONOFF_BUILD" >&2
  exit 1
fi

rm -rf "$ONOFF_BUILD"/*
cmake -S "$ONOFF_FK_DIR" -B "$ONOFF_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNUM_PASSES="$NUM_PASSES" \
  -DNUM_RADIX_BITS=10 \
  -DBINS_PER_PART=32 >/dev/null
cmake --build "$ONOFF_BUILD" -j"$(nproc)" >/dev/null

# Run OnOff-FK 3 times (append only program output).
pushd "$ONOFF_BUILD" >/dev/null
for _ in 1 2 3; do
  ./OblRadix "$THREADS" "$IMDB" >>"$OUT_FILE" 2>&1
done
popd >/dev/null

# ---------------- Baseline FK build ----------------
BASE_FK_DIR="$REPO_ROOT/baselines/obliviatorFK-TDX"
BASE_FK_BIN="$BASE_FK_DIR/standalone_join"

if [[ ! -d "$BASE_FK_DIR" ]]; then
  echo "Error: missing directory: $BASE_FK_DIR" >&2
  exit 1
fi

if [[ ! -x "$BASE_FK_BIN" ]]; then
  (cd "$BASE_FK_DIR" && make -f Makefile.standalone clean >/dev/null && make -f Makefile.standalone >/dev/null)
fi

for _ in 1 2 3; do
  (cd "$BASE_FK_DIR" && ./standalone_join "$THREADS" "$IMDB") >>"$OUT_FILE" 2>&1
done
