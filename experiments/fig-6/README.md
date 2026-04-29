# Online vs. Offline latency under varying radix and bin bits

## Dataset

- **Twitter-2**: Prepare using `datasets/real/twitter/` scripts, then set `TWITTER_2_PATH` to the respective produced join input file.

## Radix (Online + Offline Phases)

We evaluate different configurations of:
- **Radix bits (r)** ∈ {8, 10, 12}
- **Bin bits (b)** ∈ {6, 8, 10}

Run the following script:

```bash
#!/bin/bash

set -e

RADIX_BITS_LIST=(8 10 12)
BIN_BITS_LIST=(6 8 10)
THREADS=32
TRIALS=3

cd OnOff-NFK

for r in "${RADIX_BITS_LIST[@]}"; do
  for b in "${BIN_BITS_LIST[@]}"; do
    echo "========================================"
    echo "Running config: r=$r, b=$b"
    echo "========================================"

    rm -rf build
    mkdir build && cd build

    cmake .. \
      -DNUM_RADIX_BITS=$r \
      -DBINS_PER_PART=$((1 << b)) \
      -DNUM_PASSES=2 \
      -DONLINE_ONLY=OFF

    make -j$(nproc)

    for trial in $(seq 1 $TRIALS); do
      echo "[r=$r, b=$b] Trial $trial"
      ./OblRadix $THREADS "$TWITTER_2_PATH"
    done

    cd ..
  done
done
```
