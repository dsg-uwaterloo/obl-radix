# Thread scaling on real-world dataset

## Dataset

- Twitter-2: prepare via `datasets/real/twitter/` scripts, then set `TWITTER_2_PATH` to the respective produced join input file.

## Radix

Build (tune `BINS_PER_PART` / `NUM_RADIX_BITS` / `NUM_PASSES` for your machine):

```bash
cd OnOff-NFK
mkdir -p build && cd build
cmake .. -DNUM_RADIX_BITS=<NUM_RADIX_BITS> -DBINS_PER_PART=<BINS_PER_PART> -DNUM_PASSES=<NUM_PASSES>
make -j$(nproc)
```

Run 3 trials for each thread count:

```bash
cd OnOff-NFK/build
for t in 2 4 8 16 32; do
  for trial in 1 2 3; do
    ./OblRadix "$t" "$TWITTER_2_PATH"
  done
done
```

## OBL-TDX

Build:

```bash
cd baselines/obliviatorNFK-TDX
make -f Makefile.standalone clean
make -f Makefile.standalone -j
```

Run 3 trials for each thread count:

```bash
cd baselines/obliviatorNFK-TDX
for t in 2 4 8 16 32; do
  for trial in 1 2 3; do
    ./standalone_join "$t" "$TWITTER_2_PATH"
  done
done
```
