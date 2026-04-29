# Real-world datasets evaluation

## Datasets (real)

- Jokes: `datasets/real/jokes/jokes.txt`
- Amazon: `datasets/real/amazon.txt`
- Slashdot: `datasets/real/slashdot.txt`
- Twitter-1: prepare via `datasets/real/twitter/` scripts, then set `TWITTER_1_PATH` to the respective produced join input file
- Twitter-2: prepare via `datasets/real/twitter/` scripts, then set `TWITTER_2_PATH` to the respective produced join input file

## Radix

Build (tune `BINS_PER_PART` / `NUM_RADIX_BITS` / `NUM_PASSES` for your machine):

```bash
cd OnOff-NFK
mkdir -p build && cd build
cmake .. -DNUM_RADIX_BITS=<NUM_RADIX_BITS> -DBINS_PER_PART=<BINS_PER_PART> -DNUM_PASSES=<NUM_PASSES>
make -j$(nproc)
```

Run 3 trials for each dataset:

```bash
cd OnOff-NFK/build
for trial in 1 2 3; do ./OblRadix 32 ../../datasets/real/jokes/jokes.txt; done
for trial in 1 2 3; do ./OblRadix 32 ../../datasets/real/amazon.txt; done
for trial in 1 2 3; do ./OblRadix 32 ../../datasets/real/slashdot.txt; done
for trial in 1 2 3; do ./OblRadix 32 "$TWITTER_1_PATH"; done
for trial in 1 2 3; do ./OblRadix 32 "$TWITTER_2_PATH"; done
```

## OBL-TDX

Build:

```bash
cd baselines/obliviatorNFK-TDX
make -f Makefile.standalone clean
make -f Makefile.standalone -j
```

Run 3 trials for each dataset:

```bash
cd baselines/obliviatorNFK-TDX
for trial in 1 2 3; do ./standalone_join 32 ../../datasets/real/jokes/jokes.txt; done
for trial in 1 2 3; do ./standalone_join 32 ../../datasets/real/amazon.txt; done
for trial in 1 2 3; do ./standalone_join 32 ../../datasets/real/slashdot.txt; done
for trial in 1 2 3; do ./standalone_join 32 "$TWITTER_1_PATH"; done
for trial in 1 2 3; do ./standalone_join 32 "$TWITTER_2_PATH"; done
```

## KKS-TDX

For KKS-TDX, we used the optimized version from Obliviator’s artifacts, and ran it 3 times on:
- Jokes
- Amazon
- Slashdot

Build and run (from `join_kks` in Obliviator's artifacts available at `https://zenodo.org/records/17299489`):

```bash
make prototype
./prototype INPUT_FILE OUTPUT_FILE
```
