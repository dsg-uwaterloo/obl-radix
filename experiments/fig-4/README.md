# Offline cost amortization

This experiment measures the total latency for running a sequence of repeated join queries, where Radix uses a **fixed offline preprocessing** phase that can be amortized across queries.

Twitter-1 and Twitter-2 share one input table. Therefore, computing the offline preprocessing once for both queries requires processing 3 distinct tables in total:
- both tables for Twitter-1 (2 tables)
- only the remaining (non-shared) table for Twitter-2 (1 additional table)

We run Twitter-1 and Twitter-2 12 times each to obtain the per-query online times and to construct totals for Q ∈ {1, 2, 3, 6, 12}.

## Dataset

Prepare Twitter datasets using the scripts under `datasets/real/twitter/`, then set:
- `TWITTER_1_PATH`: join input file for Twitter-1
- `TWITTER_2_PATH`: join input file for Twitter-2

## Radix (fixed offline preprocessing)

### 1) Offline for Twitter-1 (process both tables)

1) Replace `main.cpp` with `main_offline.cpp` at line 37 in `OnOff-NFK/CMakeLists.txt` to build the offline driver.

2) Ensure `OnOff-NFK/main_offline.cpp` has both flags enabled:
- `#define PROCESS_R`
- `#define PROCESS_S`

3) Build and run once to record the combined offline time for the two Twitter-1 tables.

```bash
cd OnOff-NFK
mkdir -p build && cd build
cmake .. -DNUM_RADIX_BITS=<NUM_RADIX_BITS> -DBINS_PER_PART=<BINS_PER_PART> -DNUM_PASSES=<NUM_PASSES>
make -j$(nproc)
./OblRadix "$TWITTER_1_PATH"
```

### 2) Offline for Twitter-2 (process only the remaining table)

1) Edit `OnOff-NFK/main_offline.cpp` and comment out `#define PROCESS_S` so that only the remaining table is processed.

2) Rebuild and run once to record the offline time for the additional table needed by Twitter-2:

```bash
cd OnOff-NFK/build
rm -rf *
cmake .. -DNUM_RADIX_BITS=<NUM_RADIX_BITS> -DBINS_PER_PART=<BINS_PER_PART> -DNUM_PASSES=<NUM_PASSES>
make -j$(nproc)
./OblRadix "$TWITTER_2_PATH"
```

## Radix (online)

1) Replace `main_offline.cpp` with `main.cpp` at line 37 in `OnOff-NFK/CMakeLists.txt` to restore the normal driver.

2) Rebuild and run 12 trials for each query input.

```bash
cd OnOff-NFK/build
rm -rf *
cmake .. -DNUM_RADIX_BITS=<NUM_RADIX_BITS> -DBINS_PER_PART=<BINS_PER_PART> -DNUM_PASSES=<NUM_PASSES>
make -j$(nproc)

for trial in $(seq 1 12); do
  ./OblRadix 32 "$TWITTER_1_PATH"
  ./OblRadix 32 "$TWITTER_2_PATH"
done
```
Note: Choose optimal radix parameters for your machine and keep them constant across the offline and online stages.

## OBL-TDX

Build:

```bash
cd baselines/obliviatorNFK-TDX
make -f Makefile.standalone clean
make -f Makefile.standalone -j
```

Run 12 trials for each query input:

```bash
cd baselines/obliviatorNFK-TDX
for trial in $(seq 1 12); do
  ./standalone_join 32 "$TWITTER_1_PATH"
  ./standalone_join 32 "$TWITTER_2_PATH"
done
```
