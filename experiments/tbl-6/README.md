# Foreign key join evaluation

## Datasets

### IMDb

Prepare:

```bash
cd datasets/real/imdb
bash download.sh
python3 imdb_process_1.py
python3 imdb_process_2.py
```
Use: `datasets/real/imdb/imdb.txt`

### TPC-H (sf 10 / sf 100)

1) Generate base tables using the official TPC-H data generator from `https://www.tpc.org/tpch/` (run `dbgen` with the desired scale factor).

2) Place the generated `lineitem.tbl` and `orders.tbl` in `datasets/TPC-H/` (or adjust paths accordingly).

3) Convert to join input format:

```bash
cd datasets/TPC-H
python3 tpch_fk.py
```

This script writes `tpch_fk.txt` in the current directory. Rename/copy it to keep both scale factors, then set:

- `TPCH_SF10_PATH`: path to the produced join input for scale factor 10
- `TPCH_SF100_PATH`: path to the produced join input for scale factor 100

## Radix

Build (tune `BINS_PER_PART` / `NUM_RADIX_BITS` / `NUM_PASSES` for your machine):

```bash
cd OnOff-FK
mkdir -p build && cd build
cmake .. -DNUM_RADIX_BITS=<NUM_RADIX_BITS> -DBINS_PER_PART=<BINS_PER_PART> -DNUM_PASSES=<NUM_PASSES>
make -j$(nproc)
```

Run 3 trials per dataset:

```bash
cd OnOff-FK/build

for trial in 1 2 3; do ./OblRadix 32 ../../datasets/real/imdb/imdb.txt; done
for trial in 1 2 3; do ./OblRadix 32 "$TPCH_SF10_PATH"; done
for trial in 1 2 3; do ./OblRadix 32 "$TPCH_SF100_PATH"; done
```

## OBL-TDX 

Build:

```bash
cd baselines/obliviatorFK-TDX
make -f Makefile.standalone clean
make -f Makefile.standalone -j
```

Run 3 trials per dataset:

```bash
cd baselines/obliviatorFK-TDX

for trial in 1 2 3; do ./standalone_join 32 ../../datasets/real/imdb/imdb.txt; done
for trial in 1 2 3; do ./standalone_join 32 "$TPCH_SF10_PATH"; done
for trial in 1 2 3; do ./standalone_join 32 "$TPCH_SF100_PATH"; done
```
