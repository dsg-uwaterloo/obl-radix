# ORAM-based oblivious join evaluation 

## ORAM baseline

Baseline code: `https://github.com/zhao-chang/Oblivious-Join-Unofficial`

Use the built-in 0.05x TPC-H tables under `source-code/data/tpch/`.

Build:

```bash
cd ORAMJoin/source-code
make clean && make
```

### Run baseline

Load data once, then run Query TE1 (binary equi-join) 3 times:

```bash
cd source-code
./scripts/run_oij_type1_load.sh 1 0.05x 1
for trial in 1 2 3; do
  ./scripts/run_oij_type1_smj_equi_join.sh localhost 1 0.05x 1 queries/tpch_binary_equi_q1.txt
done
```

## Radix (single-thread)

Dataset: Generate using `datasets/TPC-H/tpch_nfk_q1.py` (requires `customer_0.05x.tbl` and `supplier_0.05x.tbl` from the baseline)

Build (tune `BINS_PER_PART` / `NUM_RADIX_BITS` / `NUM_PASSES` for your machine):

```bash
cd OnOff-NFK
mkdir -p build && cd build
cmake .. -DNUM_RADIX_BITS=<NUM_RADIX_BITS> -DBINS_PER_PART=<BINS_PER_PART> -DNUM_PASSES=<NUM_PASSES>
make -j$(nproc)
```

Notes:
- For Appendix E, we benchmark Radix configured with a single thread. Use the serial driver (`main_serial.cpp`).
- `OnOff-NFK/CMakeLists.txt` builds only `main.cpp`, replace it with `main_serial.cpp` in line 37 for this experiment.

Run 3 trials (the serial driver takes only the input file path):

```bash
cd OnOff-NFK/build
for trial in 1 2 3; do ./OblRadix ../../datasets/TPC-H/tpch_nfk_q1.txt; done
```