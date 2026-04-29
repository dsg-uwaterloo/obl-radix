# Thread scaling on synthetic datasets

## Datasets (synthetic)

Generate the 1x1 synthetic datasets using `datasets/synthetic/create_synthetic_data_1x1.py`:

```bash
cd datasets/synthetic
python3 create_synthetic_data_1x1.py
```

Use the following three dataset sizes:

- `datasets/synthetic/join_input_1x1_2power_26.txt`
- `datasets/synthetic/join_input_1x1_2power_28.txt`
- `datasets/synthetic/join_input_1x1_2power_30.txt`

## Radix

Build (tune `BINS_PER_PART` / `NUM_RADIX_BITS` / `NUM_PASSES` for your machine):

```bash
cd OnOff-NFK
mkdir -p build && cd build
cmake .. -DNUM_RADIX_BITS=<NUM_RADIX_BITS> -DBINS_PER_PART=<BINS_PER_PART> -DNUM_PASSES=<NUM_PASSES>
make -j$(nproc)
```

Run 3 trials for each thread count and dataset:

```bash
cd OnOff-NFK/build
for ds in \
  ../../datasets/synthetic/join_input_1x1_2power_26.txt \
  ../../datasets/synthetic/join_input_1x1_2power_28.txt \
  ../../datasets/synthetic/join_input_1x1_2power_30.txt
do
  for t in 2 4 8 16 32; do
    for trial in 1 2 3; do
      ./OblRadix "$t" "$ds"
    done
  done
done
```