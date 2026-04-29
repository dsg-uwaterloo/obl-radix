# Synthetic data scaling

## Datasets (synthetic)

Generate the 1x1 synthetic datasets using `datasets/synthetic/create_synthetic_data_1x1.py`:

```bash
cd datasets/synthetic
python3 create_synthetic_data_1x1.py
```

This generates the following 5 input files in `datasets/synthetic/`:

- `datasets/synthetic/join_input_1x1_2power_26.txt`
- `datasets/synthetic/join_input_1x1_2power_27.txt`
- `datasets/synthetic/join_input_1x1_2power_28.txt`
- `datasets/synthetic/join_input_1x1_2power_29.txt`
- `datasets/synthetic/join_input_1x1_2power_30.txt`


## Radix

Build (tune `BINS_PER_PART` / `NUM_RADIX_BITS` / `NUM_PASSES` for your machine):

```bash
cd OnOff-NFK
mkdir -p build && cd build
cmake .. -DNUM_RADIX_BITS=<NUM_RADIX_BITS> -DBINS_PER_PART=<BINS_PER_PART> -DNUM_PASSES=<NUM_PASSES>
make -j$(nproc)
```

Run 3 trials per dataset:

```bash
cd OnOff-NFK/build
for trial in 1 2 3; do ./OblRadix 32 ../../datasets/synthetic/join_input_1x1_2power_26.txt; done
for trial in 1 2 3; do ./OblRadix 32 ../../datasets/synthetic/join_input_1x1_2power_27.txt; done
for trial in 1 2 3; do ./OblRadix 32 ../../datasets/synthetic/join_input_1x1_2power_28.txt; done
for trial in 1 2 3; do ./OblRadix 32 ../../datasets/synthetic/join_input_1x1_2power_29.txt; done
for trial in 1 2 3; do ./OblRadix 32 ../../datasets/synthetic/join_input_1x1_2power_30.txt; done
```

## OBL-TDX

Build:

```bash
cd baselines/obliviatorNFK-TDX
make -f Makefile.standalone clean
make -f Makefile.standalone -j
```

Run 3 trials per dataset:

```bash
cd baselines/obliviatorNFK-TDX
for trial in 1 2 3; do ./standalone_join 32 ../../datasets/synthetic/join_input_1x1_2power_26.txt; done
for trial in 1 2 3; do ./standalone_join 32 ../../datasets/synthetic/join_input_1x1_2power_27.txt; done
for trial in 1 2 3; do ./standalone_join 32 ../../datasets/synthetic/join_input_1x1_2power_28.txt; done
for trial in 1 2 3; do ./standalone_join 32 ../../datasets/synthetic/join_input_1x1_2power_29.txt; done
for trial in 1 2 3; do ./standalone_join 32 ../../datasets/synthetic/join_input_1x1_2power_30.txt; done
```
