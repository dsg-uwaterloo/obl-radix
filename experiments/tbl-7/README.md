# Key-distribution sensitivity

Table 7 reports execution time across key distributions on a synthetic dataset with **N = 2^25**. In the synthetic data creation scripts, that corresponds to generating files with `2power_26` (each table has `2^(26-1)=2^25` rows).

## Datasets (synthetic)

Generate (or reuse) the following inputs under `datasets/synthetic/`:

### Uniform (1:1)

```bash
cd datasets/synthetic
python3 create_synthetic_data_1x1.py
```

Use: `datasets/synthetic/join_input_1x1_2power_26.txt`

### Single Hot Key (1:N)

```bash
cd datasets/synthetic
python3 create_synthetic_data_1xn.py
```

Use: `datasets/synthetic/join_input_1xn_2power_26.txt`

### Zipf (s ≈ 1) (Many:N)

```bash
cd datasets/synthetic
python3 create_synthetic_data_power_law.py
```

Use: `datasets/synthetic/join_input_power_law_2power_26.txt`

## Radix

Build (tune `BINS_PER_PART` / `NUM_RADIX_BITS` / `NUM_PASSES` for your machine):

```bash
cd OnOff-NFK
mkdir -p build && cd build
cmake .. -DNUM_RADIX_BITS=<NUM_RADIX_BITS> -DBINS_PER_PART=<BINS_PER_PART> -DNUM_PASSES=<NUM_PASSES>
make -j$(nproc)
```

Run 3 trials per distribution:

```bash
cd OnOff-NFK/build

for trial in 1 2 3; do ./OblRadix 32 ../../datasets/synthetic/join_input_1x1_2power_26.txt; done
for trial in 1 2 3; do ./OblRadix 32 ../../datasets/synthetic/join_input_1xn_2power_26.txt; done
for trial in 1 2 3; do ./OblRadix 32 ../../datasets/synthetic/join_input_power_law_2power_26.txt; done
```
