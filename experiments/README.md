# Experimental evaluation

This directory contains instructions to reproduce the paper's experimental evaluation.

## Dataset preparation

Before running any specific experiment, prepare all datasets:

- Some datasets in `datasets/` are already prepared and ready to use.
- Others need to be downloaded and processed using the scripts provided.
- For TPC-H, first generate the base tables using the official TPC-H data generator from `https://www.tpc.org/tpch/`, then run the processing scripts.

Any dataset can be sorted by key using `sort_tables.py` (`OnOff-NFK/sort_tables.py` or `OnOff-FK/sort_tables.py`). Some datasets already come pre-sorted.

## Experiments

Each subdirectory corresponds to a figure/table/appendix item in the paper and contain the exact command lines needed for reproducibility.

For the radix partitioning-based implementations (`OnOff-NFK` / `OnOff-FK`), use radix and cache parameters that are optimal for your hardware and workloads. See [Build Instructions](../README.md#build-instructions) for details.
