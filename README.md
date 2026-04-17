## Repository Structure

- **`OnOff-FK/`** - Radix partitioning-based join for foreign key relationships
- **`OnOff-NFK/`** - Radix partitioning-based join for non-foreign key relationships
- **`baselines/obliviatorFK-TDX/`** - Obliviator foreign-key join 
- **`baselines/obliviatorNFK-TDX/`** - Obliviator non-foreign key join 

## Build Instructions

### Building Radix Partitioning-based Implementations

For both `OnOff-NFK` and `OnOff-FK`:

```bash
cd OnOff-NFK  # or OnOff-FK
mkdir build && cd build
cmake .. \
  -DBINS_PER_PART=<BINS_PER_PART> \
  -DNUM_RADIX_BITS=<NUM_RADIX_BITS> \
  -DNUM_PASSES=<NUM_PASSES>
make -j$(nproc)
```
- `BINS_PER_PART`: Number of bins per partition  
- `NUM_RADIX_BITS`: Number of radix bits (total partitions = 2^NUM_RADIX_BITS)  
- `NUM_PASSES`: Number of radix partitioning passes

This builds the `OblRadix` executable that can be run with the following command:
```bash
./OblRadix <num_threads> <input_file>
```

**Note**: The radix partitioning-based joins are hardware-conscious algorithms. Depending on your workload and hardware, you may need to adjust default configurations for optimal performance:

- **Radix parameters**: Modify `OnOff-NFK/external/radix_partition/CMakeLists.txt` (or `OnOff-FK/external/radix_partition/CMakeLists.txt`) to update:
  - `BINS_PER_PART` (default: 32)
  - `NUM_RADIX_BITS` (default: 10)  
  - `NUM_PASSES` (default: 2)

  These parameters can also be overridden at CMake configure time using
  `-D...`, as shown in the [build instructions](#building-radix-partitioning-based-implementations) above.

- **Cache parameters**: Modify `OnOff-NFK/external/radix_partition/prj_params.h` (or `OnOff-FK/external/radix_partition/prj_params.h`) to update:
  - `CACHE_LINE_SIZE` (default: 64)
  - `L1_CACHE_SIZE` (default: 49152) 
  - `L1_ASSOCIATIVITY` (default: 12)
 
### Building Baseline Implementations

For both `obliviatorNFK-TDX` and `obliviatorFK-TDX`:

```bash
cd baselines/obliviatorNFK-TDX  # or baselines/obliviatorFK-TDX
make -f Makefile.standalone clean
make -f Makefile.standalone
```

This builds the `standalone_join` executable that can be run with the following command:
```bash
./standalone_join <num_threads> <input_file>
```

## Testing and Validation

Our radix partitioning-based implementations include Python output validation scripts that compare the C/C++ implementation results against a reference pandas implementation to ensure correctness:

```bash
# Validate results using Python script
cd OnOff-NFK  # or OnOff-FK
# First build and run the program to generate output:
# ./OblRadix <num_threads> <input_file>
# Then validate the results:
cd ..
python3 TestOutput.py <input_file> [join_output_file (build/join.txt by default)]
```

## Datasets

The repository includes several datasets for evaluation:

### Available Datasets

- **`datasets/real/`**: Real-world datasets

- **`datasets/TPC-H/`**: Scripts to generate TPC-H based join workloads

- **`datasets/synthetic/`**: Script to generate synthetic datasets

### Input Format Requirements

All implementations expect input files with this format:
```
n0 n1

key1 payload1
key2 payload2
...
(n0 records for table 0)

key1 payload1
key2 payload2
...
(n1 records for table 1)
```

### Evaluating pre-sorted datasets

Both radix-paritioning based implementations include a `sort_tables.py` script to pre-sort datasets (if not already sorted):

```bash
cd OnOff-NFK  # or OnOff-FK
python3 sort_tables.py <input_file> <output_file>
```
