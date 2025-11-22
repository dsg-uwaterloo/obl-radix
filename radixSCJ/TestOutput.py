#!/usr/bin/env python3
"""
TestOutput.py - Compare join results between Python dataframes and C/C++ oblivious join implementation

This script:
1. Reads input files the same way as inputs.h
2. Creates dataframes and performs joins
3. Compares results with build/join.txt (or the path you specify)
4. Accounts for different ordering in results
"""

import sys
import argparse
import pandas as pd
import re
from pathlib import Path


def is_blank_line(line):
    """Check if line is empty or contains only whitespace"""
    return not line.strip()


def load_two_tables(input_path, data_length):
    """
    Load two tables from input file, mimicking the C++ inputs.h logic
    
    Format:
    - First non-empty line: n0 n1
    - Next n0 + n1 lines: <key> <payload>
    
    Returns: (table0_df, table1_df)
    """
    try:
        with open(input_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: cannot open \"{input_path}\"")
        return None, None
    
    # Parse header (n0 n1)
    header_found = False
    n0, n1 = 0, 0
    line_idx = 0
    
    for line_idx, line in enumerate(lines):
        if is_blank_line(line):
            continue
        try:
            parts = line.strip().split()
            if len(parts) >= 2:
                n0, n1 = int(parts[0]), int(parts[1])
                header_found = True
                break
        except ValueError:
            print(f"Error: malformed header: \"{line.strip()}\"")
            return None, None
    
    if not header_found:
        print("Error: no valid header found")
        return None, None
        
    # Parse data records
    table0_records = []
    table1_records = []
    records_read = 0
    
    for line in lines[line_idx + 1:]:
        if is_blank_line(line):
            continue
            
        # Parse key and payload
        line = line.rstrip('\n\r')
        parts = line.split(' ', 1)  # Split into at most 2 parts
        
        if len(parts) < 1:
            print(f"Error parsing line: \"{line}\"")
            return None, None
            
        try:
            key = int(parts[0])
        except ValueError:
            print(f"Error parsing key in line: \"{line}\"")
            return None, None
        
        # Extract payload (rest of line after key)
        payload = parts[1] if len(parts) > 1 else ""
        
        # Truncate payload to data_length - 1 and ensure null termination
        if len(payload) >= data_length:
            payload = payload[:data_length-1]
        
        # Determine which table this record belongs to
        if records_read < n0:
            table0_records.append({
                'key': key,
                'payload': payload,
                'idx': records_read
            })
        elif records_read < n0 + n1:
            table1_records.append({
                'key': key,
                'payload': payload,
                'idx': records_read - n0
            })
        else:
            break
            
        records_read += 1
    
    if records_read < n0 + n1:
        print(f"Error: only read {records_read} of {n0 + n1} requested records")
        return None, None
    
    # Create DataFrames
    table0_df = pd.DataFrame(table0_records)
    table1_df = pd.DataFrame(table1_records)
    
    print(f"Table 0: {len(table0_df)} records")
    print(f"Table 1: {len(table1_df)} records")
    
    return table0_df, table1_df


def perform_dataframe_join(table0_df, table1_df):
    """
    Perform inner join on key column

    Assumes table0_df is the primary key table
    """
    # Perform inner join on 'key' column
    join_result = pd.merge(table0_df, table1_df, on='key', suffixes=('_R', '_S'))
    
    # Rename columns to match C/C++ oblivious join output format
    join_result = join_result.rename(columns={
        'key': 'keyR',
        'payload_R': 'payR',
        'payload_S': 'payS',
        'idx_R': 'idxR',
        'idx_S': 'idxS'
    })
    
    # Add keyS column (same as keyR since we joined on key)
    join_result['keyS'] = join_result['keyR']
    
    # Reorder columns to match expected format
    join_result = join_result[['keyR', 'keyS', 'payR', 'payS', 'idxR', 'idxS']]
    
    print(f"Join result: {len(join_result)} records")
    return join_result


def parse_cpp_join_output(join_txt_path):
    """
    Parse the oblivious join output from join.txt
    Format: keyR payR keyS payS
    """
    try:
        with open(join_txt_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: cannot open \"{join_txt_path}\"")
        return None
    
    cpp_results = []
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(' ', 3)  # Split into at most 4 parts
        if len(parts) < 3:
            print(f"Warning: malformed line {line_num}: \"{line}\"")
            continue
        
        try:
            keyR = int(parts[0])
            payR = parts[1]
            keyS = int(parts[2])
            payS = parts[3] if len(parts) > 3 else ""
            
            cpp_results.append({
                'keyR': keyR,
                'keyS': keyS,
                'payR': payR,
                'payS': payS
            })
            
        except ValueError as e:
            print(f"Error parsing line {line_num}: \"{line}\" - {e}")
            continue
    
    cpp_df = pd.DataFrame(cpp_results)
    print(f"Oblivious join results: {len(cpp_df)} records")
    return cpp_df


def compare_join_results(python_df, cpp_df):
    """
    Compare Python dataframe join results with oblivious join results
    Returns True if they match, False otherwise
    """
    print("\n=== COMPARISON ===")

    # Check result size
    if len(python_df) != len(cpp_df):
        print(f"❌ FAILURE: Result size mismatch")
        return False
    
    # Select only the columns that matter for comparison
    python_compare = python_df[['keyR', 'keyS', 'payR', 'payS']].copy()
    cpp_compare = cpp_df[['keyR', 'keyS', 'payR', 'payS']].copy()
    
    
    # Convert dataframes to sets of tuples for set operations
    python_records = set(python_compare.itertuples(index=False, name=None))
    cpp_records = set(cpp_compare.itertuples(index=False, name=None))
    
    # Find records that are in Python results but not in oblivious join results
    python_not_in_cpp = python_records - cpp_records
    
    # Find records that are in oblivious join results but not in Python results  
    cpp_not_in_python = cpp_records - python_records
    
    # Check for exact match
    if len(python_not_in_cpp) == 0 and len(cpp_not_in_python) == 0:
        print("✅ SUCCESS: All join results match!")
        return True
    
    print(f"❌ FAILURE: Results do not match")
    
    # Show Python records not found in oblivious join output
    if python_not_in_cpp:
        print(f"\n🔍 Python records NOT FOUND in oblivious join output ({len(python_not_in_cpp)} records):")
        count = 0
        for record in sorted(python_not_in_cpp):
            if count >= 10:  # Limit output to first 10
                print(f"   ... and {len(python_not_in_cpp) - 10} more records")
                break
            keyR, keyS, payR, payS = record
            print(f"   {keyR} {payR} {keyS} {payS}")
            count += 1
    
    # Show oblivious join results not found in Python output
    if cpp_not_in_python:
        print(f"\n🔍 Oblivious join results NOT FOUND in Python output ({len(cpp_not_in_python)} records):")
        count = 0
        for record in sorted(cpp_not_in_python):
            if count >= 10:  # Limit output to first 10
                print(f"   ... and {len(cpp_not_in_python) - 10} more records")
                break
            keyR, keyS, payR, payS = record
            print(f"   {keyR} {payR} {keyS} {payS}")
            count += 1
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare Python dataframe join with C/C++ oblivious join results"
    )
    parser.add_argument("input_file", help="Path to input file")
    parser.add_argument("join_output", nargs='?', default="build/join.txt", 
                        help="Path to C/C++ oblivious join output file (default: build/join.txt)")
    
    args = parser.parse_args()
    data_length = 8
    
    print(f"Input file: {args.input_file}")
    print(f"Oblivious join output: {args.join_output}")
    print()
    
    # Load input tables
    table0_df, table1_df = load_two_tables(args.input_file, data_length)
    if table0_df is None or table1_df is None:
        print("Failed to load input tables")
        return 1
    
    # Perform Python dataframe join
    print("\n=== PYTHON JOIN ===")
    python_join_df = perform_dataframe_join(table0_df, table1_df)
    
    # Load oblivious join results
    print("\n=== OBLIVIOUS JOIN RESULTS ===")
    cpp_join_df = parse_cpp_join_output(args.join_output)
    if cpp_join_df is None:
        print("Failed to load oblivious join results")
        return 1
    
    # Compare results
    success = compare_join_results(python_join_df, cpp_join_df)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 
    