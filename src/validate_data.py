#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Validation Script for DeepLog Training Data

This script validates the training data files to detect:
- NaN values
- Inf values
- Abnormal values (extremely large or small)
- Data statistics (min, max, mean, std)

Usage:
    python validate_data.py --data_dir ../data/processed/Integrated/T1105
"""

import argparse
import numpy as np
from pathlib import Path
import sys


def validate_line(line_num, line, stats):
    """
    Validate a single line of data.
    
    Args:
        line_num: Line number in the file
        line: Line content
        stats: Dictionary to accumulate statistics
        
    Returns:
        True if line is valid, False otherwise
    """
    values = line.strip().split()
    
    if not values:
        return True  # Empty line is okay
    
    has_error = False
    
    for val_idx, val_pair in enumerate(values):
        parts = val_pair.split(',')
        
        for part_idx, part in enumerate(parts):
            try:
                val = float(part)
                
                # Update statistics
                stats['total_values'] += 1
                stats['all_values'].append(val)
                
                # Check for NaN
                if np.isnan(val):
                    print(f"❌ Line {line_num}, Value {val_idx}, Part {part_idx}: NaN detected")
                    stats['nan_count'] += 1
                    has_error = True
                
                # Check for Inf
                elif np.isinf(val):
                    print(f"❌ Line {line_num}, Value {val_idx}, Part {part_idx}: Inf detected ({val})")
                    stats['inf_count'] += 1
                    has_error = True
                
                # Check for extremely large values
                elif abs(val) > 1e6:
                    print(f"⚠️  Line {line_num}, Value {val_idx}, Part {part_idx}: "
                          f"Extremely large value ({val})")
                    stats['large_value_count'] += 1
                
                # Update min/max
                if val < stats['min_value']:
                    stats['min_value'] = val
                if val > stats['max_value']:
                    stats['max_value'] = val
                    
            except ValueError:
                print(f"❌ Line {line_num}, Value {val_idx}, Part {part_idx}: "
                      f"Cannot convert to float: '{part}'")
                stats['parse_error_count'] += 1
                has_error = True
    
    return not has_error


def validate_file(filepath):
    """
    Validate a single data file.
    
    Args:
        filepath: Path to the file to validate
        
    Returns:
        Dictionary containing validation results and statistics
    """
    print(f"\n{'='*80}")
    print(f"Validating file: {filepath}")
    print(f"{'='*80}\n")
    
    stats = {
        'total_lines': 0,
        'valid_lines': 0,
        'invalid_lines': 0,
        'total_values': 0,
        'nan_count': 0,
        'inf_count': 0,
        'large_value_count': 0,
        'parse_error_count': 0,
        'min_value': float('inf'),
        'max_value': float('-inf'),
        'all_values': []
    }
    
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                stats['total_lines'] += 1
                
                if validate_line(line_num, line, stats):
                    stats['valid_lines'] += 1
                else:
                    stats['invalid_lines'] += 1
                
                # Progress indicator
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} lines...", end='\r')
        
        print(f"Processed {stats['total_lines']} lines... Done!")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None
    
    return stats


def print_statistics(stats, filename):
    """
    Print validation statistics.
    
    Args:
        stats: Statistics dictionary
        filename: Name of the file
    """
    print(f"\n{'-'*80}")
    print(f"Validation Results for: {filename}")
    print(f"{'-'*80}")
    
    print(f"\n📊 Line Statistics:")
    print(f"  Total lines:      {stats['total_lines']:,}")
    print(f"  Valid lines:      {stats['valid_lines']:,}")
    print(f"  Invalid lines:    {stats['invalid_lines']:,}")
    
    print(f"\n📊 Value Statistics:")
    print(f"  Total values:     {stats['total_values']:,}")
    print(f"  NaN count:        {stats['nan_count']:,}")
    print(f"  Inf count:        {stats['inf_count']:,}")
    print(f"  Large values:     {stats['large_value_count']:,} (|val| > 1e6)")
    print(f"  Parse errors:     {stats['parse_error_count']:,}")
    
    if stats['all_values']:
        values_array = np.array(stats['all_values'])
        finite_values = values_array[np.isfinite(values_array)]
        
        if len(finite_values) > 0:
            print(f"\n📈 Data Range (finite values only):")
            print(f"  Minimum:          {finite_values.min():.6f}")
            print(f"  Maximum:          {finite_values.max():.6f}")
            print(f"  Mean:             {finite_values.mean():.6f}")
            print(f"  Std Dev:          {finite_values.std():.6f}")
            print(f"  Median:           {np.median(finite_values):.6f}")
    
    # Overall status
    print(f"\n{'='*80}")
    if stats['invalid_lines'] == 0 and stats['nan_count'] == 0 and stats['inf_count'] == 0:
        print("✅ VALIDATION PASSED: No issues detected")
    else:
        print("❌ VALIDATION FAILED: Issues detected")
        if stats['nan_count'] > 0:
            print(f"   - {stats['nan_count']} NaN values found")
        if stats['inf_count'] > 0:
            print(f"   - {stats['inf_count']} Inf values found")
        if stats['parse_error_count'] > 0:
            print(f"   - {stats['parse_error_count']} parse errors")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate DeepLog training data files"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing training data files'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        default=['train', 'test_normal', 'test_abnormal'],
        help='List of files to validate (default: train test_normal test_abnormal)'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"❌ Error: Directory does not exist: {data_dir}")
        sys.exit(1)
    
    print(f"\n{'#'*80}")
    print(f"# Data Validation Tool")
    print(f"# Directory: {data_dir}")
    print(f"{'#'*80}")
    
    all_valid = True
    results = {}
    
    for filename in args.files:
        filepath = data_dir / filename
        
        if not filepath.exists():
            print(f"\n⚠️  Warning: File does not exist: {filepath}")
            print(f"   Skipping...")
            continue
        
        stats = validate_file(filepath)
        
        if stats is not None:
            results[filename] = stats
            print_statistics(stats, filename)
            
            if stats['invalid_lines'] > 0 or stats['nan_count'] > 0 or stats['inf_count'] > 0:
                all_valid = False
    
    # Summary
    print(f"\n{'#'*80}")
    print(f"# Validation Summary")
    print(f"{'#'*80}")
    
    for filename, stats in results.items():
        status = "✅ PASS" if (stats['invalid_lines'] == 0 and 
                              stats['nan_count'] == 0 and 
                              stats['inf_count'] == 0) else "❌ FAIL"
        print(f"  {filename:20s}: {status}")
    
    print(f"{'#'*80}\n")
    
    if all_valid:
        print("✅ All files validated successfully!")
        sys.exit(0)
    else:
        print("❌ Some files have issues. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
