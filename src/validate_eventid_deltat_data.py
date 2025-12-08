#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Validation Script for EventID,deltaT format

This script validates training data files with format: EventID,deltaT EventID,deltaT ...
where EventID is a hexadecimal string and deltaT is a float value.

Usage:
    python validate_eventid_deltat_data.py --data_file <path_to_file>
"""

import argparse
import numpy as np
from pathlib import Path
import sys


def validate_file(filepath):
    """
    Validate a single data file with EventID,deltaT format.
    
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
        'total_pairs': 0,
        'nan_count': 0,
        'inf_count': 0,
        'negative_deltat_count': 0,
        'large_value_count': 0,
        'parse_error_count': 0,
        'min_deltat': float('inf'),
        'max_deltat': float('-inf'),
        'all_deltats': []
    }
    
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                stats['total_lines'] += 1
                
                # Strip whitespace and split by spaces
                pairs = line.strip().split()
                
                if not pairs:
                    continue  # Empty line
                
                for pair_idx, pair in enumerate(pairs):
                    # Split EventID,deltaT
                    parts = pair.split(',')
                    
                    if len(parts) != 2:
                        print(f"❌ Line {line_num}, Pair {pair_idx}: Invalid format '{pair}' (expected EventID,deltaT)")
                        stats['parse_error_count'] += 1
                        continue
                    
                    event_id, deltat_str = parts
                    
                    # Validate EventID (should be hex string)
                    # We don't need to validate this strictly, just skip it
                    
                    # Validate deltaT (should be a number)
                    try:
                        deltat = float(deltat_str)
                        stats['total_pairs'] += 1
                        stats['all_deltats'].append(deltat)
                        
                        # Check for NaN
                        if np.isnan(deltat):
                            print(f"❌ Line {line_num}, Pair {pair_idx}: NaN detected in deltaT")
                            stats['nan_count'] += 1
                            continue
                        
                        # Check for Inf
                        if np.isinf(deltat):
                            print(f"❌ Line {line_num}, Pair {pair_idx}: Inf detected in deltaT ({deltat})")
                            stats['inf_count'] += 1
                            continue
                        
                        # Check for negative values (might be unusual)
                        if deltat < 0:
                            print(f"⚠️  Line {line_num}, Pair {pair_idx}: Negative deltaT ({deltat})")
                            stats['negative_deltat_count'] += 1
                        
                        # Check for extremely large values
                        if abs(deltat) > 1e6:
                            print(f"⚠️  Line {line_num}, Pair {pair_idx}: Extremely large deltaT ({deltat})")
                            stats['large_value_count'] += 1
                        
                        # Update min/max
                        if deltat < stats['min_deltat']:
                            stats['min_deltat'] = deltat
                        if deltat > stats['max_deltat']:
                            stats['max_deltat'] = deltat
                            
                    except ValueError:
                        print(f"❌ Line {line_num}, Pair {pair_idx}: Cannot convert deltaT to float: '{deltat_str}'")
                        stats['parse_error_count'] += 1
                
                # Progress indicator
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} lines...", end='\r')
        
        print(f"Processed {stats['total_lines']} lines... Done!\n")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return stats


def print_statistics(stats, filename):
    """
    Print validation statistics.
    
    Args:
        stats: Statistics dictionary
        filename: Name of the file
    """
    print(f"{'-'*80}")
    print(f"Validation Results for: {filename}")
    print(f"{'-'*80}")
    
    print(f"\n📊 Data Statistics:")
    print(f"  Total lines:           {stats['total_lines']:,}")
    print(f"  Total pairs:           {stats['total_pairs']:,}")
    
    print(f"\n⚠️  Issues Found:")
    print(f"  NaN count:             {stats['nan_count']:,}")
    print(f"  Inf count:             {stats['inf_count']:,}")
    print(f"  Negative deltaT:       {stats['negative_deltat_count']:,}")
    print(f"  Large values:          {stats['large_value_count']:,} (|val| > 1e6)")
    print(f"  Parse errors:          {stats['parse_error_count']:,}")
    
    if stats['all_deltats']:
        deltats_array = np.array(stats['all_deltats'])
        finite_deltats = deltats_array[np.isfinite(deltats_array)]
        
        if len(finite_deltats) > 0:
            print(f"\n📈 DeltaT Range (finite values only):")
            print(f"  Minimum:               {finite_deltats.min():.6f}")
            print(f"  Maximum:               {finite_deltats.max():.6f}")
            print(f"  Mean:                  {finite_deltats.mean():.6f}")
            print(f"  Std Dev:               {finite_deltats.std():.6f}")
            print(f"  Median:                {np.median(finite_deltats):.6f}")
            
            # Percentiles
            print(f"\n📊 Percentiles:")
            print(f"  25th percentile:       {np.percentile(finite_deltats, 25):.6f}")
            print(f"  75th percentile:       {np.percentile(finite_deltats, 75):.6f}")
            print(f"  95th percentile:       {np.percentile(finite_deltats, 95):.6f}")
            print(f"  99th percentile:       {np.percentile(finite_deltats, 99):.6f}")
    
    # Overall status
    print(f"\n{'='*80}")
    if stats['nan_count'] == 0 and stats['inf_count'] == 0 and stats['parse_error_count'] == 0:
        print("✅ VALIDATION PASSED: No critical issues detected")
    else:
        print("❌ VALIDATION FAILED: Critical issues detected")
        if stats['nan_count'] > 0:
            print(f"   - {stats['nan_count']} NaN values found")
        if stats['inf_count'] > 0:
            print(f"   - {stats['inf_count']} Inf values found")
        if stats['parse_error_count'] > 0:
            print(f"   - {stats['parse_error_count']} parse errors")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate EventID,deltaT format training data files"
    )
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to training data file'
    )
    
    args = parser.parse_args()
    
    data_file = Path(args.data_file)
    
    if not data_file.exists():
        print(f"❌ Error: File does not exist: {data_file}")
        sys.exit(1)
    
    print(f"\n{'#'*80}")
    print(f"# EventID,DeltaT Data Validation Tool")
    print(f"# File: {data_file}")
    print(f"{'#'*80}")
    
    stats = validate_file(data_file)
    
    if stats is not None:
        print_statistics(stats, data_file.name)
        
        if stats['nan_count'] > 0 or stats['inf_count'] > 0 or stats['parse_error_count'] > 0:
            print("❌ Data validation failed. Please review the output above.")
            sys.exit(1)
        else:
            print("✅ Data validation successful!")
            sys.exit(0)
    else:
        print("❌ Data validation failed due to errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
