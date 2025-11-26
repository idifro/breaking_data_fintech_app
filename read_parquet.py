#!/usr/bin/env python3
"""
Script to read and display parquet files
"""

import pandas as pd
import sys
from pathlib import Path


def read_parquet(file_path: str):
    """
    Read and display a parquet file
    
    Args:
        file_path: Path to the parquet file
    """
    print("=" * 80)
    print(f"📊 Reading Parquet File: {file_path}")
    print("=" * 80)
    
    try:
        # Check if file exists
        if not Path(file_path).exists():
            print(f"\n❌ File not found: {file_path}")
            return None
        
        # Read parquet file
        print(f"\n[PANDAS] Reading parquet file...")
        df = pd.read_parquet(file_path)
        
        # Display basic info
        print(f"\n✅ Successfully loaded parquet file!")
        
        # Display shape
        print(f"\n📊 DataFrame Shape: {df.shape}")
        print(f"   Rows: {df.shape[0]}")
        print(f"   Columns: {df.shape[1]}")
        
        # Display column info
        print(f"\n📋 Columns:")
        print("=" * 80)
        print(df.dtypes)
        
        # Display first rows
        print(f"\n📋 First 10 Rows:")
        print("=" * 80)
        print(df.head(10).to_string())
        
        # Display last rows
        print(f"\n📋 Last 5 Rows:")
        print("=" * 80)
        print(df.tail(5).to_string())
        
        # Display summary statistics
        print(f"\n📊 Summary Statistics:")
        print("=" * 80)
        print(df.describe().to_string())
        
        # Display null counts
        print(f"\n📊 Null Value Counts:")
        print("=" * 80)
        null_counts = df.isnull().sum()
        print(null_counts[null_counts > 0])
        if null_counts.sum() == 0:
            print("✅ No null values found!")
        
        # Display unique values for categorical columns
        if 'Symbol' in df.columns:
            print(f"\n📊 Unique Stocks:")
            print("=" * 80)
            print(df['Symbol'].unique())
            print(f"\nTotal unique stocks: {df['Symbol'].nunique()}")
        
        # Display full dataframe
        print(f"\n📋 Complete DataFrame:")
        print("=" * 80)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.to_string())
        
        return df
        
    except Exception as e:
        print(f"\n❌ Error reading parquet file: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    
    # Default path
    parquet_path = "tmp/test_yfinance_enriched.parquet"
    
    # Check if custom path provided
    if len(sys.argv) > 1:
        parquet_path = sys.argv[1]
    
    # Read and display the parquet file
    df = read_parquet(parquet_path)
    
    if df is not None:
        print("\n" + "=" * 80)
        print("✅ Parquet File Read Complete!")
        print("=" * 80)
        
        # Save to CSV option
        save_csv = input("\nSave to CSV? (y/n): ").strip().lower()
        if save_csv == 'y':
            csv_path = parquet_path.replace('.parquet', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"✅ Saved to: {csv_path}")


if __name__ == "__main__":
    main()
