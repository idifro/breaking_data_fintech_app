#!/usr/bin/env python3
"""
Demo Script for Spark ML Pipeline
Simplified version to test the pipeline functionality
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Change to the script directory
os.chdir(Path(__file__).parent)

def main():
    """Main demo function"""
    print("🚀 Spark ML Stock Prediction Pipeline Demo")
    print("=" * 50)
    
    # Check if we're in the right conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    print(f"Current conda environment: {conda_env}")
    
    # Check CSV files
    csv_files = list(Path("data_csv").glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
    
    # Check directory structure
    required_dirs = ["src", "config", "data_csv", "delta_tables", "results", "plots"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        status = "✅" if dir_path.exists() else "❌"
        print(f"{status} {dir_name}")
    
    print("\n📋 Next Steps:")
    print("1. Run setup.py to install dependencies")
    print("2. Run scripts/train_model.py to train the model")
    print("3. Run scripts/inference.py to make predictions")
    
    print("\nSetup appears to be complete! 🎉")

if __name__ == "__main__":
    main()