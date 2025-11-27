#!/usr/bin/env python3
"""
Configuration file for the BDF project
Contains all the path configurations and database settings
"""

import os
from pathlib import Path

# Figure out where the BDF root folder is
# Since we're in src/backend/, we need to go up 3 levels to reach BDF/
BDF_ROOT = Path(__file__).parent.parent.parent  
DELTA_TABLES_PATH = BDF_ROOT / "delta_tables"  # Where stock data is stored
SRC_PATH = BDF_ROOT / "src"
BACKEND_PATH = Path(__file__).parent  # Current backend folder

# Try to find the nosql_db folder - it could be in different places
PROJECT_ROOT = BDF_ROOT.parent  # Go up one more level

# Check these locations in order until we find the nosql_db folder
possible_nosql_paths = [
    PROJECT_ROOT / "nosql_db",  # Outside BDF folder
    BDF_ROOT / "nosql_db",      # Inside BDF folder
    BDF_ROOT / "data" / "nosql_db",  # In a data subfolder
]

NOSQL_DB_PATH = None
for path in possible_nosql_paths:
    if path.exists():
        NOSQL_DB_PATH = path
        break

# If we didn't find it anywhere, just use the first option
if NOSQL_DB_PATH is None:
    NOSQL_DB_PATH = possible_nosql_paths[0]

# Paths to different data folders
HISTORICAL_DATA_PATH = DELTA_TABLES_PATH / "stock_data"
PREDICTIONS_DATA_PATH = DELTA_TABLES_PATH / "stock_predictions"

# Helper function to get the path for a specific stock's data
def get_stock_delta_path(symbol: str) -> Path:
    """Returns the folder path where a stock's data is stored"""
    return DELTA_TABLES_PATH / f"stock_{symbol.upper()}"

# List of all stock symbols we're tracking
AVAILABLE_SYMBOLS = [
    "AAL", "ABBV", "AMD", "AMGN", "BABA", "BIIB", 
    "CMCSA", "CMG", "COP", "COST", "CRM", "CVX", "EBAY", 
    "GE", "GSK", "MRK", "NKE", "ORCL", 
    "PEP", "PYPL", "QCOM", "QQQ", "TSM", "USO", "WFC"
]

# AstraDB settings for cloud database
ASTRA_DB_ENDPOINT = 'https://d1091311-1ba8-4970-bd74-937d6cda55b0-us-east-2.apps.astra.datastax.com'
ASTRA_DB_TOKEN = 'AstraCS:YOUR_TOKEN_HERE'
ASTRA_DB_TABLE_NAME = 'news_sentiment_1'

def ensure_directories():
    """Check if all the required folders exist"""
    if not NOSQL_DB_PATH.exists():
        print(f"Warning: NoSQL DB path not found: {NOSQL_DB_PATH}")
        print("Checked locations:")
        for i, path in enumerate(possible_nosql_paths, 1):
            status = " EXISTS" if path.exists() else " NOT FOUND"
            print(f"  {i}. {path} - {status}")
        print("Please ensure the nosql_db folder exists in one of these locations")
    return NOSQL_DB_PATH.exists()

if __name__ == "__main__":
    print(f"BDF Root: {BDF_ROOT}")
    print(f"Delta Tables: {DELTA_TABLES_PATH}")
    print(f"Source Code: {SRC_PATH}")
    print(f"Available Symbols: {len(AVAILABLE_SYMBOLS)}")
    ensure_directories()