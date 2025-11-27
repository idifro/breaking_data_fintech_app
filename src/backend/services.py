#!/usr/bin/env python3
"""
Services module for the BDF Stock Market Application
Contains three main service classes that handle data access:
- Historical stock data from Delta Lake
- Prediction data from ML models
- News and sentiment data from AstraDB
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, date, timedelta
import pandas as pd
import os
from pathlib import Path
from astrapy import DataAPIClient
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    BDF_ROOT, DELTA_TABLES_PATH, PREDICTIONS_DATA_PATH, 
    get_stock_delta_path, NOSQL_DB_PATH, AVAILABLE_SYMBOLS,
    ASTRA_DB_ENDPOINT, ASTRA_DB_TOKEN, ASTRA_DB_TABLE_NAME
)

class BDFHistoricalDataService:
    """
    Service for loading historical stock price data
    Data is stored in Delta Lake format (parquet files)
    """
    
    def __init__(self):
        self._cache = {}  # Keep loaded data in memory to speed things up
        self._load_available_symbols()
    
    def _load_available_symbols(self):
        """Figure out which stocks we have data for"""
        self.available_symbols = []
        for symbol in AVAILABLE_SYMBOLS:
            stock_path = get_stock_delta_path(symbol)
            # Check if the folder exists and has parquet files
            if stock_path.exists() and list(stock_path.glob("*.parquet")):
                self.available_symbols.append(symbol)
        print(f" Found {len(self.available_symbols)} stock tables in BDF structure")
    
    def _load_stock_data(self, symbol: str) -> pd.DataFrame:
        """Load data for one stock - checks cache first to avoid reloading"""
        # Return from cache if we already loaded it
        if symbol in self._cache:
            return self._cache[symbol]
        
        try:
            stock_path = get_stock_delta_path(symbol)
            if not stock_path.exists():
                print(f" No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Get all the parquet files for this stock
            parquet_files = list(stock_path.glob("*.parquet"))
            if not parquet_files:
                print(f" No parquet files found for {symbol}")
                return pd.DataFrame()
            
            # Load all parquet files for this stock
            dfs = []
            for parquet_file in parquet_files:
                try:
                    df = pd.read_parquet(parquet_file)
                    dfs.append(df)
                except Exception as e:
                    print(f" Error reading {parquet_file}: {e}")
            
            if not dfs:
                return pd.DataFrame()
            
            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Ensure we have the required columns and normalize them
            if 'Date' in combined_df.columns:
                combined_df['Date'] = pd.to_datetime(combined_df['Date'])
                combined_df = combined_df.sort_values('Date')
            
            # Add symbol column if not present
            if 'Symbol' not in combined_df.columns:
                combined_df['Symbol'] = symbol
            
            # Cache the result
            self._cache[symbol] = combined_df
            print(f" Loaded {len(combined_df)} records for {symbol}")
            return combined_df
            
        except Exception as e:
            print(f" Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_stock_data(self, stock_symbol: str, start_date: date = None, end_date: date = None) -> pd.DataFrame:
        """Get historical stock data for a symbol within date range"""
        df = self._load_stock_data(stock_symbol.upper())
        if df.empty:
            return df
        
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        
        return df
    
    def get_latest_stock_data(self, stock_symbol: str, days: int = 50) -> pd.DataFrame:
        """Get latest N days of stock data"""
        df = self._load_stock_data(stock_symbol.upper())
        if df.empty:
            return df
        
        # Get last N days
        return df.tail(days)
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available stock symbols"""
        return self.available_symbols
    
    def get_date_range(self, stock_symbol: str) -> tuple[date, date]:
        """Get available date range for a stock symbol"""
        df = self._load_stock_data(stock_symbol.upper())
        if df.empty or 'Date' not in df.columns:
            return None, None
        
        start_date = df['Date'].min().date()
        end_date = df['Date'].max().date()
        return start_date, end_date
    
    def get_stock_summary(self, stock_symbol: str) -> Dict[str, Any]:
        """Get summary statistics for a stock"""
        df = self._load_stock_data(stock_symbol.upper())
        if df.empty:
            return {}
        
        return {
            'symbol': stock_symbol.upper(),
            'total_records': len(df),
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d'),
                'end': df['Date'].max().strftime('%Y-%m-%d')
            },
            'price_stats': {
                'avg_close': float(df['Close'].mean()) if 'Close' in df.columns else 0,
                'min_close': float(df['Close'].min()) if 'Close' in df.columns else 0,
                'max_close': float(df['Close'].max()) if 'Close' in df.columns else 0,
                'latest_close': float(df['Close'].iloc[-1]) if 'Close' in df.columns else 0
            }
        }
    
    def get_latest_available_date(self, stock_symbol: str) -> Optional[date]:
        """Get the latest available date for a stock symbol"""
        df = self._load_stock_data(stock_symbol.upper())
        if df.empty or 'Date' not in df.columns:
            return None
        return df['Date'].max().date()
    
    def get_date_range_data(self, stock_symbol: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Get stock data within a specific date range using latest available data"""
        df = self._load_stock_data(stock_symbol.upper())
        if df.empty:
            return []
        
        # If requested dates are beyond available data, use available data range
        available_start = df['Date'].min().date()
        available_end = df['Date'].max().date()
        
        # Adjust dates to available range
        actual_start = max(start_date, available_start)
        actual_end = min(end_date, available_end)
        
        filtered_df = df[
            (df['Date'].dt.date >= actual_start) & 
            (df['Date'].dt.date <= actual_end)
        ].copy()
        
        filtered_df['date'] = filtered_df['Date'].dt.strftime('%Y-%m-%d')
        return filtered_df.to_dict('records')


class BDFPredictionDataService:
    """Prediction data service using BDF Delta Lake structure"""
    
    def __init__(self):
        self._df = None
        self._load_data()
    
    def _load_data(self):
        """Load prediction data from BDF structure"""
        try:
            predictions_path = PREDICTIONS_DATA_PATH
            if predictions_path.exists():
                # Find parquet files in predictions directory
                parquet_files = list(predictions_path.glob("*.parquet"))
                if parquet_files:
                    dfs = []
                    for parquet_file in parquet_files:
                        df = pd.read_parquet(parquet_file)
                        dfs.append(df)
                    
                    if dfs:
                        self._df = pd.concat(dfs, ignore_index=True)
                        self._df['prediction_date'] = pd.to_datetime(self._df['prediction_date'])
                        self._df = self._df.sort_values(['stock_symbol', 'prediction_date'])
                        print(f" Loaded {len(self._df)} prediction records from BDF structure")
                        return
            
            print(" No prediction data found in BDF structure")
            self._df = pd.DataFrame()
            
        except Exception as e:
            print(f" Error loading prediction data: {e}")
            self._df = pd.DataFrame()
    
    def get_latest_predictions(self, stock_symbol: str) -> List[Dict[str, Any]]:
        """Get latest predictions for a stock symbol"""
        if self._df.empty:
            return []
        
        # Case-insensitive matching
        symbol_data = self._df[self._df['stock_symbol'].str.upper() == stock_symbol.upper()]
        if symbol_data.empty:
            return []
        
        # Convert to list of dictionaries
        predictions = []
        for _, row in symbol_data.iterrows():
            pred_dict = row.to_dict()
            # Ensure date format
            if hasattr(pred_dict['prediction_date'], 'strftime'):
                pred_dict['prediction_date'] = pred_dict['prediction_date'].strftime('%Y-%m-%d')
            predictions.append(pred_dict)
        
        return predictions
    
    def get_prediction_by_date(self, stock_symbol: str, prediction_date: date) -> Optional[Dict[str, Any]]:
        """Get prediction for a specific date"""
        if self._df.empty:
            return None
        
        mask = (
            (self._df['stock_symbol'].str.upper() == stock_symbol.upper()) &
            (self._df['prediction_date'].dt.date == prediction_date)
        )
        result = self._df[mask]
        
        if result.empty:
            return None
        
        return result.iloc[0].to_dict()
    
    def get_latest_prediction_date(self, stock_symbol: str) -> Optional[date]:
        """Get the latest prediction date available for a stock symbol"""
        if self._df.empty:
            return None
        
        symbol_data = self._df[self._df['stock_symbol'].str.upper() == stock_symbol.upper()]
        if symbol_data.empty:
            return None
        
        return symbol_data['prediction_date'].max().date()
    
    def get_all_latest_predictions(self) -> List[Dict[str, Any]]:
        """Get latest predictions for all stocks"""
        if self._df.empty:
            return []
        
        latest_predictions = []
        for symbol in self._df['stock_symbol'].unique():
            symbol_data = self._df[self._df['stock_symbol'] == symbol]
            latest = symbol_data.loc[symbol_data['prediction_date'].idxmax()]
            pred_dict = latest.to_dict()
            # Convert dates to strings
            if hasattr(pred_dict['prediction_date'], 'strftime'):
                pred_dict['prediction_date'] = pred_dict['prediction_date'].strftime('%Y-%m-%d')
            latest_predictions.append(pred_dict)
        
        return latest_predictions


class BDFNewsDataService:
    """
    Service for fetching news articles and sentiment scores from AstraDB
    Uses concurrent processing to fetch data for multiple stocks faster
    """
    
    def __init__(self):
        """Connect to the AstraDB cloud database"""
        try:
            print(' Connecting to AstraDB...')
            client = DataAPIClient()
            self.database = client.get_database(ASTRA_DB_ENDPOINT, token=ASTRA_DB_TOKEN)
            self.collection = self.database.get_collection(ASTRA_DB_TABLE_NAME)
            print(f' Connected to AstraDB collection: {ASTRA_DB_TABLE_NAME}')
        except Exception as e:
            print(f" Error connecting to AstraDB: {e}")
            self.collection = None
    
    def get_latest_news(self, stock_symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest news entry for a stock symbol"""
        try:
            if not self.collection:
                print(" AstraDB collection not initialized")
                return None
            
            # Query for the stock symbol (will be sorted by published_at DESC due to clustering)
            cursor = self.collection.find(
                {"stock_symbol": stock_symbol.upper()},
                limit=1
            )
            
            result = list(cursor)
            if result:
                news_item = result[0]
                # Convert published_at timestamp to date string
                published_at = news_item.get('published_at')
                if published_at:
                    if isinstance(published_at, datetime):
                        date_str = published_at.strftime('%Y-%m-%d')
                    else:
                        date_str = str(published_at)[:10]  # Extract date part
                else:
                    date_str = datetime.now().strftime('%Y-%m-%d')
                
                return {
                    'stock_symbol': news_item.get('stock_symbol'),
                    'date': date_str,
                    'news_summary': news_item.get('news_summary'),
                    'sentiment_score': float(news_item.get('sentiment_score', 0)),
                    'sentiment_type': self._get_sentiment_type(news_item.get('sentiment_score', 0)),
                    'news_url': news_item.get('news_url'),
                    'timestamp': published_at,
                    # Additional stock data
                    'stock_close': news_item.get('stock_close'),
                    'stock_open': news_item.get('stock_open'),
                    'stock_high': news_item.get('stock_high'),
                    'stock_low': news_item.get('stock_low'),
                    'stock_volume': news_item.get('stock_volume')
                }
            return None
        except Exception as e:
            print(f" Error getting latest news for {stock_symbol}: {e}")
            return None
    
    def get_latest_news_articles(self, stock_symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get latest N news articles for a stock symbol"""
        try:
            if not self.collection:
                print(" AstraDB collection not initialized")
                return []
            
            # Query for the stock symbol with limit (sorted by published_at DESC)
            cursor = self.collection.find(
                {"stock_symbol": stock_symbol.upper()},
                limit=limit
            )
            
            results = []
            for news_item in cursor:
                published_at = news_item.get('published_at')
                if published_at:
                    if isinstance(published_at, datetime):
                        date_str = published_at.strftime('%Y-%m-%d')
                    else:
                        date_str = str(published_at)[:10]
                else:
                    date_str = datetime.now().strftime('%Y-%m-%d')
                
                results.append({
                    'stock_symbol': news_item.get('stock_symbol'),
                    'date': date_str,
                    'news_summary': news_item.get('news_summary'),
                    'sentiment_score': float(news_item.get('sentiment_score', 0)),
                    'sentiment_type': self._get_sentiment_type(news_item.get('sentiment_score', 0)),
                    'news_url': news_item.get('news_url'),
                    'timestamp': published_at,
                    'stock_close': news_item.get('stock_close'),
                    'stock_open': news_item.get('stock_open'),
                    'stock_high': news_item.get('stock_high'),
                    'stock_low': news_item.get('stock_low'),
                    'stock_volume': news_item.get('stock_volume')
                })
            
            return results
        except Exception as e:
            print(f" Error getting news articles for {stock_symbol}: {e}")
            return []
    
    def get_news_by_date_range(self, stock_symbol: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Get news entries for a stock within a date range (kept for backward compatibility)"""
        try:
            if not self.collection:
                print(" AstraDB collection not initialized")
                return []
            
            # Convert dates to datetime for comparison
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            
            # Query with date range filter
            cursor = self.collection.find({
                "stock_symbol": stock_symbol.upper(),
                "published_at": {"$gte": start_datetime, "$lte": end_datetime}
            })
            
            results = []
            for news_item in cursor:
                published_at = news_item.get('published_at')
                if published_at:
                    if isinstance(published_at, datetime):
                        date_str = published_at.strftime('%Y-%m-%d')
                    else:
                        date_str = str(published_at)[:10]
                else:
                    date_str = start_date.strftime('%Y-%m-%d')
                
                results.append({
                    'stock_symbol': news_item.get('stock_symbol'),
                    'date': date_str,
                    'news_summary': news_item.get('news_summary'),
                    'sentiment_score': float(news_item.get('sentiment_score', 0)),
                    'sentiment_type': self._get_sentiment_type(news_item.get('sentiment_score', 0)),
                    'news_url': news_item.get('news_url'),
                    'timestamp': published_at,
                    'stock_close': news_item.get('stock_close'),
                    'stock_open': news_item.get('stock_open'),
                    'stock_high': news_item.get('stock_high'),
                    'stock_low': news_item.get('stock_low'),
                    'stock_volume': news_item.get('stock_volume')
                })
            
            return sorted(results, key=lambda x: x['date'])
        except Exception as e:
            print(f" Error getting news for {stock_symbol}: {e}")
            return []
    
    def get_all_stocks_latest_news(self) -> List[Dict[str, Any]]:
        """Get latest news for all stocks"""
        try:
            if not self.collection:
                print(" AstraDB collection not initialized")
                return []
            
            latest_news = []
            
            # Get distinct stock symbols from available symbols
            for symbol in AVAILABLE_SYMBOLS:
                latest = self.get_latest_news(symbol)
                if latest:
                    latest_news.append(latest)
            
            return latest_news
        except Exception as e:
            print(f" Error getting all latest news: {e}")
            return []
    
    def get_news_for_stock_dates(self, stock_symbol: str, last_stock_date: date) -> List[Dict[str, Any]]:
        """
        Get news for the last trading date + next day only
        This ensures we only show relevant news for the stock's latest data
        """
        try:
            if not self.collection:
                print("AstraDB collection not initialized")
                return []
            
            # We want news for last trading date and the next day
            next_day = last_stock_date + timedelta(days=1)
            target_dates = {
                last_stock_date.strftime('%Y-%m-%d'),
                next_day.strftime('%Y-%m-%d')
            }
            
            # Fetch some records and filter for our dates
            # AstraDB returns newest first, so we need to fetch enough to reach our dates
            cursor = self.collection.find(
                {"stock_symbol": stock_symbol.upper()},
                limit=1000  # Should be enough to reach back to Dec 2023
            )
            
            results = []
            for news_item in cursor:
                published_at = news_item.get('published_at')
                if published_at:
                    if isinstance(published_at, datetime):
                        date_str = published_at.strftime('%Y-%m-%d')
                    else:
                        date_str = str(published_at)[:10]
                else:
                    continue
                
                # Only include news from target dates
                if date_str in target_dates:
                    results.append({
                        'stock_symbol': news_item.get('stock_symbol'),
                        'date': date_str,
                        'news_summary': news_item.get('news_summary'),
                        'sentiment_score': float(news_item.get('sentiment_score', 0)),
                        'sentiment_type': self._get_sentiment_type(news_item.get('sentiment_score', 0)),
                        'news_url': news_item.get('news_url'),
                        'timestamp': published_at,
                        'stock_close': news_item.get('stock_close'),
                        'stock_open': news_item.get('stock_open'),
                        'stock_high': news_item.get('stock_high'),
                        'stock_low': news_item.get('stock_low'),
                        'stock_volume': news_item.get('stock_volume')
                    })
                    
                    if len(results) >= 20:
                        break
            
            return sorted(results, key=lambda x: x['date'], reverse=True)
        except Exception as e:
            print(f"Error getting news for {stock_symbol} on stock dates: {e}")
            return []
    
    def get_news_summary_for_stocks(self, stock_dates: Dict[str, date]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch news for multiple stocks at the same time (concurrent processing)
        This is way faster than fetching one by one
        
        Args:
            stock_dates: Dictionary with stock symbol as key and last trading date as value
        Returns:
            Dictionary with stock symbol as key and news data as value
        """
        try:
            if not self.collection:
                print("AstraDB collection not initialized")
                return {}
            
            results = {}
            
            def fetch_news_for_symbol(symbol: str, last_date: date) -> tuple:
                """Internal function that gets news for one stock"""
                next_day = last_date + timedelta(days=1)
                target_dates = {
                    last_date.strftime('%Y-%m-%d'),
                    next_day.strftime('%Y-%m-%d')
                }
                
                try:
                    # Fetch some news articles for this stock
                    cursor = self.collection.find(
                        {"stock_symbol": symbol.upper()},
                        limit=150  # Reduced limit for faster response
                    )
                    
                    for news_item in cursor:
                        published_at = news_item.get('published_at')
                        if published_at:
                            if isinstance(published_at, datetime):
                                date_str = published_at.strftime('%Y-%m-%d')
                            else:
                                date_str = str(published_at)[:10]
                            
                            if date_str in target_dates:
                                return (symbol, {
                                    'sentiment_score': float(news_item.get('sentiment_score', 0)),
                                    'news_date': date_str
                                })
                    
                    # If we didn't find news for those dates, return default
                    return (symbol, {'sentiment_score': 0, 'news_date': 'N/A'})
                    
                except Exception as e:
                    print(f"Error fetching news for {symbol}: {e}")
                    return (symbol, {'sentiment_score': 0, 'news_date': 'N/A'})
            
            # This is the concurrent part - fetch news for all stocks at once
            # ThreadPoolExecutor runs up to 10 fetches in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Start all the fetch operations
                future_to_symbol = {
                    executor.submit(fetch_news_for_symbol, symbol, last_date): symbol
                    for symbol, last_date in stock_dates.items()
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_symbol):
                    symbol, result = future.result()
                    # Store with uppercase key so lookups work correctly
                    results[symbol.upper()] = result
                    # Store with uppercase key for consistency
                    results[symbol.upper()] = result
            
            return results
            
        except Exception as e:
            print(f"Error getting news summary for stocks: {e}")
            return {}
    
    def _get_sentiment_type(self, score: float) -> str:
        """Determine sentiment type from score"""
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"


def configure_bdf_services():
    """Configure services for BDF structure"""
    global _historical_service, _prediction_service, _news_service
    
    _historical_service = BDFHistoricalDataService()
    _prediction_service = BDFPredictionDataService() 
    _news_service = BDFNewsDataService()
    
    print(" BDF services configured successfully!")


def get_historical_service() -> BDFHistoricalDataService:
    """Get the historical data service"""
    return _historical_service


def get_prediction_service() -> BDFPredictionDataService:
    """Get the prediction service"""
    return _prediction_service


def get_news_service() -> BDFNewsDataService:
    """Get the news service"""
    return _news_service


# Global service instances
_historical_service = None
_prediction_service = None
_news_service = None