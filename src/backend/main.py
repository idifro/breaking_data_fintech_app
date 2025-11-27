#!/usr/bin/env python3
"""
FastAPI Backend Server for Stock Market Application
Provides REST API endpoints to access stock data, predictions, and news
"""

from fastapi import FastAPI, HTTPException, Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import uvicorn

from services import configure_bdf_services, get_historical_service, get_prediction_service, get_news_service

# Create the FastAPI application
app = FastAPI(
    title="BDF Stock Market Data API",
    description="REST API for stock market data, predictions, and news sentiment",
    version="2.0.0"
)

# Allow cross-origin requests (needed for web dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models that define what our API returns
class StockDataPoint(BaseModel):
    """Single day of stock price data"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str

class PredictionItem(BaseModel):
    """ML model prediction for a future date"""
    prediction_date: str
    predicted_close: float
    model_confidence: float
    model_version: int
    features_used: str
    days_ahead: int
    stock_symbol: str

class NewsItem(BaseModel):
    """News article with sentiment analysis"""
    stock_symbol: str
    date: str
    news_summary: str
    sentiment_score: float
    sentiment_type: str
    news_url: Optional[str] = None
    # Extra stock info stored with the news
    stock_close: Optional[float] = None
    stock_open: Optional[float] = None
    stock_high: Optional[float] = None
    stock_low: Optional[float] = None
    stock_volume: Optional[int] = None

class DashboardSummary(BaseModel):
    """Summary info for one stock shown on dashboard"""
    symbol: str
    latest_close: float
    price_change_percent: Optional[float]
    latest_sentiment: float
    latest_prediction: Optional[float]
    latest_news_date: str

@app.on_event("startup")
async def startup_event():
    """This runs when the server starts - sets up all the data services"""
    print(" Starting BDF Stock Market Data API Server...")
    print(" Server will be available at: http://localhost:8000")
    print(" API Documentation: http://localhost:8000/docs")
    print("=" * 50)
    
    # Initialize all the services (historical data, predictions, news)
    configure_bdf_services()
    print(" BDF FastAPI backend started successfully!")

@app.get("/")
async def root():
    """Basic endpoint to check if API is running"""
    return {"message": "BDF Stock Market Data API", "version": "2.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check - used to verify server is up"""
    return {"status": "healthy", "message": "BDF Stock Market API is running"}

@app.get("/api/data-info/{symbol}")
async def get_data_info(symbol: str):
    """Get info about what data is available for a stock (for debugging)"""
    """Get available date ranges for a symbol (for debugging academic projects)"""
    try:
        historical_service = get_historical_service()
        prediction_service = get_prediction_service()
        news_service = get_news_service()
        
        info = {
            "symbol": symbol.upper(),
            "historical_data": {
                "available": False,
                "latest_date": None,
                "date_range": None
            },
            "predictions": {
                "available": False,
                "latest_date": None,
                "count": 0
            },
            "news": {
                "available": False,
                "latest_date": None,
                "count": 0
            }
        }
        
        # Historical data info
        latest_hist_date = historical_service.get_latest_available_date(symbol)
        if latest_hist_date:
            summary = historical_service.get_stock_summary(symbol)
            info["historical_data"] = {
                "available": True,
                "latest_date": latest_hist_date.strftime('%Y-%m-%d'),
                "date_range": summary.get('date_range'),
                "total_records": summary.get('total_records', 0)
            }
        
        # Prediction info
        latest_pred_date = prediction_service.get_latest_prediction_date(symbol)
        if latest_pred_date:
            predictions = prediction_service.get_latest_predictions(symbol)
            info["predictions"] = {
                "available": True,
                "latest_date": latest_pred_date.strftime('%Y-%m-%d'),
                "count": len(predictions)
            }
        
        # News info
        latest_news = news_service.get_latest_news(symbol)
        if latest_news:
            info["news"] = {
                "available": True,
                "latest_date": latest_news.get('date'),
                "sentiment": latest_news.get('sentiment_score')
            }
        
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data info: {str(e)}")

@app.get("/api/stocks/available")
async def get_available_stocks() -> List[str]:
    """Get list of available stock symbols"""
    try:
        historical_service = get_historical_service()
        return historical_service.get_available_symbols()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching available stocks: {str(e)}")

@app.get("/api/historical/{symbol}")
async def get_historical_data(
    symbol: str, 
    days: int = QueryParam(50, description="Number of days of historical data")
) -> Dict[str, Any]:
    """Get historical stock data using latest available dates"""
    try:
        historical_service = get_historical_service()
        
        # Get latest available date for this symbol (academic project approach)
        latest_date = historical_service.get_latest_available_date(symbol)
        if not latest_date:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
        
        # Calculate start date based on latest available data
        start_date = latest_date - timedelta(days=days)
        
        # Get data in the available date range
        records = historical_service.get_date_range_data(symbol, start_date, latest_date)
        
        if not records:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
        
        # Convert to list of dictionaries
        data_points = []
        for row in records:
            data_points.append({
                "date": row.get('date', ''),
                "open": float(row.get('Open', 0)),
                "high": float(row.get('High', 0)),
                "low": float(row.get('Low', 0)),
                "close": float(row.get('Close', 0)),
                "volume": int(row.get('Volume', 0)),
                "symbol": symbol.upper()
            })
        
        return {
            "symbol": symbol.upper(),
            "total_records": len(data_points),
            "data_points": data_points
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching historical data: {str(e)}")

@app.get("/api/predictions/{symbol}")
async def get_predictions(symbol: str) -> List[PredictionItem]:
    """Get predictions for a stock symbol"""
    try:
        prediction_service = get_prediction_service()
        predictions = prediction_service.get_latest_predictions(symbol)
        
        if not predictions:
            raise HTTPException(status_code=404, detail=f"No predictions found for {symbol}")
        
        return [PredictionItem(**pred) for pred in predictions]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching predictions: {str(e)}")

@app.get("/api/news/{symbol}")
async def get_news(symbol: str) -> Dict[str, Any]:
    """
    Get news articles for a stock
    Only returns news from the last trading date + next day
    """
    try:
        historical_service = get_historical_service()
        news_service = get_news_service()
        
        # Find out when we last had stock data for this symbol
        latest_stock_date = historical_service.get_latest_available_date(symbol)
        
        if not latest_stock_date:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
        
        # Fetch news only for that date and the next day
        news_items = news_service.get_news_for_stock_dates(symbol, latest_stock_date)
        
        if not news_items:
            return {
                "symbol": symbol.upper(),
                "last_stock_date": latest_stock_date.strftime('%Y-%m-%d'),
                "latest_news_date": "N/A",
                "total_news": 0,
                "latest_sentiment": 0,
                "news_items": []
            }
        
        # Latest sentiment from most recent news in fetched results
        latest_sentiment = news_items[0].get('sentiment_score', 0)
        latest_news_date = news_items[0].get('date')
        
        return {
            "symbol": symbol.upper(),
            "last_stock_date": latest_stock_date.strftime('%Y-%m-%d'),
            "latest_news_date": latest_news_date,
            "total_news": len(news_items),
            "latest_sentiment": latest_sentiment,
            "news_items": [NewsItem(**item).dict() for item in news_items]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")

@app.get("/api/dashboard")
async def get_dashboard_data() -> Dict[str, Any]:
    """
    Get summary data for all stocks to display on dashboard
    Uses concurrent processing to fetch news quickly
    """
    try:
        historical_service = get_historical_service()
        news_service = get_news_service()
        prediction_service = get_prediction_service()
        
        # Get list of all stocks we have
        symbols = historical_service.get_available_symbols()
        
        # Build a map of stock symbols to their last trading dates
        stock_dates = {}
        for symbol in symbols:
            latest_date = historical_service.get_latest_available_date(symbol)
            if latest_date:
                stock_dates[symbol] = latest_date
        
        # Fetch news for all stocks at once (this is the fast part!)
        news_summary_dict = news_service.get_news_summary_for_stocks(stock_dates)
        news_summary_dict = news_service.get_news_summary_for_stocks(stock_dates)
        
        # Get predictions for all symbols
        all_predictions = prediction_service.get_all_latest_predictions()
        prediction_dict = {}
        for pred in all_predictions:
            prediction_dict[pred['stock_symbol']] = pred['predicted_close']
        
        # Build dashboard data
        stock_summaries = []
        for symbol in symbols:
            try:
                # Get latest historical data
                df = historical_service.get_latest_stock_data(symbol, 2)
                if df.empty:
                    continue
                
                latest_close = float(df['Close'].iloc[-1]) if 'Close' in df.columns else 0.0
                
                # Calculate price change
                price_change_percent = None
                if len(df) >= 2 and 'Close' in df.columns:
                    prev_close = float(df['Close'].iloc[-2])
                    if prev_close != 0:
                        price_change_percent = ((latest_close - prev_close) / prev_close) * 100
                
                # Get sentiment and news date from batch results
                symbol_upper = symbol.upper()
                news_summary = news_summary_dict.get(symbol_upper, {'sentiment_score': 0, 'news_date': 'N/A'})
                latest_sentiment = news_summary['sentiment_score']
                latest_news_date = news_summary['news_date']
                latest_prediction = prediction_dict.get(symbol_upper)
                
                stock_summaries.append({
                    "symbol": symbol.upper(),
                    "latest_close": latest_close,
                    "price_change_percent": price_change_percent,
                    "latest_sentiment": latest_sentiment,
                    "latest_prediction": latest_prediction,
                    "latest_news_date": latest_news_date
                })
                
            except Exception as e:
                print(f"Error processing symbol {symbol}: {e}")
                continue
        
        # Calculate market metrics
        total_stocks = len(stock_summaries)
        market_sentiment_avg = sum(stock['latest_sentiment'] for stock in stock_summaries) / total_stocks if total_stocks > 0 else 0
        last_updated = datetime.now().isoformat()
        
        return {
            "total_stocks": total_stocks,
            "market_sentiment_avg": market_sentiment_avg,
            "last_updated": last_updated,
            "stock_summaries": stock_summaries
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating dashboard data: {str(e)}")

@app.get("/api/stock/{symbol}/detail")
async def get_stock_detail(symbol: str) -> Dict[str, Any]:
    """Get detailed information for a specific stock"""
    try:
        historical_service = get_historical_service()
        news_service = get_news_service()
        prediction_service = get_prediction_service()
        
        # Get stock summary
        stock_summary = historical_service.get_stock_summary(symbol)
        if not stock_summary:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Get news for last stock date + 1 day only
        latest_stock_date = historical_service.get_latest_available_date(symbol)
        news_items = news_service.get_news_for_stock_dates(symbol, latest_stock_date) if latest_stock_date else []
        
        news_summary = {
            "total_news_items": len(news_items),
            "latest_sentiment": news_items[0]['sentiment_score'] if news_items else 0,
            "latest_news_date": news_items[0]['date'] if news_items else "N/A"
        }
        
        # Get predictions summary
        predictions = prediction_service.get_latest_predictions(symbol)
        predictions_summary = {
            "total_predictions": len(predictions),
            "latest_prediction_date": predictions[0]['prediction_date'] if predictions else "N/A"
        }
        
        return {
            "symbol": symbol.upper(),
            "stock_summary": stock_summary,
            "news_summary": news_summary,
            "predictions_summary": predictions_summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock detail: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )