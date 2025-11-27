#!/usr/bin/env python3
"""
Streamlit Dashboard for Stock Market App
This is the frontend - shows charts, news, and predictions in a web browser
"""

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import time
import sys
import os
from pathlib import Path

# Need to import stuff from backend folder
# Go up two levels: app.py -> streamlit_dashboard -> src -> then down to backend
backend_path = Path(__file__).parent.parent / "backend"
sys.path.append(str(backend_path))

# Backend API settings
API_BASE_URL = "http://localhost:8000/api"

# Configure the web page settings
st.set_page_config(
    page_title="BDF Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling to make things look nicer
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .positive { color: #28a745; font-weight: bold; }
    .negative { color: #dc3545; font-weight: bold; }
    .neutral { color: #6c757d; font-weight: bold; }
    
    .stMetric { background-color: #ffffff; padding: 1rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes - don't fetch same data repeatedly
def fetch_api_data(endpoint: str) -> Optional[Dict[str, Any]]:
    """Get data from the backend API
    Uses caching so we don't make too many requests
    """
    try:
        # Call the backend API and wait up to 30 seconds for response
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

def get_sentiment_color(score: float) -> str:
    """Figure out what color to show for sentiment
    Positive = green, negative = red, neutral = gray
    """
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    else:
        return "neutral"

def format_sentiment_score(score: float) -> str:
    """Make sentiment score look nice with colors and emojis"""
    color_class = get_sentiment_color(score)
    if score > 0:
        return f'<span class="{color_class}">+{score:.3f} 😊</span>'
    elif score < 0:
        return f'<span class="{color_class}">{score:.3f} 😟</span>'
    else:
        return f'<span class="{color_class}">{score:.3f} 😐</span>'

def create_price_chart(historical_data: List[Dict], predictions: List[Dict], symbol: str):
    """Create the main price chart
    Shows historical prices as a line and next day prediction as a dotted line
    """
    if not historical_data:
        st.warning("No historical data available")
        return None
    
    # Convert data to pandas DataFrame (easier to work with)
    df_hist = pd.DataFrame([point for point in historical_data])
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    
    # Create the chart
    fig = go.Figure()
    
    # Add the blue line for historical prices
    fig.add_trace(go.Scatter(
        x=df_hist['date'],
        y=df_hist['close'],
        mode='lines',
        name='Historical Close Price',
        line=dict(color='blue', width=2),
        hovertemplate="<b>%{text}</b><br>" +
                      "Date: %{x}<br>" +
                      "Close: $%{y:.2f}<br>" +
                      "<extra></extra>",
        text=[f"{symbol} Historical"] * len(df_hist)
    ))
    
    # Add prediction line if we have prediction data
    if predictions:
        # Convert predictions to DataFrame and find the next day's prediction
        df_pred = pd.DataFrame(predictions)
        df_pred['prediction_date'] = pd.to_datetime(df_pred['prediction_date'])
        
        # Find the last date we have real stock data for
        latest_hist_date = df_hist['date'].max()
        
        # Look for predictions after the last historical date
        future_predictions = df_pred[df_pred['prediction_date'] > latest_hist_date]
        
        if not future_predictions.empty:
            # Get the closest prediction (next day)
            next_prediction = future_predictions.loc[future_predictions['prediction_date'].idxmin()]
            next_price = next_prediction['predicted_close']
            next_date = next_prediction['prediction_date']
            confidence = next_prediction.get('model_confidence', 0) * 100
            
            # Draw an orange dotted line showing the predicted price
            x_range = [df_hist['date'].min(), next_date]
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[next_price, next_price],
                mode='lines',
                name=f'Next Day Prediction: ${next_price:.2f}',
                line=dict(color='orange', width=2, dash='dot'),
                hovertemplate="<b>Next Day Prediction</b><br>" +
                              f"Date: {next_date.strftime('%Y-%m-%d')}<br>" +
                              f"Predicted Price: ${next_price:.2f}<br>" +
                              f"Confidence: {confidence:.1f}%<br>" +
                              "<extra></extra>",
                showlegend=True
            ))
            
            # Add a diamond marker at the prediction point
            fig.add_trace(go.Scatter(
                x=[next_date],
                y=[next_price],
                mode='markers',
                name='Prediction Point',
                marker=dict(size=10, color='orange', symbol='diamond'),
                hovertemplate="<b>Prediction Point</b><br>" +
                              f"Date: {next_date.strftime('%Y-%m-%d')}<br>" +
                              f"Predicted Price: ${next_price:.2f}<br>" +
                              f"Confidence: {confidence:.1f}%<br>" +
                              "<extra></extra>",
                showlegend=False
            ))
    
    # Make the chart look nice with labels and formatting
    fig.update_layout(
        title=f"{symbol} Stock Price Analysis & Next Day Prediction",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=500
    )
    
    return fig

def show_dashboard():
    """Show the main dashboard page with all stocks overview"""
    st.title("📈 BDF Stock Market Dashboard")
    st.markdown("---")
    
    # Get data for all stocks from backend
    dashboard_data = fetch_api_data("/dashboard")
    
    if not dashboard_data:
        st.error("Failed to load dashboard data")
        return
    
    # Show main statistics at the top in 4 boxes
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stocks", dashboard_data.get('total_stocks', 0))
    
    with col2:
        avg_sentiment = dashboard_data.get('market_sentiment_avg', 0)
        sentiment_delta = f"{avg_sentiment:+.3f}"
        st.metric("Market Sentiment", f"{avg_sentiment:.3f}", sentiment_delta)
    
    with col3:
        last_updated = dashboard_data.get('last_updated', '')
        if last_updated:
            update_time = datetime.fromisoformat(last_updated.replace('Z', '')).strftime('%H:%M:%S')
            st.metric("Last Updated", update_time)
    
    with col4:
        # Count how many stocks have positive news sentiment
        stock_summaries = dashboard_data.get('stock_summaries', [])
        positive_count = sum(1 for stock in stock_summaries if stock['latest_sentiment'] > 0)
        st.metric("Positive Sentiment", f"{positive_count}/{len(stock_summaries)}")
    
    st.markdown("---")
    
    # Show table with all stocks and their data
    st.subheader(" Stock Overview")
    
    if stock_summaries:
        # Build the table data
        display_data = []
        for stock in stock_summaries:
            display_data.append({
                'Symbol': stock['symbol'],
                'Latest Close': f"${stock['latest_close']:.2f}",
                'Price Change %': f"{stock['price_change_percent']:.2f}%" if stock['price_change_percent'] else "N/A",
                'Sentiment Score': f"{stock['latest_sentiment']:.3f}",
                'Predicted Price': f"${stock['latest_prediction']:.2f}" if stock['latest_prediction'] else "N/A",
                'Latest News Date': stock['latest_news_date']
            })
        
        df_display = pd.DataFrame(display_data)
        
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400
        )

def show_stock_detail():
    """Show detailed view for one stock
    Includes price chart, predictions, and recent news
    """
    st.title("🔍 BDF Stock Detail Analysis")
    
    # Get list of all available stocks for dropdown
    available_stocks = fetch_api_data("/stocks/available")
    
    if not available_stocks:
        st.error("Failed to load available stocks")
        return
    
    selected_symbol = st.selectbox(
        "Select a stock symbol:",
        options=available_stocks,
        index=0 if available_stocks else None
    )
    
    if not selected_symbol:
        st.warning("Please select a stock symbol")
        return
    
    # Load all the data for the selected stock (shows spinner while loading)
    with st.spinner(f"Loading data for {selected_symbol}..."):
        stock_detail = fetch_api_data(f"/stock/{selected_symbol}/detail")
        historical_data = fetch_api_data(f"/historical/{selected_symbol}?days=50")  # Last 50 days
        predictions = fetch_api_data(f"/predictions/{selected_symbol}")
        news_data = fetch_api_data(f"/news/{selected_symbol}")  # News for last stock date + next day
    
    if not stock_detail:
        st.error(f"Failed to load data for {selected_symbol}")
        return
    
    # Show key numbers at the top
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stock_summary = stock_detail.get('stock_summary', {})
        price_stats = stock_summary.get('price_stats', {})
        latest_close = price_stats.get('latest_close', 0)
        st.metric("Close Price", f"${latest_close:.2f}")
    
    with col2:
        # Calculate how much the price is predicted to change
        if predictions and len(predictions) > 0 and historical_data:
            # Find the next day prediction (same logic as chart)
            df_pred = pd.DataFrame(predictions)
            df_pred['prediction_date'] = pd.to_datetime(df_pred['prediction_date'])
            
            # Get the latest historical date
            data_points = historical_data.get('data_points', [])
            if data_points:
                latest_hist_date = pd.to_datetime(data_points[-1]['date'])
                
                # Filter predictions to find the next day (closest future prediction)
                future_predictions = df_pred[df_pred['prediction_date'] > latest_hist_date]
                
                if not future_predictions.empty:
                    # Get the earliest future prediction (next day)
                    next_prediction = future_predictions.loc[future_predictions['prediction_date'].idxmin()]
                    predicted_close = next_prediction['predicted_close']
                    
                    if latest_close > 0 and predicted_close > 0:
                        estimated_change = ((predicted_close - latest_close) / latest_close) * 100
                        st.metric("Estimated Price Change", f"{estimated_change:.2f}%", f"{estimated_change:.2f}%")
                    else:
                        st.metric("Estimated Price Change", "N/A")
                else:
                    st.metric("Estimated Price Change", "N/A")
            else:
                st.metric("Estimated Price Change", "N/A")
        else:
            st.metric("Estimated Price Change", "N/A")
    
    with col3:
        news_summary = stock_detail.get('news_summary', {})
        latest_sentiment = news_summary.get('latest_sentiment', 0)
        st.metric("Latest Sentiment", f"{latest_sentiment:.3f}")
    
    st.markdown("---")
    
    # Show the price chart with prediction
    st.subheader("📈 Price Analysis & Next Day Prediction")
    
    if historical_data and predictions:
        data_points = historical_data.get('data_points', [])
        fig = create_price_chart(data_points, predictions, selected_symbol)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Show recent news articles with sentiment scores
    st.subheader("📰 Recent News & Sentiment")
    
    if news_data and news_data.get('news_items'):
        news_items = news_data.get('news_items', [])
        last_stock_date = news_data.get('last_stock_date', 'N/A')
        latest_news_date = news_data.get('latest_news_date', 'N/A')
        
        # Get the latest sentiment score
        latest_sentiment = news_data.get('latest_sentiment', 0)
        
        # Build date display text
        date_info = f"News from: {latest_news_date}"
        if last_stock_date != latest_news_date[:10]:  # Compare date part only
            date_info = f" Last Stock Date: {last_stock_date} | Latest News: {latest_news_date}"
        
        st.markdown(f"""
        <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <strong>{date_info}</strong><br>
            <strong>Latest Sentiment Score:</strong> {format_sentiment_score(latest_sentiment)}<br>
            <strong>News Count:</strong> {len(news_items)} articles
        </div>
        """, unsafe_allow_html=True)
        
        # Loop through all news articles and display them
        for news_item in news_items:
            sentiment_html = format_sentiment_score(news_item['sentiment_score'])
            news_url = news_item.get('news_url', '')
            
            # Build the news card with optional URL link
            news_summary = news_item['news_summary'][:200] if len(news_item['news_summary']) > 200 else news_item['news_summary']
            
            url_link = f'<br><a href="{news_url}" target="_blank">🔗 Read more</a>' if news_url else ''
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>📅 {news_item['date']}</strong><br>
                <strong>Sentiment:</strong> {sentiment_html}<br>
                <strong>Summary:</strong> {news_summary}...{url_link}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("No news available for this stock")

def main():
    """Main function that runs the whole dashboard"""
    
    # Sidebar navigation
    # Sidebar with navigation menu
    st.sidebar.title("🏠 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Stock Detail Analysis", "Dashboard"],
        index=0  # Start with detail view
    )
    
    # Check if backend API is running
    try:
        health_check = requests.get("http://localhost:8000/", timeout=5)
        if health_check.status_code == 200:
            st.sidebar.success(" API Connected")
        else:
            st.sidebar.error(" API Error")
    except:
        st.sidebar.error(" API Offline")
    
    # Option to auto-refresh the page every 30 seconds
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)")
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Button to manually refresh data
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()  # Clear cache to force fresh data
        st.rerun()
    
    # Display whichever page the user selected
    if page == "Dashboard":
        show_dashboard()
    elif page == "Stock Detail Analysis":
        show_stock_detail()

if __name__ == "__main__":
    main()