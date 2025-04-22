"""
Module for collecting stock data, news, and related market information
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data for a specific ticker within a date range
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data collection
        end_date: End date for data collection
        
    Returns:
        Pandas DataFrame with stock data or None if error
    """
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            return None
        
        # Calculate additional technical indicators
        # Simple Moving Averages
        stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
        
        # Calculate MACD
        stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
        stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
        stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
        stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate RSI
        delta = stock_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))
        
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None

def fetch_company_news(ticker, max_items=20):
    """
    Fetch news articles for a company
    
    Args:
        ticker: Stock ticker symbol
        max_items: Maximum number of news items to return
        
    Returns:
        List of news items with title, summary, date, url, and source
    """
    try:
        # Using Yahoo Finance for news
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return []
        
        parsed_news = []
        for item in news[:max_items]:  # Limit to max_items
            # Convert timestamp to datetime
            news_date = datetime.fromtimestamp(item.get('providerPublishTime', 0))
            
            parsed_news.append({
                'title': item.get('title', ''),
                'summary': item.get('summary', ''),
                'date': news_date.strftime('%Y-%m-%d'),
                'url': item.get('link', ''),
                'source': item.get('publisher', '')
            })
        
        return parsed_news
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

def get_stock_info(ticker):
    """
    Get general information about a stock
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        print(f"Error fetching stock info for {ticker}: {e}")
        return None

def get_available_tickers():
    """
    Get a list of available stock tickers for the dashboard
    
    Returns:
        List of ticker symbols
    """
    # Importing here to avoid circular imports
    from config import TOP_STOCKS
    
    # For now, return a predefined list of top stocks
    return TOP_STOCKS

def get_sector_stocks(ticker):
    """
    Get stocks in the same sector as the given ticker
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        List of ticker symbols in the same sector
    """
    try:
        # Get the sector of the current stock
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', None)
        
        if not sector:
            return []
        
        # For now, return a subset of stocks from the top stocks list that might be in the same sector
        # In a real application, you would query a proper database of stocks by sector
        from config import TOP_STOCKS
        
        # Get info for a sample of stocks to find those in the same sector
        sector_stocks = []
        for sample_ticker in TOP_STOCKS:
            try:
                if sample_ticker != ticker:  # Skip the original ticker
                    sample_stock = yf.Ticker(sample_ticker)
                    sample_info = sample_stock.info
                    if sample_info.get('sector') == sector:
                        sector_stocks.append({
                            'symbol': sample_ticker,
                            'sector': sector
                        })
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.2)
                    
                    # Limit to 5 stocks in the sector
                    if len(sector_stocks) >= 5:
                        break
            except:
                continue
        
        # Always include the original ticker
        sector_stocks.append({
            'symbol': ticker,
            'sector': sector
        })
        
        return sector_stocks
    except Exception as e:
        print(f"Error finding sector stocks for {ticker}: {e}")
        return []
