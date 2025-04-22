"""
Utility functions for the stock market dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import functools
import yfinance as yf

def calculate_signals(stock_data, predicted_prices, scaler):
    """
    Calculate buy/sell signals based on predicted prices
    
    Args:
        stock_data: DataFrame with historical stock data
        predicted_prices: Numpy array with predicted prices (scaled)
        scaler: Scaler used to normalize the data
        
    Returns:
        DataFrame with signals
    """
    # Create a copy of the stock data
    signals = stock_data[['Close']].copy()
    
    # Get the actual closing prices for the last few days
    last_prices = signals['Close'].values[-5:]
    
    # Transform predictions back to original scale
    predicted = scaler.inverse_transform(predicted_prices)
    
    # Calculate short-term trend (last 5 days)
    short_term_trend = np.polyfit(range(len(last_prices)), last_prices, 1)[0]
    
    # Calculate predicted trend
    predicted_trend = predicted[1][0] - predicted[0][0] if len(predicted) > 1 else 0
    
    # Initialize signal column
    signals['signal'] = 0
    
    # Generate signals based on predicted values and trends
    current_price = signals['Close'].iloc[-1]
    next_price = predicted[0][0]
    
    # Simple signal logic based on price movement and trends
    if next_price > current_price * 1.02 and predicted_trend > 0:
        # Strong buy signal: predicted price up >2% and positive trend
        signals.loc[signals.index[-1], 'signal'] = 1
    elif next_price < current_price * 0.98 and predicted_trend < 0:
        # Strong sell signal: predicted price down >2% and negative trend
        signals.loc[signals.index[-1], 'signal'] = -1
    else:
        # Hold signal
        signals.loc[signals.index[-1], 'signal'] = 0
    
    return signals

def calculate_sector_performance(ticker, lookback_days=7):
    """
    Calculate performance of stocks in the same sector
    
    Args:
        ticker: Stock ticker symbol
        lookback_days: Number of days to look back for performance calculation
        
    Returns:
        DataFrame with sector performance data
    """
    # Import here to avoid circular imports
    from data_collector import get_sector_stocks
    
    try:
        # Get stocks in the same sector
        sector_stocks = get_sector_stocks(ticker)
        
        if not sector_stocks:
            return None
        
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 5)  # Add buffer days
        
        # Get performance data for each stock
        performance_data = []
        
        for stock_info in sector_stocks:
            try:
                stock_symbol = stock_info['symbol']
                stock_data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
                
                if not stock_data.empty and len(stock_data) > 1:
                    # Calculate performance (percentage change over the period)
                    first_price = stock_data['Close'].iloc[0]
                    last_price = stock_data['Close'].iloc[-1]
                    perf = ((last_price - first_price) / first_price) * 100
                    
                    performance_data.append({
                        'symbol': stock_symbol,
                        'sector': stock_info['sector'],
                        'performance': perf,
                        'last_price': last_price
                    })
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.2)
            except Exception as e:
                print(f"Error getting performance for {stock_info['symbol']}: {e}")
                continue
        
        if not performance_data:
            return None
        
        # Convert to DataFrame and sort by performance
        perf_df = pd.DataFrame(performance_data)
        perf_df = perf_df.sort_values(by='performance', ascending=False)
        
        return perf_df
    except Exception as e:
        print(f"Error calculating sector performance: {e}")
        return None

def cache_data(func):
    """
    Decorator for caching function results
    
    Args:
        func: Function to cache
        
    Returns:
        Cached function
    """
    # Import here to avoid circular imports
    from config import CACHE_EXPIRY
    
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key from function name and arguments
        key = str(func.__name__) + str(args) + str(kwargs)
        
        # Check if result is in cache and not expired
        current_time = time.time()
        if key in cache and (current_time - cache[key]['timestamp']) < CACHE_EXPIRY:
            return cache[key]['data']
        
        # Call the function and cache the result
        result = func(*args, **kwargs)
        cache[key] = {
            'data': result,
            'timestamp': current_time
        }
        
        return result
    
    return wrapper
