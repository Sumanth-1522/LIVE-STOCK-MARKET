"""
Data processor module for stock data collection and transformation.
This module handles automated data collection and preprocessing for Power BI integration.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import schedule
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "power_bi_data"
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
SECTOR_ETFS = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE']
SECTOR_NAMES = {
    'XLF': 'Financials',
    'XLK': 'Technology',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate'
}

def ensure_data_dir():
    """Ensure the data directory exists"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory: {DATA_DIR}")

def fetch_stock_data(ticker, days=365):
    """Fetch historical stock data for a specific ticker"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if stock_data.empty:
            logger.warning(f"No data found for {ticker}")
            return None
            
        # Calculate technical indicators
        # Moving Averages
        stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
        
        # MACD
        stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
        stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
        stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
        stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = stock_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        stock_data['BB_Middle'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['BB_Std'] = stock_data['Close'].rolling(window=20).std()
        stock_data['BB_Upper'] = stock_data['BB_Middle'] + (stock_data['BB_Std'] * 2)
        stock_data['BB_Lower'] = stock_data['BB_Middle'] - (stock_data['BB_Std'] * 2)
        
        # Calculate daily returns
        stock_data['Daily_Return'] = stock_data['Close'].pct_change() * 100
        
        # Volatility (20-day rolling standard deviation of returns)
        stock_data['Volatility'] = stock_data['Daily_Return'].rolling(window=20).std()
        
        logger.info(f"Successfully processed data for {ticker}")
        return stock_data
    
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def fetch_sector_data(days=30):
    """Fetch data for sector ETFs to analyze sector performance"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching sector data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        sector_data = []
        
        for ticker in SECTOR_ETFS:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                
                if not data.empty:
                    # Calculate performance metrics
                    first_price = data['Close'].iloc[0]
                    last_price = data['Close'].iloc[-1]
                    change = ((last_price - first_price) / first_price) * 100
                    
                    # Calculate volatility
                    returns = data['Close'].pct_change() * 100
                    volatility = returns.std()
                    
                    # Calculate volume trend
                    avg_volume = data['Volume'].mean()
                    recent_volume = data['Volume'].iloc[-5:].mean()
                    volume_trend = ((recent_volume / avg_volume) - 1) * 100
                    
                    sector_data.append({
                        'Ticker': ticker,
                        'Sector': SECTOR_NAMES.get(ticker, ticker),
                        'Change_Pct': change,
                        'Volatility': volatility,
                        'Volume_Trend': volume_trend,
                        'Last_Price': last_price
                    })
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error processing sector {ticker}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(sector_data)} sectors")
        return pd.DataFrame(sector_data)
    
    except Exception as e:
        logger.error(f"Error fetching sector data: {e}")
        return pd.DataFrame()

def save_to_json(data, filename):
    """Save DataFrame to JSON file"""
    try:
        full_path = os.path.join(DATA_DIR, filename)
        
        if isinstance(data, pd.DataFrame):
            if not data.empty:
                # Reset index to include date in the JSON output
                if isinstance(data.index, pd.DatetimeIndex):
                    data = data.reset_index()
                
                data.to_json(full_path, orient='records', date_format='iso')
                logger.info(f"Saved data to {full_path}")
                return True
            else:
                logger.warning(f"DataFrame is empty, not saving {filename}")
        else:
            logger.warning(f"Data is not a DataFrame, not saving {filename}")
        
        return False
    
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {e}")
        return False

def generate_market_summary():
    """Generate overall market summary"""
    try:
        # Use major indices as market indicators
        indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']
        index_names = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000'
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        market_data = []
        
        for index in indices:
            try:
                data = yf.download(index, start=start_date, end=end_date)
                
                if not data.empty:
                    # Calculate metrics
                    last_price = data['Close'].iloc[-1]
                    prev_day = data['Close'].iloc[-2]
                    day_change = ((last_price - prev_day) / prev_day) * 100
                    
                    week_ago = data['Close'].iloc[-6] if len(data) >= 6 else data['Close'].iloc[0]
                    week_change = ((last_price - week_ago) / week_ago) * 100
                    
                    month_ago = data['Close'].iloc[0]
                    month_change = ((last_price - month_ago) / month_ago) * 100
                    
                    # Calculate volatility
                    returns = data['Close'].pct_change() * 100
                    volatility = returns.std()
                    
                    market_data.append({
                        'Index': index_names.get(index, index),
                        'Last_Price': last_price,
                        'Day_Change': day_change,
                        'Week_Change': week_change,
                        'Month_Change': month_change,
                        'Volatility': volatility,
                        'Timestamp': end_date.strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error processing index {index}: {e}")
                continue
        
        logger.info(f"Generated market summary for {len(market_data)} indices")
        return pd.DataFrame(market_data)
    
    except Exception as e:
        logger.error(f"Error generating market summary: {e}")
        return pd.DataFrame()

def run_data_collection():
    """Run the data collection process for all tickers and sectors"""
    try:
        logger.info("Starting data collection process")
        
        # Ensure data directory exists
        ensure_data_dir()
        
        # Collect stock data for each ticker
        for ticker in DEFAULT_TICKERS:
            stock_data = fetch_stock_data(ticker)
            if stock_data is not None:
                save_to_json(stock_data, f"{ticker}_historical.json")
                
                # Get company info
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # Create a simplified info dictionary with relevant fields
                    simple_info = {
                        'Symbol': ticker,
                        'Name': info.get('shortName', ticker),
                        'Sector': info.get('sector', 'Unknown'),
                        'Industry': info.get('industry', 'Unknown'),
                        'Market_Cap': info.get('marketCap', 0),
                        'PE_Ratio': info.get('trailingPE', 0),
                        'Dividend_Yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                        'Beta': info.get('beta', 0),
                        'Year_High': info.get('fiftyTwoWeekHigh', 0),
                        'Year_Low': info.get('fiftyTwoWeekLow', 0),
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Save company info
                    pd.DataFrame([simple_info]).to_json(os.path.join(DATA_DIR, f"{ticker}_info.json"), orient='records')
                    logger.info(f"Saved company info for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error fetching company info for {ticker}: {e}")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
        
        # Collect sector data
        sector_data = fetch_sector_data()
        if not sector_data.empty:
            save_to_json(sector_data, "sector_performance.json")
        
        # Generate market summary
        market_summary = generate_market_summary()
        if not market_summary.empty:
            save_to_json(market_summary, "market_summary.json")
        
        logger.info("Data collection process completed")
        return True
    
    except Exception as e:
        logger.error(f"Error in data collection process: {e}")
        return False

def schedule_data_collection(interval_minutes=60):
    """Schedule regular data collection using the schedule library"""
    logger.info(f"Scheduling data collection every {interval_minutes} minutes")
    
    # Run immediately once
    run_data_collection()
    
    # Schedule regular runs
    schedule.every(interval_minutes).minutes.do(run_data_collection)
    
    logger.info("Starting scheduled data collection")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check schedule every minute
    except KeyboardInterrupt:
        logger.info("Data collection scheduler stopped by user")
    except Exception as e:
        logger.error(f"Error in data collection scheduler: {e}")

if __name__ == "__main__":
    # When run directly, start the scheduled data collection
    schedule_data_collection(60)  # Run every 60 minutes