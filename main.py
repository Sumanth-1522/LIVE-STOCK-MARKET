"""
Main entry point for the Stock Market Analysis & Prediction system.
This script orchestrates the different components of the system.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os
import time
import argparse
import subprocess
import threading

# Import custom modules
from data_processor import run_data_collection, schedule_data_collection
from ml_models import LinearRegressionModel, train_all_models, predict_all
import api

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
DATA_DIR = "power_bi_data"
MODELS_DIR = "models"

def ensure_directories():
    """Ensure all required directories exist"""
    for directory in [DATA_DIR, MODELS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def fetch_data_for_tickers(tickers, days=365):
    """Fetch stock data for all tickers"""
    try:
        logger.info(f"Fetching data for {len(tickers)} tickers")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock_data_dict = {}
        
        for ticker in tickers:
            try:
                logger.info(f"Fetching data for {ticker}")
                stock_data = yf.download(ticker, start=start_date, end=end_date)
                
                if not stock_data.empty:
                    stock_data_dict[ticker] = stock_data
                    logger.info(f"Successfully fetched data for {ticker}")
                else:
                    logger.warning(f"No data found for {ticker}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
        
        return stock_data_dict
    
    except Exception as e:
        logger.error(f"Error fetching data for tickers: {e}")
        return {}

def run_streamlit_dashboard():
    """Run the Streamlit dashboard"""
    try:
        logger.info("Starting Streamlit dashboard")
        subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "5000"])
    except Exception as e:
        logger.error(f"Error starting Streamlit dashboard: {e}")

def run_flask_api():
    """Run the Flask API server"""
    try:
        logger.info("Starting Flask API server")
        api.app.run(host='0.0.0.0', port=5001)
    except Exception as e:
        logger.error(f"Error starting Flask API: {e}")

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='Stock Market Analysis & Prediction System')
    parser.add_argument('--dashboard', action='store_true', help='Run the Streamlit dashboard')
    parser.add_argument('--api', action='store_true', help='Run the Flask API server')
    parser.add_argument('--collector', action='store_true', help='Run the data collector')
    parser.add_argument('--train', action='store_true', help='Train prediction models')
    parser.add_argument('--all', action='store_true', help='Run all components')
    parser.add_argument('--interval', type=int, default=60, help='Data collection interval in minutes')
    
    args = parser.parse_args()
    
    # Ensure all directories exist
    ensure_directories()
    
    # Determine what to run
    run_dashboard = args.dashboard or args.all
    run_api_server = args.api or args.all
    run_collector = args.collector or args.all
    run_training = args.train or args.all
    
    # Run the requested components
    if run_collector:
        # Start data collection in a separate thread
        collector_thread = threading.Thread(
            target=schedule_data_collection,
            args=(args.interval,)
        )
        collector_thread.daemon = True
        collector_thread.start()
        logger.info(f"Data collector scheduled to run every {args.interval} minutes")
    
    if run_training:
        # Fetch data if needed
        stock_data_dict = fetch_data_for_tickers(DEFAULT_TICKERS)
        
        if stock_data_dict:
            # Train models
            results = train_all_models(DEFAULT_TICKERS, stock_data_dict)
            logger.info(f"Training results: {results}")
            
            # Generate predictions
            predictions = predict_all(DEFAULT_TICKERS, stock_data_dict)
            logger.info(f"Generated predictions for {len(predictions)} tickers")
    
    if run_dashboard:
        # Start the Streamlit dashboard
        run_streamlit_dashboard()
    
    if run_api_server:
        # Start the Flask API server
        # Note: This will block the main thread
        run_flask_api()
    elif run_collector:
        # If only running the collector (and not the API which blocks),
        # keep the main thread alive
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Application stopped by user")

if __name__ == "__main__":
    main()