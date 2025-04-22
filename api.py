"""
API endpoints to serve stock data to Power BI
"""

from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import os
import json

app = Flask(__name__)

# Helper functions
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data for a specific ticker within a date range"""
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

def predict_stock_prices(stock_data, days=7):
    """Simple linear regression prediction for stock prices"""
    if stock_data is None or len(stock_data) < 30:
        return None
    
    # Use the last 30 days to predict the next 'days' days
    last_30_days = stock_data['Close'].values[-30:]
    
    # Simple linear model
    X = np.arange(len(last_30_days)).reshape(-1, 1)
    y = last_30_days
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 'days' days
    X_pred = np.arange(len(last_30_days), len(last_30_days) + days).reshape(-1, 1)
    predictions = model.predict(X_pred)
    
    return predictions

def get_sector_performance():
    """Get overall sector performance"""
    try:
        # Sample sectors
        sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE']
        sector_names = {
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
        
        # Get data for the last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        sector_data = []
        for ticker in sectors:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                if not data.empty:
                    first_price = data['Close'].iloc[0]
                    last_price = data['Close'].iloc[-1]
                    change = ((last_price - first_price) / first_price) * 100
                    sector_data.append({
                        'Sector': sector_names.get(ticker, ticker),
                        'Change': change
                    })
            except:
                continue
        
        return pd.DataFrame(sector_data)
    except Exception as e:
        print(f"Error getting sector performance: {e}")
        return pd.DataFrame()

# API Routes
@app.route('/')
def home():
    return jsonify({
        'message': 'Stock Market API for Power BI Integration',
        'endpoints': [
            '/api/stock/<ticker>',
            '/api/predict/<ticker>',
            '/api/sectors'
        ]
    })

@app.route('/api/stock/<ticker>')
def get_stock(ticker):
    # Get query parameters
    days = request.args.get('days', default=30, type=int)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    if stock_data is None:
        return jsonify({'error': f'Could not retrieve data for {ticker}'}), 404
    
    # Convert DataFrame to JSON
    stock_data_json = stock_data.reset_index().to_dict(orient='records')
    
    return jsonify({
        'ticker': ticker,
        'data': stock_data_json
    })

@app.route('/api/predict/<ticker>')
def predict_stock(ticker):
    # Get query parameters
    days = request.args.get('days', default=7, type=int)
    history_days = request.args.get('history', default=90, type=int)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=history_days)
    
    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    if stock_data is None:
        return jsonify({'error': f'Could not retrieve data for {ticker}'}), 404
    
    # Generate predictions
    predictions = predict_stock_prices(stock_data, days)
    
    if predictions is None:
        return jsonify({'error': 'Could not generate predictions'}), 400
    
    # Create date range for predictions
    last_date = stock_data.index[-1]
    prediction_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(predictions))]
    
    # Format predictions as JSON
    prediction_data = []
    for i, (date, price) in enumerate(zip(prediction_dates, predictions)):
        prediction_data.append({
            'date': date,
            'predicted_price': float(price)
        })
    
    return jsonify({
        'ticker': ticker,
        'predictions': prediction_data
    })

@app.route('/api/sectors')
def sector_data():
    # Get sector performance
    sector_df = get_sector_performance()
    
    if sector_df.empty:
        return jsonify({'error': 'Could not retrieve sector data'}), 404
    
    # Convert DataFrame to JSON
    sector_data_json = sector_df.to_dict(orient='records')
    
    return jsonify({
        'sectors': sector_data_json
    })

# Data export endpoint for Power BI
@app.route('/api/export')
def export_data():
    export_dir = "power_bi_data"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    # Get query parameters
    tickers = request.args.get('tickers', default='AAPL,MSFT,GOOG,AMZN', type=str)
    days = request.args.get('days', default=30, type=int)
    
    ticker_list = tickers.split(',')
    
    result = {
        'exported_files': []
    }
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Export stock data for each ticker
    for ticker in ticker_list:
        # Fetch stock data
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        
        if stock_data is not None:
            # Export stock data
            stock_filename = f"{export_dir}/{ticker}_data.json"
            stock_data_export = stock_data.reset_index()
            stock_data_export.to_json(stock_filename, orient='records', date_format='iso')
            result['exported_files'].append(stock_filename)
            
            # Generate and export predictions
            predictions = predict_stock_prices(stock_data)
            if predictions is not None:
                last_date = stock_data.index[-1]
                prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions))
                predictions_df = pd.DataFrame({
                    'Date': prediction_dates,
                    'Predicted_Price': predictions
                })
                pred_filename = f"{export_dir}/{ticker}_predictions.json"
                predictions_df.to_json(pred_filename, orient='records', date_format='iso')
                result['exported_files'].append(pred_filename)
    
    # Export sector data
    sector_df = get_sector_performance()
    if not sector_df.empty:
        sector_filename = f"{export_dir}/sector_performance.json"
        sector_df.to_json(sector_filename, orient='records')
        result['exported_files'].append(sector_filename)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)