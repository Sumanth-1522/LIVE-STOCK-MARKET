"""
Machine learning models for stock price prediction.
This module provides different predictive models for stock price forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import joblib
import os
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MODELS_DIR = "models"

def ensure_models_dir():
    """Ensure the models directory exists"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logger.info(f"Created models directory: {MODELS_DIR}")

class LinearRegressionModel:
    """Simple linear regression model for stock price prediction"""
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.model = LinearRegression()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
    
    def prepare_data(self, stock_data, sequence_length=30):
        """Prepare data for linear regression model"""
        try:
            # Extract closing prices
            close_prices = stock_data['Close'].values.reshape(-1, 1)
            
            # Normalize the data
            scaled_data = self.scaler.fit_transform(close_prices)
            
            # Create features (X) as the sequence of days and target (y) as the next day price
            X = np.arange(len(scaled_data)).reshape(-1, 1)
            y = scaled_data
            
            return X, y, scaled_data
        
        except Exception as e:
            logger.error(f"Error preparing data for {self.ticker}: {e}")
            return None, None, None
    
    def train(self, stock_data):
        """Train the linear regression model"""
        try:
            logger.info(f"Training linear regression model for {self.ticker}")
            
            X, y, scaled_data = self.prepare_data(stock_data)
            
            if X is None or y is None:
                return False
            
            # Fit the model
            self.model.fit(X, y)
            
            # Calculate training metrics
            y_pred = self.model.predict(X)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            logger.info(f"Model trained with MSE: {mse:.6f}, MAE: {mae:.6f}, R2: {r2:.6f}")
            
            self.is_trained = True
            return True
        
        except Exception as e:
            logger.error(f"Error training model for {self.ticker}: {e}")
            return False
    
    def predict(self, stock_data, days=7):
        """Predict future stock prices"""
        try:
            if not self.is_trained:
                logger.warning(f"Model for {self.ticker} is not trained yet")
                return None
            
            # Get the last day index
            X, _, scaled_data = self.prepare_data(stock_data)
            
            if X is None:
                return None
            
            last_idx = X[-1, 0]
            
            # Create prediction indices
            X_pred = np.arange(last_idx + 1, last_idx + days + 1).reshape(-1, 1)
            
            # Predict scaled values
            scaled_predictions = self.model.predict(X_pred)
            
            # Inverse transform to get actual price predictions
            predictions = self.scaler.inverse_transform(scaled_predictions)
            
            return predictions.flatten()
        
        except Exception as e:
            logger.error(f"Error making predictions for {self.ticker}: {e}")
            return None
    
    def evaluate(self, stock_data, test_size=0.2):
        """Evaluate model on test data"""
        try:
            X, y, scaled_data = self.prepare_data(stock_data)
            
            if X is None or y is None:
                return None
            
            # Split into train/test
            train_size = int(len(X) * (1 - test_size))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train on training data
            self.model.fit(X_train, y_train)
            
            # Predict on test data
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate accuracy as a percentage (simplified approach)
            accuracy = max(0, 100 * (1 - np.sqrt(mse)))
            
            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2,
                'accuracy': accuracy
            }
        
        except Exception as e:
            logger.error(f"Error evaluating model for {self.ticker}: {e}")
            return None
    
    def save(self):
        """Save the trained model"""
        try:
            if not self.is_trained:
                logger.warning(f"Cannot save untrained model for {self.ticker}")
                return False
            
            ensure_models_dir()
            
            # Save the model
            model_path = os.path.join(MODELS_DIR, f"{self.ticker}_linear_model.joblib")
            joblib.dump(self.model, model_path)
            
            # Save the scaler
            scaler_path = os.path.join(MODELS_DIR, f"{self.ticker}_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Model saved at {model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving model for {self.ticker}: {e}")
            return False
    
    def load(self):
        """Load a previously trained model"""
        try:
            model_path = os.path.join(MODELS_DIR, f"{self.ticker}_linear_model.joblib")
            scaler_path = os.path.join(MODELS_DIR, f"{self.ticker}_scaler.joblib")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                logger.info(f"Model loaded from {model_path}")
                return True
            else:
                logger.warning(f"No saved model found for {self.ticker}")
                return False
        
        except Exception as e:
            logger.error(f"Error loading model for {self.ticker}: {e}")
            return False

def train_all_models(tickers, stock_data_dict):
    """Train linear regression models for all tickers"""
    try:
        logger.info(f"Training models for {len(tickers)} tickers")
        
        results = {}
        
        for ticker in tickers:
            if ticker in stock_data_dict and not stock_data_dict[ticker].empty:
                model = LinearRegressionModel(ticker)
                
                # Train the model
                success = model.train(stock_data_dict[ticker])
                
                if success:
                    # Evaluate the model
                    metrics = model.evaluate(stock_data_dict[ticker])
                    
                    # Save the model
                    model.save()
                    
                    results[ticker] = {
                        'success': True,
                        'metrics': metrics
                    }
                else:
                    results[ticker] = {
                        'success': False
                    }
            else:
                results[ticker] = {
                    'success': False,
                    'error': 'No data available'
                }
        
        return results
    
    except Exception as e:
        logger.error(f"Error training all models: {e}")
        return {}

def predict_all(tickers, stock_data_dict, days=7):
    """Generate predictions for all tickers"""
    try:
        logger.info(f"Generating predictions for {len(tickers)} tickers")
        
        predictions = {}
        
        for ticker in tickers:
            try:
                if ticker in stock_data_dict and not stock_data_dict[ticker].empty:
                    model = LinearRegressionModel(ticker)
                    
                    # Try to load the model first
                    if not model.load():
                        # If loading fails, train a new model
                        model.train(stock_data_dict[ticker])
                    
                    # Generate predictions
                    pred_values = model.predict(stock_data_dict[ticker], days)
                    
                    if pred_values is not None:
                        # Create date range for predictions
                        last_date = stock_data_dict[ticker].index[-1]
                        pred_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(pred_values))]
                        
                        ticker_predictions = []
                        for date, price in zip(pred_dates, pred_values):
                            ticker_predictions.append({
                                'Date': date,
                                'Predicted_Price': float(price)
                            })
                        
                        predictions[ticker] = ticker_predictions
                else:
                    logger.warning(f"No data available for {ticker}")
            except Exception as e:
                logger.error(f"Error predicting for {ticker}: {e}")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error generating all predictions: {e}")
        return {}