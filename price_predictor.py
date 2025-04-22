"""
Module for predicting stock prices using LSTM models
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import math

def create_sequences(data, seq_length):
    """
    Create sequences for LSTM model
    
    Args:
        data: Input time series data
        seq_length: Length of input sequence
        
    Returns:
        X (input sequences) and y (targets)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_predict_lstm(stock_data, prediction_days=7, seq_length=60):
    """
    Train LSTM model and make predictions
    
    Args:
        stock_data: DataFrame with stock prices
        prediction_days: Number of days to predict
        seq_length: Sequence length for LSTM
        
    Returns:
        predicted_prices, scaler, actual_test_data
    """
    # Import config parameters - import here to avoid circular imports
    from config import LSTM_UNITS, LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH_SIZE, TRAINING_SPLIT, SEQ_LENGTH
    
    # Use the SEQ_LENGTH from config if seq_length parameter is not provided
    seq_length = SEQ_LENGTH if seq_length is None else seq_length
    
    # Ensure we have numpy array
    if isinstance(stock_data, pd.DataFrame):
        data = stock_data.values
    else:
        data = stock_data
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Define train-test split point
    train_size = int(len(scaled_data) * TRAINING_SPLIT)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    # Ensure shapes are correct for LSTM
    # Reshape X_train and X_test to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Create and compile the model
    model = Sequential()
    model.add(LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(LSTM_DROPOUT))
    model.add(LSTM(units=LSTM_UNITS, return_sequences=False))
    model.add(Dropout(LSTM_DROPOUT))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Get the predicted values for the test set
    predicted_test = model.predict(X_test, verbose=0)
    
    # Generate prediction for future days
    future_predictions = []
    
    # Start with the last sequence from the original data
    current_batch = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    
    # Predict next 'prediction_days' days
    for _ in range(prediction_days):
        # Get prediction for next day
        current_pred = model.predict(current_batch, verbose=0)[0]
        
        # Append to our predictions
        future_predictions.append(current_pred)
        
        # Update batch to include our new prediction (moving forward one day)
        current_batch = np.append(current_batch[:, 1:, :], 
                                  [[current_pred]], 
                                  axis=1)
    
    future_predictions = np.array(future_predictions)
    
    return future_predictions, scaler, y_test

def evaluate_model(actual, predicted):
    """
    Evaluate model performance
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary with performance metrics
    """
    # Calculate Mean Squared Error
    mse = np.mean((actual - predicted) ** 2)
    
    # Calculate Root Mean Squared Error
    rmse = math.sqrt(mse)
    
    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Calculate a simplified "accuracy" metric (how close the prediction is)
    # This is a rough approximation - true prediction accuracy is complex for time series
    accuracy = max(0, 100 * (1 - np.mean(np.abs((actual - predicted) / actual))))
    
    return mse, rmse, mae, accuracy
