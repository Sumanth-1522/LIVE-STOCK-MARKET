"""
Configuration settings for the Stock Market Sentiment & Price Prediction Dashboard
"""

# Stock data settings
DEFAULT_STOCK = "AAPL"
TOP_STOCKS = [
    "AAPL", "MSFT", "AMZN", "TSLA", "GOOG", 
    "META", "NVDA", "BRK-A", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "BAC",
    "XOM", "PFE", "DIS", "NFLX", "INTC"
]

# Default values for date selection
DEFAULT_DAYS = 90
MAX_DAYS = 365
MIN_DAYS = 30

# Default values for prediction
DEFAULT_PREDICTION_DAYS = 7
MAX_PREDICTION_DAYS = 30
MIN_PREDICTION_DAYS = 1

# LSTM model parameters
LSTM_UNITS = 50
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
TRAINING_SPLIT = 0.8
SEQ_LENGTH = 60

# Sentiment analysis settings
SENTIMENT_THRESHOLD_POSITIVE = 0.6
SENTIMENT_THRESHOLD_NEGATIVE = 0.4
MAX_NEWS_ITEMS = 100

# Caching
CACHE_EXPIRY = 3600  # seconds (1 hour)

# Display settings
DATE_FORMAT = "%Y-%m-%d"
THEME_COLOR = "#FF4B4B"
