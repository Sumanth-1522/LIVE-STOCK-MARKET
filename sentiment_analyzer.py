"""
Module for analyzing sentiment of news articles related to stocks
"""

import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from datetime import datetime
import time

# Download necessary NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    """
    Clean text for sentiment analysis
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove URLs, special characters, and numbers
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join back to string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def analyze_sentiment(news_items):
    """
    Analyze sentiment of news articles
    
    Args:
        news_items: List of news article dictionaries
        
    Returns:
        Pandas DataFrame with sentiment scores added
    """
    if not news_items:
        return pd.DataFrame()
    
    # Create dataframe from news items
    news_df = pd.DataFrame(news_items)
    
    # Initialize sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Clean text and analyze sentiment
    news_df['cleaned_text'] = news_df['summary'].apply(clean_text)
    
    # Apply sentiment analysis
    news_df['sentiment_scores'] = news_df['cleaned_text'].apply(lambda x: sid.polarity_scores(x))
    
    # Extract compound sentiment score
    news_df['sentiment'] = news_df['sentiment_scores'].apply(lambda x: x['compound'])
    
    # Normalize compound score to 0-1 range (from -1 to 1)
    news_df['sentiment'] = (news_df['sentiment'] + 1) / 2
    
    # Sort by date descending
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df = news_df.sort_values(by='date', ascending=False)
    
    return news_df

def get_sentiment_scores(sentiment_df):
    """
    Calculate aggregate sentiment scores
    
    Args:
        sentiment_df: DataFrame with sentiment scores
        
    Returns:
        Dictionary with aggregate sentiment metrics
    """
    # Import here to avoid circular imports
    from config import SENTIMENT_THRESHOLD_POSITIVE, SENTIMENT_THRESHOLD_NEGATIVE
    
    if sentiment_df.empty:
        return {
            'overall': 0.5,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'positive_pct': 0,
            'negative_pct': 0,
            'neutral_pct': 0
        }
    
    # Calculate sentiment stats
    total_articles = len(sentiment_df)
    positive_articles = sentiment_df[sentiment_df['sentiment'] >= SENTIMENT_THRESHOLD_POSITIVE]
    negative_articles = sentiment_df[sentiment_df['sentiment'] <= SENTIMENT_THRESHOLD_NEGATIVE]
    neutral_articles = sentiment_df[(sentiment_df['sentiment'] > SENTIMENT_THRESHOLD_NEGATIVE) & 
                                   (sentiment_df['sentiment'] < SENTIMENT_THRESHOLD_POSITIVE)]
    
    positive_count = len(positive_articles)
    negative_count = len(negative_articles)
    neutral_count = len(neutral_articles)
    
    # Calculate percentages
    positive_pct = (positive_count / total_articles) * 100 if total_articles > 0 else 0
    negative_pct = (negative_count / total_articles) * 100 if total_articles > 0 else 0
    neutral_pct = (neutral_count / total_articles) * 100 if total_articles > 0 else 0
    
    # Calculate overall sentiment (weighted average)
    overall_sentiment = sentiment_df['sentiment'].mean() if not sentiment_df.empty else 0.5
    
    return {
        'overall': overall_sentiment,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'positive_pct': positive_pct,
        'negative_pct': negative_pct,
        'neutral_pct': neutral_pct
    }
