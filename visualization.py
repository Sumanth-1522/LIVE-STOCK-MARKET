"""
Visualization functions for the stock market dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def plot_stock_chart(stock_data, ticker):
    """
    Create an interactive stock chart with technical indicators
    
    Args:
        stock_data: DataFrame with stock data
        ticker: Stock ticker symbol
        
    Returns:
        Plotly figure
    """
    # Create subplot with secondary y-axis for volume
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} Stock Price", "Volume")
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if 'SMA_20' in stock_data.columns and 'SMA_50' in stock_data.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['SMA_20'],
                name="20-day MA",
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['SMA_50'],
                name="50-day MA",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            name="Volume",
            marker=dict(color='rgba(0, 0, 255, 0.3)')
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def plot_prediction_vs_actual(actual_dates, actual_values, predicted_values, future_start_date, prediction_days):
    """
    Plot actual vs predicted stock prices
    
    Args:
        actual_dates: Dates for actual values
        actual_values: Actual stock values
        predicted_values: Predicted stock values
        future_start_date: Start date for future predictions
        prediction_days: Number of days predicted
        
    Returns:
        Plotly figure
    """
    # Generate future dates
    future_dates = [future_start_date + timedelta(days=i) for i in range(prediction_days)]
    
    # Create figure
    fig = go.Figure()
    
    # Add actual price line
    fig.add_trace(
        go.Scatter(
            x=actual_dates,
            y=actual_values.flatten(),
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        )
    )
    
    # Add predicted price line
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predicted_values.flatten(),
            mode='lines',
            name='Predicted',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        height=400
    )
    
    # Add a vertical line to separate actual from predicted
    fig.add_vline(
        x=future_dates[0], 
        line_width=1, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="Prediction Start",
        annotation_position="top"
    )
    
    return fig

def plot_sentiment_analysis(sentiment_df):
    """
    Plot sentiment analysis results
    
    Args:
        sentiment_df: DataFrame with sentiment data
        
    Returns:
        Plotly figure
    """
    if sentiment_df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title='No Sentiment Data Available',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            height=400
        )
        return fig
    
    # Import config values
    from config import SENTIMENT_THRESHOLD_POSITIVE, SENTIMENT_THRESHOLD_NEGATIVE
    
    # Ensure dataframe is sorted by date
    sentiment_df = sentiment_df.sort_values(by='date')
    
    # Create date bins for aggregation
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    # Group by date and calculate average sentiment
    daily_sentiment = sentiment_df.groupby(sentiment_df['date'].dt.date)['sentiment'].mean().reset_index()
    
    # Count articles per day
    daily_counts = sentiment_df.groupby(sentiment_df['date'].dt.date).size().reset_index(name='count')
    
    # Merge sentiment and counts
    daily_data = pd.merge(daily_sentiment, daily_counts, on='date')
    
    # Create figure with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add sentiment line
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['sentiment'],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='blue'),
            marker=dict(size=daily_data['count'] * 2, color='blue')
        ),
        secondary_y=False
    )
    
    # Add article count bars
    fig.add_trace(
        go.Bar(
            x=daily_data['date'],
            y=daily_data['count'],
            name='Article Count',
            marker=dict(color='rgba(0, 0, 255, 0.3)')
        ),
        secondary_y=True
    )
    
    # Add horizontal threshold lines
    fig.add_hline(
        y=SENTIMENT_THRESHOLD_POSITIVE,
        line_width=1,
        line_dash="dash",
        line_color="green",
        secondary_y=False
    )
    
    fig.add_hline(
        y=SENTIMENT_THRESHOLD_NEGATIVE,
        line_width=1,
        line_dash="dash",
        line_color="red",
        secondary_y=False
    )
    
    # Update layout
    fig.update_layout(
        title='News Sentiment Analysis',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        height=400
    )
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="Article Count", secondary_y=True)
    
    return fig

def plot_accuracy_metrics(actual_values, predicted_values, metrics):
    """
    Plot accuracy metrics for predictions
    
    Args:
        actual_values: Actual values
        predicted_values: Predicted values
        metrics: Dictionary with accuracy metrics
        
    Returns:
        Plotly figure
    """
    # Create figure with gauge charts
    fig = make_subplots(
        rows=1, 
        cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Add accuracy gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metrics['accuracy'],
            title={'text': "Prediction Accuracy"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "red"},
                       {'range': [50, 80], 'color': "orange"},
                       {'range': [80, 100], 'color': "green"}
                   ]},
            domain={'row': 0, 'column': 0}
        ),
        row=1, col=1
    )
    
    # Add error gauge (RMSE)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metrics['rmse'],
            title={'text': "Root Mean Square Error"},
            gauge={'axis': {'range': [0, max(metrics['rmse']*2, 0.1)]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, metrics['rmse']/3], 'color': "green"},
                       {'range': [metrics['rmse']/3, metrics['rmse']*1.5], 'color': "orange"},
                       {'range': [metrics['rmse']*1.5, metrics['rmse']*2], 'color': "red"}
                   ]},
            domain={'row': 0, 'column': 1}
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=300,
        template='plotly_white'
    )
    
    return fig
