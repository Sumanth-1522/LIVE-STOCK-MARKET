import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from datetime import datetime, timedelta
import time
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set dark theme for all Plotly figures
# Configure Plotly theme
pio.templates["custom_dark"] = pio.templates["plotly_dark"].update({
    "layout": {
        "paper_bgcolor": "#262730",
        "plot_bgcolor": "#262730",
        "font": {"color": "#FAFAFA"},
        "colorway": ["#FF4B4B", "#0068C9", "#29B09D", "#FFA15A", "#67E0E3", "#F06292"]
    }
})
pio.templates.default = "custom_dark"

# App configuration
st.set_page_config(
    page_title="Stock Market Analysis & Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define utility functions for data handling
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
        st.error(f"Error fetching stock data for {ticker}: {e}")
        return None

def get_stock_info(ticker):
    """Get general information about a stock"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        st.error(f"Error fetching stock info for {ticker}: {e}")
        return None

def predict_stock_prices(stock_data, days=7):
    """Simple linear regression prediction for stock prices"""
    # For demo purposes, using a very simple linear model
    # In a real application, you would use more sophisticated models like LSTM
    
    if stock_data is None or len(stock_data) < 30:
        return None
    
    # Use the last 30 days to predict the next 'days' days
    last_30_days = stock_data['Close'].values[-30:]
    
    # Simple linear model
    X = np.arange(len(last_30_days)).reshape(-1, 1)
    y = last_30_days
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 'days' days
    X_pred = np.arange(len(last_30_days), len(last_30_days) + days).reshape(-1, 1)
    predictions = model.predict(X_pred)
    
    return predictions

def get_buy_sell_signal(stock_data, predictions):
    """Generate buy/sell signal based on predictions"""
    if stock_data is None or predictions is None:
        return "HOLD", 0
    
    last_price = stock_data['Close'].iloc[-1]
    predicted_next_price = predictions[0]
    
    # Calculate percentage change
    percent_change = ((predicted_next_price - last_price) / last_price) * 100
    
    # Simple signal logic
    if percent_change > 2:
        return "BUY", percent_change
    elif percent_change < -2:
        return "SELL", percent_change
    else:
        return "HOLD", percent_change

def plot_stock_chart(stock_data, ticker):
    """Create an interactive stock chart with technical indicators"""
    # Create subplot with secondary y-axis for volume
    fig = go.Figure()
    
    # Add candlestick chart with enhanced colors
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name="Price",
            increasing=dict(line=dict(color='#26A69A'), fillcolor='#26A69A'),
            decreasing=dict(line=dict(color='#EF5350'), fillcolor='#EF5350')
        )
    )
    
    # Add moving averages with enhanced visibility
    if 'SMA_20' in stock_data.columns and 'SMA_50' in stock_data.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['SMA_20'],
                name="20-day MA",
                line=dict(color='#FFA15A', width=2.5)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['SMA_50'],
                name="50-day MA",
                line=dict(color='#29B09D', width=2.5)
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Price",
        height=600,
        xaxis_rangeslider_visible=False,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='custom_dark'
    )
    
    # Improve axis formatting
    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(128,128,128,0.3)',
        tickfont=dict(size=12),
        tickangle=-45,
        tickformat="%b %d\n%Y"
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(128,128,128,0.3)',
        tickprefix="$",
        tickfont=dict(size=12),
        tickformat=",.2f"
    )
    
    return fig

def plot_prediction_chart(stock_data, predictions):
    """Plot stock predictions vs actual prices"""
    if stock_data is None or predictions is None:
        return None
    
    # Get the last 30 days of actual data
    last_30_days = stock_data.iloc[-30:]
    
    # Create a date range for predictions
    last_date = stock_data.index[-1]
    prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions))
    
    # Create the plot
    fig = go.Figure()
    
    # Add actual price line with enhanced visibility
    fig.add_trace(
        go.Scatter(
            x=last_30_days.index,
            y=last_30_days['Close'],
            mode='lines',
            name='Actual',
            line=dict(color='#0068C9', width=2.5)
        )
    )
    
    # Add predicted price line with enhanced visibility
    fig.add_trace(
        go.Scatter(
            x=prediction_dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#FF4B4B', width=2.5, dash='dash'),
            marker=dict(size=8, symbol='circle')
        )
    )
    
    # Add a vertical line to separate actual from predicted
    fig.add_vline(
        x=last_date, 
        line_width=1, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="Prediction Start",
        annotation_position="top"
    )
    
    # Update layout
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='custom_dark',
        height=400
    )
    
    # Improve axis formatting
    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(128,128,128,0.3)',
        tickfont=dict(size=12),
        tickangle=-45,
        tickformat="%b %d\n%Y"
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(128,128,128,0.3)',
        tickprefix="$",
        tickfont=dict(size=12),
        tickformat=",.2f"
    )
    
    return fig

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
                    first_price = float(data['Close'].iloc[0])
                    last_price = float(data['Close'].iloc[-1])
                    change = ((last_price - first_price) / first_price) * 100
                    sector_data.append({
                        'Sector': sector_names.get(ticker, ticker),
                        'Change': change
                    })
            except:
                continue
        
        return pd.DataFrame(sector_data)
    except Exception as e:
        st.error(f"Error getting sector performance: {e}")
        return pd.DataFrame()

def save_to_json(data, filename):
    """Save data to JSON for Power BI consumption"""
    try:
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to JSON
            if not data.empty:
                data.to_json(filename, orient='records', date_format='iso')
                return True
        return False
    except Exception as e:
        st.error(f"Error saving data to JSON: {e}")
        return False

# Main app
def main():
    # Sidebar
    st.sidebar.title("Stock Market Dashboard")
    st.sidebar.markdown("Configure your dashboard view")
    
    # Stock selection
    default_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
    selected_ticker = st.sidebar.selectbox("Select a stock", default_tickers)
    
    # Date range selection
    date_ranges = {
        "1 Week": 7,
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "2 Years": 730
    }
    selected_range = st.sidebar.selectbox("Select date range", list(date_ranges.keys()))
    days = date_ranges[selected_range]
    
    # Prediction period
    prediction_days = st.sidebar.slider("Prediction days", 1, 30, 7)
    
    # Data export option (for Power BI)
    if st.sidebar.checkbox("Export data for Power BI", key="export_option"):
        export_dir = "power_bi_data"
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        st.sidebar.info(f"Data will be exported to the {export_dir} directory")
    
    # Fetch data
    st.title("Stock Market Analysis & Prediction Dashboard")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    with st.spinner(f"Fetching data for {selected_ticker}..."):
        # Get stock data
        stock_data = fetch_stock_data(selected_ticker, start_date, end_date)
        
        if stock_data is None or stock_data.empty:
            st.error(f"Could not retrieve data for {selected_ticker}")
            return
        
        # Get stock info
        stock_info = get_stock_info(selected_ticker)
    
    # Main dashboard layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stock price chart
        fig = plot_stock_chart(stock_data, selected_ticker)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stock info and metrics
        if stock_info:
            st.subheader(f"{stock_info.get('shortName', selected_ticker)}")
            st.markdown(f"**Sector:** {stock_info.get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {stock_info.get('industry', 'N/A')}")
            
            # Current metrics
            # Extract scalar values from Series to avoid pandas Series truth value ambiguity
            current_price = float(stock_data['Close'].iloc[-1])
            prev_close = float(stock_data['Close'].iloc[-2]) if len(stock_data) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
            
            # Display metrics
            col_a, col_b = st.columns(2)
            col_a.metric("Current Price", f"${current_price:.2f}", f"{change_pct:.2f}%")
            col_b.metric("Volume", f"{stock_info.get('volume', 'N/A'):,}")
            
            col_c, col_d = st.columns(2)
            col_c.metric("Market Cap", f"${stock_info.get('marketCap', 0)/1e9:.2f}B")
            col_d.metric("52-Week Range", f"${stock_info.get('fiftyTwoWeekLow', 0):.2f} - ${stock_info.get('fiftyTwoWeekHigh', 0):.2f}")
        
        # Generate predictions
        predictions = predict_stock_prices(stock_data, prediction_days)
        signal, percent_change = get_buy_sell_signal(stock_data, predictions)
        
        # Signal display
        signal_color = "green" if signal == "BUY" else "red" if signal == "SELL" else "gray"
        st.markdown(f"""
        <div style="background-color: {signal_color}; padding: 10px; border-radius: 5px; text-align: center; margin-top: 20px;">
            <h3 style="color: white; margin: 0;">Signal: {signal}</h3>
            <p style="color: white; margin: 0;">Predicted change: {percent_change:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Predictions section
    st.subheader(f"Price Predictions (Next {prediction_days} days)")
    pred_fig = plot_prediction_chart(stock_data, predictions)
    if pred_fig:
        st.plotly_chart(pred_fig, use_container_width=True)
    
    # Sector performance
    st.subheader("Sector Performance (7-day)")
    sector_df = get_sector_performance()
    
    if not sector_df.empty:
        # Sector performance chart
        fig = px.bar(
            sector_df,
            x='Sector',
            y='Change',
            title='7-Day Sector Performance',
            color='Change',
            color_continuous_scale=['red', 'yellow', 'green'],
            labels={'Change': '% Change'},
            range_color=[-5, 5]
        )
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        fig.update_layout(
            height=400,
            template='custom_dark'
        )
        
        # Improve axis formatting
        fig.update_xaxes(
            tickfont=dict(size=12),
            tickangle=-45
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(128,128,128,0.3)',
            ticksuffix="%",
            tickfont=dict(size=12),
            tickformat=".1f"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Export data for Power BI if selected
    if st.sidebar.checkbox("Export data for Power BI", key="export_action"):
        export_dir = "power_bi_data"
        
        # Export stock data
        stock_data_export = stock_data.reset_index()
        save_to_json(stock_data_export, f"{export_dir}/{selected_ticker}_data.json")
        
        # Export predictions
        if predictions is not None:
            last_date = stock_data.index[-1]
            prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions))
            predictions_df = pd.DataFrame({
                'Date': prediction_dates,
                'Predicted_Price': predictions
            })
            save_to_json(predictions_df, f"{export_dir}/{selected_ticker}_predictions.json")
        
        # Export sector data
        if not sector_df.empty:
            save_to_json(sector_df, f"{export_dir}/sector_performance.json")
        
        st.sidebar.success("Data exported successfully")
    
    st.markdown("---")
    st.caption("Data source: Yahoo Finance | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.caption("⚠️ Disclaimer: This tool is for informational purposes only. Not financial advice.")

if __name__ == "__main__":
    main()