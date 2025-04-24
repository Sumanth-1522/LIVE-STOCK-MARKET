# Stock Market Analysis & Prediction Dashboard

This project implements a comprehensive stock market analytics system with live data updates, price predictions, and integration with Power BI. It combines Python data processing with interactive visualizations to provide valuable insights for stock market analysis.

## Features

- **Live Stock Data**: Fetches real-time and historical stock data from Yahoo Finance
- **Technical Indicators**: Calculates MACD, RSI, SMA, Bollinger Bands, and more
- **Price Predictions**: Uses machine learning models to forecast future stock prices
- **Sector Analysis**: Monitors performance across different market sectors
- **Interactive Dashboard**: Offers a Streamlit-based visualization interface
- **API Endpoints**: Provides REST API endpoints for Power BI integration
- **Data Export**: Exports processed data in formats compatible with Power BI
- **Automated Updates**: Schedules regular data refreshes and model retraining

## Project Structure

- `app.py`: Streamlit dashboard for interactive visualization
- `api.py`: Flask API serving data for Power BI integration
- `data_processor.py`: Data collection and preprocessing module
- `ml_models.py`: Machine learning models for price prediction
- `main.py`: Main entry point orchestrating all components
- `power_bi_data/`: Directory for exported data files
- `models/`: Directory for trained ML models

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (see below)
- Power BI Desktop (optional, for dashboard integration)

### Installation

1. Clone this repository
2. Install the required packages:

```
pip install pandas numpy yfinance streamlit flask plotly scikit-learn joblib schedule
```

### Running the Application

#### Streamlit Dashboard

```
streamlit run app.py --server.port 5000
```

#### Flask API Server

```
python api.py
```

#### Data Collection

```
python main.py --collector --interval 60
```

#### Run All Components

```
python main.py --all
```

## Power BI Integration

1. Export data using the export functionality in the Streamlit dashboard or by calling the API endpoint:
   ```
   GET /api/export?tickers=AAPL,MSFT,GOOG&days=30
   ```

2. In Power BI Desktop:
   - Use "Get Data" > "Web" > "JSON"
   - Enter the URL to your API endpoint (e.g., `http://localhost:5001/api/stock/AAPL`)
   - Alternatively, use the exported JSON files from the `power_bi_data/` directory

3. Create your custom visualizations in Power BI using the imported data

## Customization

- Modify the list of default tickers in `config.py`
- Adjust prediction periods and models in `ml_models.py`
- Customize the dashboard layout in `app.py`
- Add new API endpoints in `api.py`

## Notes

- This is a prototype system and should not be used for actual trading decisions
- Data is sourced from Yahoo Finance via the yfinance library
- The prediction models are simplified for demonstration purposes
- For production use, additional security measures would be required

## Deploying in Production

For a production environment, consider:

1. Using a more robust database for data storage
2. Implementing proper authentication for the API
3. Setting up a more sophisticated scheduling system
4. Using more advanced machine learning models
# Executive Budget Tracker
## Project Demo
You can view the local version of the project at the following link while it's running:

********[Local Project Link](http://127.0.0.1:8502/)
## Project Demo Video

You can watch the demo video of the Stock Market Analysis & Prediction Dashboard here:

[Watch the Video](https://github.com/Sumanth-1522/LIVE-STOCK-MARKET/raw/main/Stock%20Market%20Analysis%20%26%20Prediction%20Dashboard%20-%20Google%20Chrome%202025-04-24%2005-16-02%20(1).mp4)

