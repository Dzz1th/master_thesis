from fredapi import Fred
import pandas as pd
from datetime import datetime, timedelta
import os
import yfinance as yf

fred_api_key = "24ef48aa8d486c9af88bed2f2d378a8c"
fred = Fred(api_key=fred_api_key) 

# Dictionary of yield series IDs
YIELD_SERIES = {
    'DGS1MO': '1-Month Treasury Constant Maturity Rate',
    'DGS3MO': '3-Month Treasury Constant Maturity Rate',
    'DGS6MO': '6-Month Treasury Constant Maturity Rate',
    'DGS1': '1-Year Treasury Constant Maturity Rate',
    'DGS2': '2-Year Treasury Constant Maturity Rate',
    'DGS5': '5-Year Treasury Constant Maturity Rate',
    'DGS10': '10-Year Treasury Constant Maturity Rate',
    'DGS30': '30-Year Treasury Constant Maturity Rate'
}

TIPS_SERIES = {
    'DFII5': '5-Year TIPS',
    'DFII10': '10-Year TIPS',
    'DFII20': '20-Year TIPS',
    'DFII30': '30-Year TIPS'
}

MARKET_SERIES = {
    'VIXCLS': 'VIX Volatility Index',
    'DTWEXBGS': 'Trade Weighted US Dollar Index: Broad',
    'DTWEXM': 'Trade Weighted US Dollar Index: Major Currencies',
    'BAMLH0A0HYM2': 'ICE BofA US High Yield Index Option-Adjusted Spread'
}

def get_snp500_yahoo(start_date='2011-01-01', end_date=None):
    """
    Fetch S&P 500 data from Yahoo Finance
    
    Parameters:
    start_date (str): Start date in format 'YYYY-MM-DD'
    end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
    pandas DataFrame with S&P 500 data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading S&P 500 data from Yahoo Finance ({start_date} to {end_date})")
    
    # Download S&P 500 data using ^GSPC ticker
    sp500_ticker = yf.Ticker("^GSPC")
    sp500_data = sp500_ticker.history(start=start_date, end=end_date)
    
    # Remove timezone awareness to match FRED data
    sp500_data.index = sp500_data.index.tz_localize(None)
    
    # Create DataFrame with relevant columns
    df = pd.DataFrame({
        'SP500': sp500_data['Close']
    })
    

    return df

def get_yield_data(start_date=None, end_date=None):
    """
    Fetch yield curve data from FRED
    
    Parameters:
    start_date (str): Start date in format 'YYYY-MM-DD'
    end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
    pandas DataFrame with yield data
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    yield_data = {}
    
    for series_id, description in YIELD_SERIES.items():
        try:
            # Get data for each series
            series = fred.get_series(series_id, 
                                   observation_start=start_date,
                                   observation_end=end_date)
            yield_data[description] = series
            print(f"Downloaded {description}")
        except Exception as e:
            print(f"Error downloading {description}: {str(e)}")
            
    # Combine all series into a single DataFrame
    df = pd.DataFrame(yield_data)
    
    # Forward fill missing values (weekends and holidays)
    df = df.fillna(method='ffill')
    
    return df


def get_tips_data(start_date=None, end_date=None):
    """
    Fetch TIPS data from FRED
    """
    tips_data = {}
    for series_id, description in TIPS_SERIES.items():
        try:
            series = fred.get_series(series_id, 
                                   observation_start=start_date,
                                   observation_end=end_date)
            tips_data[description] = series
            print(f"Downloaded {description}")
        except Exception as e:
            print(f"Error downloading {description}: {str(e)}")
            
    # Combine all series into a single DataFrame
    df = pd.DataFrame(tips_data)
    
    # Forward fill missing values (weekends and holidays)
    df = df.fillna(method='ffill')

    return df

def get_market_data(start_date=None, end_date=None):
    """
    Fetch market data including S&P 500 (from Yahoo), VIX, USD Index, and HY Spread (from FRED)
    """
    if start_date is None:
        start_date = '2011-01-01'
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get S&P 500 from Yahoo Finance
    sp500_df = get_snp500_yahoo(start_date, end_date)
    
    # Get other market data from FRED
    market_data = {}
    for series_id, description in MARKET_SERIES.items():
        try:
            series = fred.get_series(series_id, 
                                   observation_start=start_date,
                                   observation_end=end_date)
            market_data[description] = series
            print(f"Downloaded {description}")
        except Exception as e:
            print(f"Error downloading {description}: {str(e)}")
    
    # Combine FRED data into DataFrame
    if market_data:
        fred_df = pd.DataFrame(market_data)
        # Combine S&P 500 and FRED data
        combined_df = pd.concat([sp500_df, fred_df], axis=1, join='outer')
    else:
        combined_df = sp500_df
    
    # Forward fill missing values
    combined_df = combined_df.fillna(method='ffill')
    
    return combined_df

def save_yield_data(df, filename='us_yields.csv'):
    """
    Save yield data to CSV file
    """
    path = os.path.join("/Users/dzz1th/Job/mgi/Soroka/data/pc_data", filename)
    df.to_csv(path)
    print(f"Data saved to {path}")


def get_economic_data(start_date=None, end_date=None):
    """
    Fetch economic data from FRED
    
    Parameters:
    start_date (str): Start date in format 'YYYY-MM-DD'
    end_date (str): End date in format 'YYYY-MM-DD'
"""
    series_ids = {
        "Capacity Utilization: Manufacturing": "TCU",
        "Real Personal Income": "W875RX1",
        "GDP": "GDPC1",
        "Unemployment": "UNRATE",
        "Labor Participation": "CIVPART",
        "Average Hourly Earnings": "AHETPI",
        "CPI": "CPIAUCSL",
        "PCE": "PCEPILFE",
        "Durable Goods Orders": "DGORDER",
        "Oil Price": "DCOILWTICO",
        "S&P 500": "SP500",
        # Additional Indicators:
        "Housing Starts": "HOUST",
        "Building Permits": "PERMIT",
        "Manufacturing PMI": "ISM/MAN_PMI",  # Verify availability; alternatives may be required
        "Consumer Confidence": "UMCSENT",
        "Job Openings": "JTSJOL",
        "Wage Growth (ECI: Wages & Salaries)": "ECIWAG",
        "Nonfarm Payrolls": "PAYEMS"
    }
    # Fetch data from FRED
    data = {}
    for name, series_id in series_ids.items():
        try:
            series_data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            data[name] = series_data
            print(f"Successfully fetched data for {name} ({series_id})")
        except Exception as e:
            print(f"Error fetching {name} ({series_id}): {e}")

    # Convert fetched data into a single DataFrame (aligned on date index)
    df = pd.DataFrame(data)
    return df


def save_economic_data(df, filename='us_economic_data.csv'):
    """
    Save economic data to CSV file
    """
    path = os.path.join("/Users/dzz1th/Job/mgi/Soroka/data/pc_data", filename)
    df.to_csv(path)
    print(f"Data saved to {path}")

def save_tips_data(df, filename='us_tips_data.csv'):
    """
    Save TIPS data to CSV file
    """
    path = os.path.join("/Users/dzz1th/Job/mgi/Soroka/data/pc_data", filename)
    df.to_csv(path)
    print(f"Data saved to {path}")

def save_market_data(df, filename='us_market_data.csv'):
    """
    Save market data to CSV file
    """
    path = os.path.join("/Users/dzz1th/Job/mgi/Soroka/data/pc_data", filename)
    df.to_csv(path)
    print(f"Data saved to {path}")

if __name__ == "__main__":
    # Set your date range
    start_date = '2011-01-01'
    end_date = '2025-02-19'  # Today's date
    
    # Get the data
    print("Fetching yields data...")
    yields_df = get_yield_data(start_date, end_date)
    
    print("Fetching TIPS data...")
    tips_df = get_tips_data(start_date, end_date)
    
    print("Fetching market data (S&P 500, VIX, USD Index, HY Spread)...")
    market_df = get_market_data(start_date, end_date)
    
    print("Fetching economic data...")
    economic_df = get_economic_data(start_date, end_date)
    
    # Basic data inspection
    print("\nYields Data Shape:", yields_df.shape)
    print("TIPS Data Shape:", tips_df.shape)
    print("Market Data Shape:", market_df.shape)
    print("Economic Data Shape:", economic_df.shape)
    
    print("\nMarket data columns:")
    print(market_df.columns.tolist())
    
    print("\nFirst few rows of market data:")
    print(market_df.head())
    
    # Save to CSV
    save_yield_data(yields_df)
    save_tips_data(tips_df)
    save_market_data(market_df)
    save_economic_data(economic_df)