import pandas as pd
import numpy as np
import time
import yfinance as yf
from sklearn.preprocessing import LabelEncoder

# Constants
BATCH_SIZE = 50
START_DATE = "2023-06-26"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
FETCH_DELAY = 2  # Delay between batches in seconds

def fetch_stock_data(tickers, start_date, end_date, batch_size):
    """
    Fetch historical stock data and calculate daily returns using yfinance.
    """
    def fetch_batch(batch):
        try:
            # Download data for the batch
            batch_data = yf.download(batch, start=start_date, end=end_date, group_by="ticker", threads=True)
            daily_returns = {}

            # Iterate through tickers in the MultiIndex
            for ticker in batch_data.columns.levels[0]:
                if 'Close' in batch_data[ticker]:
                    prices = batch_data[ticker]['Close']
                    returns = prices.pct_change().dropna()
                    daily_returns[ticker] = returns

            # Convert daily returns dictionary to a DataFrame
            daily_returns_df = pd.DataFrame(daily_returns)
            daily_returns_df = daily_returns_df.reset_index()

            return batch_data, daily_returns_df
        except Exception as e:
            print(f"Error fetching batch: {batch}. Error: {e}")
            return pd.DataFrame(), pd.DataFrame()

    # Batch the tickers
    batches = [" ".join(tickers[i:i + batch_size]) for i in range(0, len(tickers), batch_size)]

    all_data_list = []
    all_returns = []
    for i, batch in enumerate(batches):
        print(f"Fetching batch {i + 1}/{len(batches)}...")
        batch_data, batch_returns = fetch_batch(batch)
        all_data_list.append(batch_data)
        all_returns.append(batch_returns)
        time.sleep(FETCH_DELAY)

    # Combine all returns and raw data into single DataFrames
    final_returns = pd.concat(all_returns, axis=1)
    final_raw_data = pd.concat(all_data_list, axis=1)
    return final_raw_data, final_returns

def load_raw_data(file_path):
    """
    Load raw insider trading data from a CSV file.
    """
    return pd.read_csv(file_path)

def clean_data(data):
    data = data.iloc[:100]
    """
    Perform initial data cleaning and drop unnecessary columns.
    """
    data.columns = [col.strip().replace("\xa0", " ") for col in data.columns]
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 17'], errors='ignore')
    data = data.dropna(subset = ['Price', "Qty", "Owned", "ΔOwn", "Value"])
    data["Price"] = pd.to_numeric(data['Price'].str.replace('$', '').str.replace(',', ''), errors='coerce')
    data["Qty"] = pd.to_numeric(data['Qty'].str.replace(',', ''), errors='coerce')
    data['Owned'] = data['Owned'].apply(lambda x: np.nan if isinstance(x, str) and '%' in x else x)
    data['Owned'] = data['Owned'].str.replace(',', '').astype(int)
    data['ΔOwn'] = data['ΔOwn'].str.replace('%', '').str.replace("New", '0').str.replace('>', '').astype(float) / 100
    data['Value'] = data['Value'].str.strip().str.replace('$', '').str.replace(',', '').str.replace("(", '').str.replace(")", '').astype(float)
    data['Filing Date'] = pd.to_datetime(data['Filing Date'], errors='coerce').dt.tz_localize('UTC')
    data['Trade Date'] = pd.to_datetime(data['Trade Date'], errors='coerce').dt.tz_localize('UTC')
    for col in ['Title', 'Trade Type']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    
    data = data.dropna(subset=['Trade Date', 'Ticker']).reset_index(drop=True)
    return data

def calculate_returns(data, daily_stock_rets):
    """
    Calculate stock returns (`1d`, `1w`, `1m`, `6m`) for each trade.
    """
    def safe_get_return(stock_returns, ticker, trade_date, offset_date):
        try:
            date_range = (stock_returns['Date'] >= trade_date) & (stock_returns['Date'] <= offset_date)
            if stock_returns["Date"].max() < offset_date - pd.Timedelta(days = 3):
                return None
            filtered = stock_returns.loc[date_range, ticker]
            if filtered.empty:
                return None
            cumulative_return = (1 + filtered).prod() - 1
            return cumulative_return
        except Exception as e:
            print(f"Error calculating return for {ticker} from {trade_date} to {offset_date}: {e}")
            return None

    def calculate_row_returns(row):
        ticker = row['Ticker']
        trade_date = row['Trade Date']
        stock_returns = daily_stock_rets[["Date", ticker]]
        if pd.isna(trade_date) or ticker not in daily_stock_rets.columns:
            return pd.Series([None, None, None, None])
        return pd.Series([
            safe_get_return(stock_returns, ticker, trade_date, trade_date + pd.Timedelta(days=1)),
            # safe_get_return(stock_returns, ticker, trade_date, trade_date + pd.Timedelta(weeks=1)),
            # safe_get_return(stock_returns, ticker, trade_date, trade_date + pd.Timedelta(days=30)),
            # safe_get_return(stock_returns, ticker, trade_date, trade_date + pd.Timedelta(days=180))
        ])
    data = data.dropna(subset=['Filing Date', "Trade Date"])
    daily_stock_rets["Date"] = pd.to_datetime(daily_stock_rets["Date"], utc=True)

    # data[['1d', '1w', '1m', '6m']] = data.apply(calculate_row_returns, axis=1)
    data['1d'] = data.apply(calculate_row_returns, axis=1)
    return data

def save_cleaned_data(data, output_path):
    """
    Save the cleaned and processed data to a CSV file.
    """
    data.to_csv(output_path, index=False)

def data_pipeline(insider_data_path, output_path):
    """
    Complete data pipeline to process insider trading data and extract stock returns.
    """
    # Load insider trading data
    data = load_raw_data(insider_data_path)

    # Clean insider trading data
    data = clean_data(data)

    # Extract unique tickers
    tickers = data['Ticker'].unique()

    # Fetch stock data and calculate daily returns
    raw_stock_data, daily_stock_rets = fetch_stock_data(tickers, START_DATE, END_DATE, BATCH_SIZE)

    # Calculate returns
    data = calculate_returns(data, daily_stock_rets)

    # Filter rows with missing returns
    # data = data.dropna(subset=['1d', '1w', '1m', '6m']).reset_index(drop=True)
    data = data.dropna(subset=['1d']).reset_index(drop=True)

    # Save cleaned data
    save_cleaned_data(data, output_path)
    print(f"Processed data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    insider_data_path = "insder_trader_finance.csv"  # Path to insider trading data
    output_path = "cleaned_data_with_returns.csv"  # Path to save cleaned data
    data_pipeline(insider_data_path, output_path)