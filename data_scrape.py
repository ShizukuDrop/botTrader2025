import os
from binance.client import Client
from dotenv import load_dotenv
import pandas as pd

# Load API keys from .env
load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

# Initialize Binance client
client = Client(api_key, api_secret)

# Parameters
symbol = 'DOGEUSDT'  # Replace with your chosen trading pair
interval = Client.KLINE_INTERVAL_1MINUTE
start_date = '1 Jan 2023'  # Adjust the start date as needed

# Fetch historical klines (candlestick data)
print(f"Fetching data for {symbol}...")
klines = client.get_historical_klines(symbol, interval, start_date)

# Convert to DataFrame
columns = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
]
df = pd.DataFrame(klines, columns=columns)

# Convert timestamp to readable datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

# Save to CSV
output_file = f"{symbol}_historical_data.csv"
df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")

# Display sample data
print(df.head())
