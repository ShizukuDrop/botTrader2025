import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# Read the CSV file into a DataFrame
df = pd.read_csv('DOGEUSDT_historical_data.csv')

# Remove unused columns
df = df.drop(['ignore', 'close_time'], axis=1)

# Convert timestamp and create cyclical features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute

df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
df['minute_sin'] = np.sin(2 * np.pi * df['minute']/60)
df['minute_cos'] = np.cos(2 * np.pi * df['minute']/60)

# Initialize scaler and define columns
scaler = MinMaxScaler()
columns_to_scale = ['volume', 'taker_buy_base_asset_volume', 'number_of_trades',
                    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']

# Scale features
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Save scaler for later use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Drop temporal and redundant columns
df = df.drop(['hour', 'minute', 'timestamp', 'quote_asset_volume', 'taker_buy_quote_asset_volume'], axis=1)

# Calculate split index
split_idx = int(len(df) * 0.8)

# Split data
train_df = df[:split_idx]
test_df = df[split_idx:]

# Save processed DataFrames
train_df.to_csv('DOGEUSDT_train.csv', index=False)
test_df.to_csv('DOGEUSDT_test.csv', index=False)

print(f"\nTotal samples: {len(df)}")
print(f"Training samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Testing samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")