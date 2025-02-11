#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 21:33:02 2025

@author: batuhanatas
"""
import yfinance as yf
import pandas as pd
import os

# Define save path
save_path = os.path.expanduser("~/Desktop/Research Internship/lp-portfolio-optimization/Data")
os.makedirs(save_path, exist_ok=True)

# Define a diverse set of assets: Stocks, Bonds, ETFs, Commodities
assets = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'V', 'JNJ',
    'WMT', 'PG', 'DIS', 'MA', 'HD', 'BAC', 'XOM', 'PFE', 'KO', 'CSCO',
    'BND', 'AGG', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'BNDX', 'EMB', 'MUB',
    'SPY', 'QQQ', 'DIA', 'IWM', 'EFA', 'EEM', 'VNQ', 'GLD', 'SLV', 'USO',
    'GLD', 'SLV', 'USO', 'DBA', 'DBC', 'UNG', 'PALL', 'CORN', 'WEAT'
]

# Download historical data from 2024-01-01 to 2024-12-31
data = yf.download(assets, start="2024-01-01", end="2024-12-31", interval="1d")

# Ensure correct price column
if 'Adj Close' in data.columns:
    price_data = data['Adj Close']
elif 'Close' in data.columns:
    price_data = data['Close']
else:
    raise KeyError("Neither 'Adj Close' nor 'Close' found in downloaded data")

# Check for missing data
print("assets with missing data:")
print(price_data.isnull().sum()[price_data.isnull().sum() > 0])

# Fill missing values using forward-fill
price_data.ffill(inplace=True)

# Drop assets that have missing values in all rows
price_data = price_data.dropna(axis=1, how="all")

# Calculate monthly returns
returns = price_data.pct_change().dropna()

# Calculate expected returns and volatility
expected_returns = returns.mean()
volatility = returns.std()

# Save Data
price_data.to_csv(os.path.join(save_path, "financial_data.csv"))
returns.to_csv(os.path.join(save_path, "returns_data.csv"))

# Save summary data
summary_df = pd.DataFrame({
    'Expected Return': expected_returns,
    'Volatility': volatility
})

# Drop NaN values in summary
summary_df.dropna(inplace=True)
summary_df.to_csv(os.path.join(save_path, "summary_data.csv"))

print(f"Data successfully saved in: {save_path}")
print(summary_df.head())  # Show preview of summary data
