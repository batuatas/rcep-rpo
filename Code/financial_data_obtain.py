#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 21:33:02 2025

@author: batuhanatas
"""

import yfinance as yf
import pandas as pd
import os

# Define the correct save path
save_path = os.path.expanduser("~/Desktop/Research Internship/lp-portfolio-optimization/Data")


# Define stock tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Download historical stock data
data = yf.download(tickers, start="2024-01-01", end="2025-01-01", interval="1mo")

# Ensure correct price column
if 'Adj Close' in data.columns:
    price_data = data['Adj Close']
elif 'Close' in data.columns:
    price_data = data['Close']
else:
    raise KeyError("Neither 'Adj Close' nor 'Close' found in downloaded data")

# Calculate daily returns
returns = price_data.pct_change().dropna()

# Calculate expected returns and volatility
expected_returns = returns.mean()
volatility = returns.std()

# Save Data to the specified directory
price_data.to_csv(os.path.join(save_path, "financial_data.csv"))
returns.to_csv(os.path.join(save_path, "returns_data.csv"))

# Save summary data
summary_df = pd.DataFrame({
    'Expected Return': expected_returns,
    'Volatility': volatility
})
summary_df.to_csv(os.path.join(save_path, "summary_data.csv"))

# Display confirmation
print(f"Data successfully saved in: {save_path}")
