# LP Portfolio Optimization

## Overview
This repository provides an implementation of **Portfolio Optimization** using two different approaches:
1. **Mean Absolute Deviation (MAD) Model** (implemented in `lp_portfolio.py`) as a LP Model
2. **Mean-Variance Optimization (MVO) Model** (implemented in `mvo_portfolio.py`) as a quadratic model offered by Markovitz

Additionally, the repository includes a script (`financial_data_obtain.py`) for fetching and preparing financial data from Yahoo Finance.

## 📌 Scripts and Their Functions

### **1️⃣ financial_data_obtain.py** (Financial Data Collection)
This script downloads historical price data for selected assets and prepares the dataset for portfolio optimization.
- **Sources:** Uses `yfinance` to fetch stock, bond, ETF, and commodity prices.
- **Output:** Saves cleaned data (prices, returns) in the `Data/` folder.
- **Key Functions:**
  - Fetch stock prices (`Adj Close` or `Close` values)
  - Compute daily/monthly returns
  - Handle missing data (forward-fill, drop NaN)
  - Compute **target return** (saved in `target_return.txt`)
 
### **2️⃣ lp_portfolio.py** (Mean Absolute Deviation Model)
This script formulates and solves the **LP-based MAD portfolio optimization**:
- **Objective:** Minimize portfolio risk measured by mean absolute deviation.
- **Constraints:**
  - Portfolio weights sum to **1**
  - **Diversification constraints** (no asset > 20%)
  - **Minimum target return**
- **Solver:** Uses `PuLP` (Linear Programming Solver).
- **Output:** Optimal portfolio weights are:
  - **Printed in the terminal**
  - **Saved in `Data/lp_optimal_weights.csv`**

### **3️⃣ mvo_portfolio.py** (Mean-Variance Optimization Model)
This script implements the classical **Mean-Variance Optimization (MVO)** approach:
- **Objective:** Minimize portfolio variance while meeting a return target.
- **Constraints:**
  - Portfolio weights sum to **1**
  - No short-selling (weights ≥ 0)
- **Solver:** Uses `cvxpy` for quadratic programming.
- **Output:** Optimal portfolio weights are:
  - **Printed in the terminal**
  - **Saved in `Data/mvo_optimal_weights.csv`**

