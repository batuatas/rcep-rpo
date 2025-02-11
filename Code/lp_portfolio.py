#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 20:52:55 2025

@author: batuhanatas
"""

import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

# Load the returns data
save_path = "~/Desktop/Research Internship/lp-portfolio-optimization/Data/returns_data.csv"
returns = pd.read_csv(save_path, index_col=0)

# Get asset names
assets = returns.columns
n_assets = len(assets)
n_scenarios = len(returns)

# Calculate expected returns and volatility
expected_returns = returns.mean()
volatility = returns.std()

# Set confidence level for CVaR (95% confidence)
alpha = 0.95

# Define the LP Model
model = LpProblem("CVaR_Portfolio_Optimization", LpMinimize)

# Define portfolio weight variables
x = {asset: LpVariable(f"x_{asset}", lowBound=0) for asset in assets}

# Define VaR (V) and deviation variables (z_t)
V = LpVariable("VaR")
z = {t: LpVariable(f"z_{t}", lowBound=0) for t in range(n_scenarios)}

# Objective Function: Minimize CVaR (VaR + tail risk penalty)
model += V + (1 / ((1 - alpha) * n_scenarios)) * lpSum(z[t] for t in range(n_scenarios))

# Constraint: Portfolio weights sum to 1
model += lpSum(x[asset] for asset in assets) == 1

# CVaR Risk Constraints
for t in range(n_scenarios):
    model += z[t] >= -lpSum(returns.iloc[t, j] * x[assets[j]] for j in range(n_assets)) - V

# Diversification Constraints
max_weight = 0.05 # No single asset > 20%

for asset in assets:
    model += x[asset] <= max_weight

# Risk-Adjusted Target Return Constraint
adjusted_target_return = expected_returns.mean() / (1 + volatility.mean())
model += lpSum(expected_returns[stock] * x[stock] for stock in assets) >= adjusted_target_return

# Solve the model
model.solve()

# Retrieve Optimal Portfolio Weights
optimal_weights = {asset: x[asset].varValue for asset in assets}
weights_df = pd.DataFrame(list(optimal_weights.items()), columns=["Asset", "Optimal Weight"])

# Print results
print("Optimal Portfolio Allocation (CVaR Optimization with Diversification):")
print(weights_df)
