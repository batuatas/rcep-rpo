#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LP Portfolio Optimization using Mean Absolute Deviation (MAD)
Author: Batuhan Atas
"""

import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum
import os

# Load the returns data
save_path = os.path.expanduser("~/Desktop/Research Internship/lp-portfolio-optimization/Data/")
returns = pd.read_csv(os.path.join(save_path, "returns_data.csv"), index_col=0)

# Load the unified target return
with open(os.path.join(save_path, "target_return.txt"), "r") as f:
    target_return = float(f.read().strip())

# Get asset names
assets = returns.columns
n_assets = len(assets)
n_scenarios = len(returns)

# Calculate expected returns
expected_returns = returns.mean()

# Define the MAD LP Model
model = LpProblem("MAD_Portfolio_Optimization", LpMinimize)

# Define portfolio weight variables
x = {asset: LpVariable(f"x_{asset}", lowBound=0) for asset in assets}

# Define deviation variables
d_plus = {t: LpVariable(f"d_plus_{t}", lowBound=0) for t in range(n_scenarios)}
d_minus = {t: LpVariable(f"d_minus_{t}", lowBound=0) for t in range(n_scenarios)}

# Objective Function: Minimize Mean Absolute Deviation (MAD)
model += lpSum((d_plus[t] + d_minus[t]) / n_scenarios for t in range(n_scenarios))

# Constraint: Portfolio weights sum to 1
model += lpSum(x[asset] for asset in assets) == 1

# Constraints for MAD deviation calculations
for t in range(n_scenarios):
    model += d_plus[t] - d_minus[t] == lpSum(returns.iloc[t, j] * x[assets[j]] for j in range(n_assets)) - lpSum(expected_returns[j] * x[assets[j]] for j in range(n_assets))

# Apply the same target return for consistency
model += lpSum(expected_returns[stock] * x[stock] for stock in assets) >= target_return

# Diversification Constraints
max_weight = 0.20  # No single asset > 20%
for asset in assets:
    model += x[asset] <= max_weight

# Solve the model
model.solve()

# Retrieve Optimal Portfolio Weights
optimal_weights = {asset: x[asset].varValue for asset in assets}
weights_df = pd.DataFrame(list(optimal_weights.items()), columns=["Asset", "Optimal Weight"])

# Save results to CSV
weights_df.to_csv(os.path.join(save_path, "lp_optimal_weights.csv"), index=False)

print(f"✅ MAD Optimization Completed. Target Return: {target_return:.6f}")
print(weights_df)
