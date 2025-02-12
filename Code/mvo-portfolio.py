#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean-Variance Optimization using Quadratic Programming
Author: Batuhan Atas
"""

import pandas as pd
import cvxpy as cp
import os

# Load the returns data
save_path = os.path.expanduser("~/Desktop/Research Internship/lp-portfolio-optimization/Data/")
returns = pd.read_csv(os.path.join(save_path, "returns_data.csv"), index_col=0)

# Load the unified target return
with open(os.path.join(save_path, "target_return.txt"), "r") as f:
    target_return = float(f.read().strip())

# Calculate expected returns and the covariance matrix
expected_returns = returns.mean().values
cov_matrix = returns.cov().values
n_assets = len(expected_returns)

# Define decision variables
weights = cp.Variable(n_assets)

# Define the objective function: Minimize portfolio variance
objective = cp.Minimize(cp.quad_form(weights, cov_matrix))

# Constraints
constraints = [
    cp.sum(weights) == 1,  # Portfolio must be fully invested
    weights >= 0,  # No short-selling
    expected_returns @ weights >= target_return  # Use the same target return as MAD model
]

# Formulate and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Retrieve the optimal weights
optimal_weights = weights.value

# Save results as a DataFrame
weights_df = pd.DataFrame({
    'Asset': returns.columns,
    'Optimal Weight': optimal_weights
})

# Save results to CSV
weights_df.to_csv(os.path.join(save_path, "mvo_optimal_weights.csv"), index=False)

print(f"✅ MVO Optimization Completed. Target Return: {target_return:.6f}")
print(weights_df)
