#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:18:55 2025

@author: batuhanatas
"""

import numpy as np
import pandas as pd
import cvxpy as cp

# Load the returns data
save_path = "~/Desktop/Research Internship/lp-portfolio-optimization/Data/returns_data.csv"
returns = pd.read_csv(save_path, index_col=0)

# Calculate expected returns and the covariance matrix
expected_returns = returns.mean().values
cov_matrix = returns.cov().values

# Number of assets
n_assets = len(expected_returns)

# Define the optimization variables
weights = cp.Variable(n_assets)

# Define the target return (e.g., 0.01 for 1%)
target_return = 0.1

# Define the objective function (minimize portfolio variance)
objective = cp.Minimize(cp.quad_form(weights, cov_matrix))

# Define the constraints
constraints = [
    cp.sum(weights) == 1,  # Sum of weights must be 1
    weights >= 0,          # No short-selling
    expected_returns @ weights >= target_return  # Target return constraint
]

# Formulate the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Retrieve the optimal weights
optimal_weights = weights.value

# Create a DataFrame for better visualization
weights_df = pd.DataFrame({
    'Asset': returns.columns,
    'Optimal Weight': optimal_weights
})

# Display the optimal weights
print("Optimal Portfolio Allocation (Mean-Variance Optimization):")
print(weights_df)
