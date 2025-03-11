# 📈 Explainable Portfolio Optimization

This repository provides multiple implementations of **Portfolio Optimization** models, with a focus on **interpretability**, **risk management**, and **robustness**. It includes both **classic linear models** and **robust optimization** techniques, along with **parameter analysis** for deeper insights into model behavior.

---

## 🗂️ Table of Contents
1. [Overview](#overview)
2. [Linear Portfolio Trials](#linear-portfolio-trials)
   - [Mean Absolute Deviation (MAD)](#1️⃣-mad-model---lp_portfoliopy)
   - [Mean-Variance Optimization (MVO)](#2️⃣-mvo-model---mvo_portfoliopy)
3. [Robust Portfolio Optimization](#robust-portfolio-optimization)
   - [Lambda-Kappa Parameter Analysis](#lambda-kappa-parameter-analysis)
4. [Data Files](#data-files)
5. [How to Run](#how-to-run)
6. [Dependencies](#dependencies)
7. [Folder Structure](#folder-structure)
8. [Conclusion](#conclusion)

---

## 📖 Overview

This projects' aim is **Explainable Portfolio Optimization**, several approaches are tried:
1. **Mean Absolute Deviation (MAD)**: Linear programming approach.
2. **Mean-Variance Optimization (MVO)**: Quadratic programming following Markowitz.
3. **Robust Portfolio Optimization (SOCP)**: Accounts for model uncertainty and risk aversion through **lambda (λ)** and **kappa (κ)** parameter analysis.

Additionally, it includes tools to:
- **Fetch and preprocess financial data** (using Yahoo Finance)
- **Perform parameter sensitivity analysis**  
- **Export results for further exploration and visualization**

---

## 💼 Linear Portfolio Trials

### 1️⃣ MAD Model - `lp_portfolio.py`
Implements **Mean Absolute Deviation (MAD)** portfolio optimization using **Linear Programming (LP)**.

- **Objective**:  
  Minimize the **mean absolute deviation** (MAD) of portfolio returns, offering an alternative risk measure to variance.
  
- **Constraints**:
  - Fully invested (weights sum to 1)
  - Diversification cap (no single asset > 20%)
  - Achieve a **minimum target return** (from `target_return.txt`)

- **Solver**:  
  `PuLP`

- **Output**:
  - Optimal weights printed to terminal
  - Results saved to:  
    `Data/lp_optimal_weights.csv`

---

### 2️⃣ MVO Model - `mvo_portfolio.py`
Implements **Markowitz Mean-Variance Optimization (MVO)** using **Quadratic Programming (QP)**.

- **Objective**:  
  Minimize **portfolio variance**, aiming for an **efficient frontier** allocation.

- **Constraints**:
  - Fully invested (weights sum to 1)
  - No short-selling (weights ≥ 0)
  - Minimum target return (same as MAD)

- **Solver**:  
  `cvxpy`

- **Output**:
  - Optimal weights printed to terminal  
  - Results saved to:  
    `Data/mvo_optimal_weights.csv`

---

## 🛡️ Robust Portfolio Optimization

### Robust Portfolio Model - `robust_portfolio.py`
Implements **Robust Portfolio Optimization** via **Second-Order Cone Programming (SOCP)** with **Mosek Fusion API**.

- **Objective**:  
  Considering the **worst-case scenario** in portfolio optimization with parameters:
  - **Lambda (λ)** → risk aversion  
  - **Kappa (κ)** → robustness to return uncertainty  

- **Model Structure**:  
