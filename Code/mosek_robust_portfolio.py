import mosek.fusion as mf
import numpy as np
import pandas as pd
from scipy.linalg import svd
import os


df = pd.read_csv("~/Desktop/Research Internship/Mosek_model/financial_data.csv", index_col=0, parse_dates=True)

# compute simple returns  - we can also use logreturn
x = df.pct_change().dropna()

# expected returns
x_bar = x.mean().values.reshape(-1, 1)

# covariance matrix
Sigma = x.cov().values

# uncertainty matrix is set as the diagonal of variances
Omega = np.diag(x.var().values)

# SVD decomposition on Omega
U, S, _ = svd(Omega)  # singular value decomp
S_sqrt = np.sqrt(np.diag(S))  # sqrt of singular values
GQ = U @ S_sqrt  # transformed matrix for robustness constraint

n = len(x_bar) # num of assets

# === Parameter ranges ===
kappas = np.linspace(0.1, 1, 4)  
lambs = np.linspace(1, 10, 10)   

# to keep results for different lambda_
columns = ["lambda", "kappa", "obj", "return", "risk"] + df.columns.tolist()
df_result = pd.DataFrame(columns=columns)

"""
We can model both terms using the second-order cones. 
For the term with square-root, the quadratic cone, 
while the portfolio variance term can be modeled using the rotated quadratic cone.
"""

with mf.Model("Robust") as M:
    
    # variables
    theta = M.variable("theta", n, mf.Domain.greaterThan(0.0))  # portfolio weights
    s = M.variable("s", 1, mf.Domain.greaterThan(0.0))  # portfolio risk
    t = M.variable("t", 1, mf.Domain.greaterThan(0.0))  # robustness term

    # sum(theta) = 1
    M.constraint('budget', mf.Expr.sum(theta), mf.Domain.equalsTo(1.0))

    # Parameters
    lambda_param = M.parameter()
    kappa_param = M.parameter()
    
    # objective func
    worst_case_return = mf.Expr.sub(mf.Expr.dot(theta, x_bar.flatten()), mf.Expr.mul(kappa_param, t))
    M.objective("obj", mf.ObjectiveSense.Maximize, mf.Expr.sub(worst_case_return, mf.Expr.mul(lambda_param, s)))

    # robustness constraint
    M.constraint("robustness", mf.Expr.vstack(t, mf.Expr.reshape(mf.Expr.mul(GQ.T, theta), n)), mf.Domain.inQCone())
     
    # risk constraint 
    G = np.linalg.cholesky(Sigma)  
    M.constraint("risk", mf.Expr.vstack(s, mf.Expr.constTerm(1.0), mf.Expr.reshape(mf.Expr.mul(G.T, theta), n)), mf.Domain.inRotatedQCone())

    # Solve for different risk parameters
    for kappa_val in kappas:
        kappa_param.setValue(kappa_val)
        for d in lambs:
            lambda_param.setValue(d)  # set risk tolerance parameter
            M.solve()
        
            #results
            portfolio_return = np.dot(x_bar.T, theta.level())[0] - kappa_val * t.level()[0]
            portfolio_risk = np.sqrt(2 * s.level()[0])
            row = pd.Series([d, kappa_val, M.primalObjValue(), portfolio_return, portfolio_risk] + list(theta.level()), index=columns)
        
           
            df_result = pd.concat([df_result, pd.DataFrame([row])], ignore_index=True)


print(df_result)


# === Convert to long-form for asset weights ===
df_long = df_result.melt(
    id_vars=["lambda", "kappa"],
    value_vars=df.columns.tolist(),
    var_name="Asset",
    value_name="Weight"
)

df_long['Asset'] = pd.Categorical(df_long['Asset'], categories=df.columns.tolist(), ordered=True)
df_long = df_long.sort_values(by=['lambda', 'kappa', 'Asset'])

# === Export results to CSV ===
df_result_csv_path = os.path.join(os.getcwd(), "portfolio_results_lambda_kappa.csv")
df_result.to_csv(df_result_csv_path, index=False)
print(f"✅ Wide format df_result exported to: {df_result_csv_path}")

df_long_csv_path = os.path.join(os.getcwd(), "portfolio_weights_long_lambda_kappa.csv")
df_long.to_csv(df_long_csv_path, index=False)
print(f"✅ Long format df_long exported to: {df_long_csv_path}")
