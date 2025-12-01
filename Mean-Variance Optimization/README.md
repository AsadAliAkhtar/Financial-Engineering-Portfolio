# Overview
This project focuses on **portfolio optimization** under **Markowitz** framework, using real historical price data of large-cap U.S. equities.
The following optimization task are performed:
1. Minimum-Variance Portfolio (MVP) using *CVXPY*
   - Unconstraint Upper Bound MVP
   - Constarint Upper Bound MVP (upper weight bound $\le$ 0.15)
   - Comparison of Portfolio Weights
2. Maximum-Return Portfolio at Target Volatility using *PyPortfolioOpt*
   - Construction of the efficient frontier
   - Selecting a portfolio that **maximizes return** for a chosen volatility
   - Plotting the frontier with optimal point
# Mathematical Formulation
1. Minimum-Variance Portfolio<br>

The objective function for this prblem is:<br>
<div align="center">
  
$$\min_{w} w^T \Sigma w$$ 

</div>
Subject to:<br>

$$
\Sigma_{i} w_{i} = 1, w_{i} \ge 0
$$

2. Maximum-Return Portfolio at Target Risk<br>

The objective function for this problem is:<br>
$$
max_{w} \mu^T w
$$

For the bounded constraint version of MVP, the constraint will be: <br>
$$
0 \le w \le 0.15
$$
