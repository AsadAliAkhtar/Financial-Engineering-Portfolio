# Overview
This project implements the **Black-Litterman (BL) Model** for portfolio optimization, combining market equilibrium returns with investor-specific views to generate optimal asset allocations.

Unlike traditional mean-variance optimization that relies solely on historical returns, the Black-Litterman model addresses key limitations by:
- Starting with **market equilibrium** as a neutral prior
- Incorporating **subjective views** about specific assets
- Producing **stable, intuitive portfolios** that blend market consensus with investor insights

The project uses real historical price data of large-cap U.S. equities and implements both absolute and relative views to demonstrate the model's flexibility.

# Mathematical Formulation

## 1. Market Equilibrium Return (Prior)

The implied equilibrium excess returns are calculated using the Capital Asset Pricing Model (CAPM):

$$\Pi = \lambda \Sigma w_{mkt}$$

where:
- $\Pi$ = Vector of implied equilibrium excess returns (the prior)
- $\lambda$ = Risk aversion parameter: $\lambda = \frac{E(R_m) - R_f}{\sigma^2_m}$
- $\Sigma$ = Covariance matrix of asset returns
- $w_{mkt}$ = Vector of market capitalization weights

## 2. Investor Views

Views are expressed as linear combinations of expected returns:

$$P \cdot \mu = Q + \epsilon$$

where:
- $P$ = Picking matrix that links views to assets
- $\mu$ = Vector of expected returns
- $Q$ = Vector of view returns
- $\epsilon$ = Error term $\sim N(0, \Omega)$

**Two types of views:**
- **Absolute View**: Expected return of a single asset (e.g., "NVDA will increase by 5%")
- **Relative View**: Expected outperformance between assets (e.g., "AMZN will outperform XOM by 3%")

## 3. Confidence Matrix

The uncertainty in views is represented by:

$$\Omega = \text{diag}(\tau P \Sigma P^T)$$

where:
- $\tau$ = Scaling factor (typically 0.025 - 0.5)
- Higher confidence = Smaller $\Omega$ values

## 4. Black-Litterman Posterior Returns

The model combines equilibrium and views using Bayesian updating:

$$\mu_{BL} = [(\tau\Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau\Sigma)^{-1}\Pi + P^T \Omega^{-1}Q]$$

Simplified form:

$$\mu_{BL} = \Pi + \tau \Sigma P^T [P \tau \Sigma P^T + \Omega]^{-1} (Q - P\Pi)$$

## 5. Portfolio Optimization

**Maximum Quadratic Utility:**

$$\max_w \left[ w^T \mu_{BL} - \frac{\lambda}{2} w^T \Sigma w \right]$$

subject to:

$$\sum_i w_i = 1, \quad 0 \le w_i \le 0.20$$

# Dataset and Parameters

## Assets
The portfolio consists of 8 large-cap U.S. equities:
- **AAPL** - Apple Inc.
- **NVDA** - NVIDIA Corporation
- **JPM** - JPMorgan Chase & Co.
- **AMZN** - Amazon.com Inc.
- **GOOGL** - Alphabet Inc.
- **PG** - Procter & Gamble Co.
- **XOM** - Exxon Mobil Corporation
- **JNJ** - Johnson & Johnson

## Time Period
- **Start Date**: January 1, 2020
- **End Date**: October 30, 2025
- **Duration**: ~6 years of daily price data

## Market Benchmark
- **S&P 500 Index** (^GSPC) - Used for calculating market-implied risk aversion

## Risk-Free Rate
- **3-Month U.S. Treasury Bills** (DGS3MO) - Retrieved via FRED API
- **Average Rate**: 2.75% (annualized)

## Model Parameters
- **Risk Aversion (λ)**: 2.75 (market-implied)
- **Tau (τ)**: 0.1
- **Weight Bounds**: 0% to 20% per asset

# Investor Views Specification

The following hypothetical views were incorporated:

| View Type | Description | Specification |
|-----------|-------------|---------------|
| **Absolute** | NVDA will perform 5% better than equilibrium | $Q_1 = \Pi_{NVDA} + 0.05$ |
| **Absolute** | PG will downperform 2% by equilibrium | $Q_2 = \Pi_{PG} - 0.02$ |
| **Relative** | AMZN will perform better by 3% with respect to XOM | $Q_3 = 0.03$ |
| **Relative** | JPM will perform better by 2% with respect to JNJ | $Q_4 = 0.02$ |

# Results

## Market Equilibrium Returns (Prior)

The implied equilibrium excess returns (annualized) are:

| Asset | Equilibrium Return |
|-------|-------------------|
| NVDA | 44.50% |
| AMZN | 26.15% |
| AAPL | 24.63% |
| GOOGL | 24.55% |
| JPM | 15.15% |
| XOM | 11.15% |
| PG | 8.26% |
| JNJ | 7.76% |

[Market Equilibrium Returns](./Market_Equilibrium_Returns.png)

## Black-Litterman Posterior Returns

After incorporating investor views, the posterior expected returns are:

| Asset | Posterior Return |
|-------|-----------------|
| AMZN | 32.20% |
| NVDA | 30.90% |
| AAPL | 23.93% |
| GOOGL | 23.24% |
| JPM | 15.69% |
| XOM | 14.52% |
| JNJ | 9.00% |
| PG | 7.51% |

Notable changes:
- **AMZN** increased from 26.15% to 32.20% (relative view effect)
- **PG** decreased from 8.26% to 7.51% (negative absolute view)
- **JPM** increased relative to JNJ (relative view effect)

## Optimal Portfolio Allocation

### Maximum Quadratic Utility Portfolio
The final portfolio weights based on risk-adjusted optimization are:

| Asset | Weight | Allocation |
|-------|--------|------------|
| **AAPL** | 20.00% | ████████████████████ |
| **AMZN** | 20.00% | ████████████████████ |
| **GOOGL** | 20.00% | ████████████████████ |
| **XOM** | 15.88% | ███████████████▉ |
| **JNJ** | 10.92% | ██████████▉ |
| **NVDA** | 7.81% | ███████▊ |
| **JPM** | 5.39% | █████▍ |
| **PG** | 0.00% | |

![Portfolio Allocation](./portfolio_allocation.png)

### Key Insights

1. **Technology Dominance**: AAPL, AMZN, and GOOGL each hit the 20% upper bound, comprising 60% of the portfolio due to their strong risk-adjusted returns.

2. **NVDA's Limited Allocation**: Despite a positive 5% view and high equilibrium return (44.5%), NVDA receives only 7.81% allocation due to its **significantly higher volatility** (variance = 0.249, nearly 3× other assets). The optimizer balances return potential against risk.

3. **PG Excluded**: The negative view (-2%) combined with low equilibrium return (8.26%) resulted in zero allocation.

4. **Diversification Benefits**: JNJ (10.92%) provides valuable diversification despite lower returns due to low correlation with tech stocks.

5. **Energy Exposure**: XOM (15.88%) offers diversification and inflation protection with moderate returns.

### Alternative: Maximum Sharpe Ratio Portfolio
For comparison, the tangency portfolio (maximum Sharpe ratio) produces:
- Similar concentration in AMZN (25.00%)
- Different risk-return tradeoff
- May be preferred by more risk-tolerant investors

# Implementation Details

## Tools and Libraries
- **Python 3.x** - Core programming language
- **PyPortfolioOpt** - Black-Litterman model implementation and optimization
- **NumPy/Pandas** - Data manipulation and numerical computations
- **yfinance** - Historical price data retrieval
- **FRED API** - Risk-free rate data (3-Month T-Bills)
- **Matplotlib** - Data visualization
- **python-dotenv** - Environment variable management

## Key Functions Used
```python
# Market-implied risk aversion
black_litterman.market_implied_risk_aversion()

# Prior returns from market caps
black_litterman.market_implied_prior_returns()

# Confidence matrix
BlackLittermanModel.default_omega()

# Black-Litterman posterior
BlackLittermanModel.bl_returns()

# Portfolio optimization
EfficientFrontier.max_quadratic_utility()
```

# Advantages of Black-Litterman

1. **Stability**: Less sensitive to estimation errors than traditional mean-variance optimization
2. **Intuitive**: Starts from market equilibrium (neutral position)
3. **Flexible**: Allows partial views on any subset of assets
4. **Confidence-Weighted**: Views can be expressed with varying degrees of certainty
5. **Diversified**: Produces well-diversified portfolios without extreme positions

# Limitations and Considerations

1. **View Specification**: Requires careful formulation of views with appropriate confidence levels
2. **Parameter Sensitivity**: Results depend on tau (τ) selection, though impact is often limited
3. **Market Cap Timing**: Current implementation uses recent market caps with historical data (potential mismatch)
4. **Historical Covariance**: Assumes past correlations persist into the future

# Future Enhancements

- [ ] Sensitivity analysis for tau (τ) parameter
- [ ] Rolling window implementation for time-varying allocations
- [ ] Comparison with traditional mean-variance optimization
- [ ] Out-of-sample backtesting
- [ ] Transaction cost integration
- [ ] Dynamic view updating based on market signals

# References

1. Black, F., & Litterman, R. (1992). "Global Portfolio Optimization." *Financial Analysts Journal*, 48(5), 28-43.
2. He, G., & Litterman, R. (1999). "The Intuition Behind Black-Litterman Model Portfolios." *Investment Management Research*, Goldman Sachs.
3. Idzorek, T. (2005). "A Step-by-Step Guide to the Black-Litterman Model." *Zephyr Associates*.

# License

This project is available for educational and research purposes.

---

**Author**: [Your Name]  
**Date**: December 11, 2025  
**Contact**: [Your Email/GitHub]
