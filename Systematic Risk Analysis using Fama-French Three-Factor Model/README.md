# Systematic Risk Analysis Using Fama-French Three-Factor Model

## Overview

This project applies the **Fama-French 3-Factor Model** to decompose stock returns into systematic risk components across 7 stocks from different sectors over an 11-year period (2015-2025).

Financial returns are driven by multiple sources of risk. Traditional single-factor models (like CAPM) only consider market risk, but the Fama-French model recognizes that stock returns are influenced by three distinct risk factors:
- **Market risk** (systematic exposure to overall market movements)
- **Size risk** (small-cap vs large-cap behavior)  
- **Value risk** (value vs growth characteristics)

This project goes beyond static analysis by implementing **rolling window regression** to capture how these risk exposures evolve over time. Markets don't stand still — companies transform, economic regimes shift, and risk profiles change. The rolling window approach reveals when and how dramatically these shifts occur.

## Dataset and Stocks Analyzed

### Assets
Seven stocks across three sectors:
- **Technology**: Apple (AAPL), NVIDIA (NVDA), Microsoft (MSFT)
- **Financials**: JPMorgan Chase (JPM), Goldman Sachs (GS)
- **Consumer/Energy**: Walmart (WMT), Exxon Mobil (XOM)

### Time Period
- **Start Date**: January 1, 2015
- **End Date**: December 30, 2025
- **Duration**: ~11 years of monthly data
- **Total Observations**: 131 months

### Data Frequency
- **Monthly returns** (last trading day of each month)
- Matches Fama-French factor data frequency
- Reduces noise compared to daily data while capturing meaningful variation

### Factor Data Source
- **Kenneth French Data Library** (official source for Fama-French factors)
- Mkt-RF, SMB, HML, and RF (risk-free rate)

## Mathematical Framework

### 1. Excess Returns

The model explains excess returns (returns above the risk-free rate):

$$r_{i,t}^{excess} = r_{i,t} - r_{f,t}$$

where:
- $r_{i,t}$ = Stock return at time $t$
- $r_{f,t}$ = Risk-free rate (1-month Treasury bill)

### 2. Fama-French 3-Factor Model

$$r_{i,t} - r_{f,t} = \alpha_i + \beta_i(r_{m,t} - r_{f,t}) + s_i \cdot SMB_t + h_i \cdot HML_t + \epsilon_{i,t}$$

**Model Components:**

| Component | Description | Interpretation |
|-----------|-------------|----------------|
| $\alpha_i$ | Jensen's alpha | Risk-adjusted outperformance (intercept) |
| $\beta_i$ | Market beta | Sensitivity to market movements |
| $r_{m,t} - r_{f,t}$ | Market premium | Excess return of market over risk-free rate |
| $s_i$ | Size factor loading | Exposure to small-cap vs large-cap premium |
| $SMB_t$ | Small Minus Big | Return difference: small-cap − large-cap |
| $h_i$ | Value factor loading | Exposure to value vs growth premium |
| $HML_t$ | High Minus Low | Return difference: value − growth |
| $\epsilon_{i,t}$ | Residual | Idiosyncratic (stock-specific) return |

### 3. Statistical Interpretation

**Alpha ($\alpha$):**
- $\alpha > 0$ (significant): Outperformance beyond factor exposures
- $\alpha < 0$ (significant): Underperformance relative to risk factors
- $\alpha \approx 0$: Returns fully explained by factors

**Beta ($\beta$):**
- $\beta > 1$: Amplifies market movements (high systematic risk)
- $\beta = 1$: Moves with market (average risk)
- $\beta < 1$: Dampens market movements (defensive)

**SMB ($s$):**
- $s > 0$: Behaves like small-cap stock
- $s < 0$: Behaves like large-cap stock

**HML ($h$):**
- $h > 0$: Value stock characteristics
- $h < 0$: Growth stock characteristics

**R-squared ($R^2$):**

$$R^2 = 1 - \frac{SS_{residual}}{SS_{total}}$$

- Percentage of return variance explained by the three factors
- High $R^2$: Systematic (factor-driven) returns
- Low $R^2$: Idiosyncratic (company-specific) returns

### 4. Rolling Window Specification

For time-varying analysis, the model is estimated on 36-month windows:

$$\hat{\beta}_{i,t} = f(\text{returns}_{t-35:t})$$

- Window size: 36 months (3 years)
- Estimation: 96 rolling windows (moving forward month by month)
- Reveals when coefficients change over time

## Methodology

### 1. Data Collection and Preparation
- Downloaded historical stock prices using `yfinance`
- Fetched Fama-French factors from Kenneth French Data Library
- Calculated monthly returns: $(P_t - P_{t-1}) / P_{t-1}$
- Computed excess returns: $r_{excess} = r_{stock} - r_f$
- Verified data quality (no missing values)

### 2. Static Fama-French Regression

**Approach:**
- Single OLS regression per stock using all 131 months
- Estimated using `statsmodels` library
- Student's t-distribution assumed for error terms (captures fat tails)

**Extracted Statistics:**
- Coefficients: $\alpha, \beta, s, h$
- Standard errors and t-statistics
- P-values (significance tests)
- R-squared (model fit)

**Purpose:** Establish baseline factor exposures over the full period

### 3. Rolling Window Analysis

**Approach:**
- 36-month expanding window
- Re-estimate model 96 times (once per month from 2018-01 to 2025-12)
- Store time-varying coefficients

**Why 36 Months?**
- Academic standard (Ferson & Harvey, 1999)
- Sufficient observations for stable estimates (n ≥ 30)
- Short enough to capture regime changes

**Rationale:** Static regression assumes constant factor exposures — unrealistic over a decade of market evolution, COVID disruption, and sector rotations

### 4. Visualization Strategy

**Static Analysis:**
- Market beta comparison (horizontal bar chart)
- Color-coded by risk level (red/orange/green)

**Rolling Analysis:**
- Time-series plots of beta evolution
- Time-series plots of HML evolution
- COVID period highlighted (2020)

## Results

### Static Factor Exposures (Full Period: 2015-2025)

| Stock | Alpha | Market Beta | Value Beta (HML) | Size Beta (SMB) | R² |
|-------|-------|-------------|------------------|-----------------|-----|
| **NVDA** | 0.0348*** | 1.76*** | -0.97*** | -0.18 | 0.42 |
| **GS** | 0.0035 | 1.34*** | 0.69*** | 0.27* | 0.68 |
| **AAPL** | 0.0069 | 1.18*** | -0.45** | -0.14 | 0.48 |
| **JPM** | 0.0061 | 1.12*** | 0.80*** | 0.06 | 0.73 |
| **MSFT** | 0.0085* | 1.04*** | -0.49*** | -0.62*** | 0.61 |
| **XOM** | -0.0005 | 0.86*** | 1.04*** | -0.03 | 0.51 |
| **WMT** | 0.0056 | 0.54*** | -0.16 | -0.46** | 0.21 |

*Note: *** p<0.01, ** p<0.05, * p<0.10*

### Key Statistical Findings

**1. Alpha (Risk-Adjusted Performance)**
- **NVDA**: Massive positive alpha (3.48%/month ≈ 42% annualized, highly significant) — AI boom outperformance beyond systematic risk factors
- **Most stocks**: Alpha not statistically significant — Returns explained by factor exposures, not stock selection skill

**2. Market Beta (Systematic Risk)**
- **High Beta** (β > 1.3): NVDA (1.76), GS (1.34) — amplify market swings
- **Medium Beta** (1.0 < β < 1.3): AAPL (1.18), JPM (1.12), MSFT (1.04)
- **Low Beta** (β < 1.0): XOM (0.86), WMT (0.54) — defensive stocks

**3. Style Factors (Value vs Growth)**
- **Value Stocks** (HML > 0): XOM (1.04), JPM (0.80), GS (0.69) — Energy and financials exhibit classic value characteristics
- **Growth Stocks** (HML < 0): NVDA (-0.97), MSFT (-0.49), AAPL (-0.45) — Tech sector shows strong growth orientation

**4. Model Fit (R²)**
- **High R²** (> 0.6): JPM (73%), GS (68%), MSFT (61%) — Returns primarily driven by systematic factors
- **Low R²** (< 0.5): WMT (21%), AAPL (48%), NVDA (42%) — Significant idiosyncratic risk (company-specific factors)

### Rolling Window Insights

![Systematic Risk Comparison](./Systematic_Risk_Comparison.png)

**Beta Instability:**

| Stock | Beta Range | Volatility | Stability Rating |
|-------|------------|------------|------------------|
| NVDA | 1.2 to 2.6 | Very High | ★☆☆☆☆ Unstable |
| AAPL | 0.7 to 1.4 | Moderate | ★★★☆☆ Moderate |
| JPM | 0.9 to 1.3 | Low | ★★★★☆ Stable |
| XOM | 0.4 to 1.1 | Moderate | ★★★☆☆ Declining |

**Key Patterns Observed:**

1. **NVDA**: Spiked to β = 2.6 in 2018-2019 (crypto boom), collapsed to β = 1.2 during COVID crash, rebuilt to β = 2.3 during AI boom (2021-2024), recent normalization to β = 1.6. **Extreme beta instability** — static estimate highly misleading.

2. **JPM**: Most stable: β ≈ 1.0 ± 0.2 throughout entire period with slight cyclicality but no regime shifts. **Static beta well-represents true risk profile.**

3. **XOM**: Declining trend from β = 1.0 (2018) → β = 0.4 (2025). Energy sector decoupling from market. **Growing diversification benefit.**

4. **AAPL**: Increased from β = 1.0 to β = 1.4 (2020-2024), then sharp drop to β = 0.7 (2025-2026). **Major recent regime shift** from market-average to defensive.

![Time-Varying Market Beta](./Rolling_Beta.png)

![Time-Varying Value/Growth Characteristics](./Rolling_HML.png)

**Style Drift Observations:**
- **Tech stocks**: Became MORE growth-oriented over time (HML more negative)
- **Value stocks**: Maintained stable value characteristics (XOM, JPM)
- **COVID impact**: Temporary disruption in all factor relationships

## Key Insights

### 1. Risk is NOT Static
The static beta assumption is violated. NVDA's beta ranged from 1.2 to 2.6 — using a single estimate (1.76) misses massive variation. Portfolio managers relying on static betas systematically underestimate risk during volatile regimes.

### 2. Clear Sector Risk Patterns
- **Technology**: High beta (1.0-1.8), growth-oriented (negative HML), moderate R²
- **Financials**: Medium-high beta (1.1-1.3), value characteristics (positive HML), high R²
- **Consumer/Energy**: Low beta (0.5-0.9), defensive or value, varying R²

### 3. Alpha is Rare
Only NVDA generated statistically significant alpha. Even this may reflect missing factors (AI revolution not in Fama-French), data mining (selecting NVDA ex-post), or luck rather than skill. Most stocks delivered returns consistent with their factor exposures — efficient market hypothesis supported.

### 4. COVID Disrupted Everything (Temporarily)
All stocks experienced beta disruption during 2020, but recovery paths diverged. Tech stocks increased market correlation (higher beta post-COVID), energy decreased correlation (lower beta post-COVID), and financials returned to baseline.

### 5. Model Fit Varies Dramatically
High R² stocks (JPM, GS) are factor-driven — suitable for factor-based portfolios. Low R² stocks (WMT, NVDA) carry significant idiosyncratic risk — require deeper company analysis.

### 6. Time-Varying Analysis is Essential
Static regression gives one answer. Rolling windows show when risk changed, how persistent changes are, and whether changes were temporary (COVID) or structural (energy decoupling). This information is actionable for dynamic portfolio rebalancing.

## Limitations

1. **Sample Size**: Limited to 7 large-cap stocks with survivorship bias and missing small/mid-cap representation.

2. **Factor Model Completeness**: 3-Factor model may miss important risks. Fama-French 5-factor adds profitability and investment factors. Momentum factor (Carhart 4-factor) not included.

3. **Regression Assumptions**: Assumes linear relationships and constant coefficients within each 36-month window. Ignores potential non-linearities.

4. **Statistical Power**: 36-month windows provide only 36 observations. Trade-off between adaptivity (short window) and precision (long window).

5. **Hindsight Bias**: Alpha may not persist out-of-sample. Past factor loadings don't guarantee future loadings.

## Tools and Libraries

- **Python 3.x** - Core programming language
- **pandas** - Time series data manipulation and analysis
- **numpy** - Numerical computations
- **yfinance** - Financial data retrieval from Yahoo Finance
- **pandas_datareader** - Access to Kenneth French Data Library
- **statsmodels** - OLS regression estimation and diagnostics
- **matplotlib** - Data visualization and plotting

## Future Enhancements

### Model Extensions
1. **Fama-French 5-Factor Model**: Add profitability (RMW) and investment (CMA) factors
2. **Carhart 4-Factor**: Include momentum factor for better growth stock modeling
3. **Time-Varying Covariance**: Implement DCC-GARCH for dynamic factor correlations

### Analysis Depth
4. **Out-of-Sample Testing**: Split data into train/test, validate predictive power
5. **Event Studies**: Analyze factor behavior around specific events (earnings, product launches)
6. **Structural Break Tests**: Formal statistical tests for regime changes (Chow test, Bai-Perron)

### Portfolio Applications
7. **Factor-Tilted Portfolio**: Construct portfolios based on factor exposures
8. **Risk Parity**: Equal risk contribution from each factor
9. **Dynamic Rebalancing**: Update portfolio weights as rolling betas change

### Broader Universe
10. **Sector Analysis**: Extend to 50-100 stocks across all sectors
11. **International Factors**: Apply to global markets (Europe, Asia)
12. **Small-Cap Focus**: Analyze where size premium is strongest


## Author

**Asad Ali Akhtar**

---

**Disclaimer**: This project is for educational purposes only. Factor exposures estimated from historical data do not guarantee future performance. Always conduct thorough due diligence before making investment decisions.

---

*Last Updated: February 2026*
