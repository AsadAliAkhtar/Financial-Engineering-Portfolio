# Kelly Criterion Portfolio Optimization — Pakistan Oil & Gas Sector

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

A quantitative portfolio optimization project that implements the **Kelly Criterion** (continuous multi-asset extension) on Pakistan Stock Exchange (PSX) Oil & Gas stocks using Half Kelly allocation, Ledoit-Wolf shrinkage covariance estimation, and practical position constraints.

![Project Banner](images/banner.png)
<!-- Replace with your own banner image or remove this line -->

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Methodology](#methodology)
- [Data Quality](#data-quality)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Visualizations](#results--visualizations)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

The **Kelly Criterion**, developed by John L. Kelly Jr. (1956), determines the optimal capital allocation to maximize long-term geometric wealth growth. For a portfolio of $n$ correlated assets, the multi-asset Kelly solves:

$$f^* = \arg\max_w \left[ w^\top (\mu - r_f) - \frac{1}{2} w^\top \Sigma w \right]$$

This is equivalent to **mean-variance optimization with risk aversion λ = 1**. This project applies the framework to 8 Oil & Gas stocks listed on the Pakistan Stock Exchange over a 6-year period (2020–2025).

### Why This Project?

- **Emerging market application** — Kelly Criterion is rarely applied to frontier markets like Pakistan, where the risk-free rate (13.51%) creates a high hurdle for equity allocation
- **End-to-end pipeline** — from raw data download through data quality screening, covariance estimation, optimization, and portfolio performance reporting
- **Practical constraints** — demonstrates the trade-off between theoretical Kelly optimality and real-world position limits

---

## Key Results

| Metric | Value |
|--------|-------|
| **Universe** | 8 PSX Oil & Gas stocks (MARI excluded — data issue) |
| **Time Horizon** | Jan 2020 – Dec 2025 |
| **Risk-Free Rate** | 13.51% (avg. Pakistan T-bill) |
| **Strategy** | Half Kelly with 20% position cap |
| **Expected Return** | 17.24% |
| **Portfolio Volatility** | 15.15% |
| **Sharpe Ratio** | 0.2464 |
| **Implied Cash Position** | 50% |

### Half Kelly Allocation (20% Cap)

| Stock | Weight | Sector Role |
|-------|--------|-------------|
| APL.KA | 10.0% | Exploration & Production |
| ATRL.KA | 10.0% | Refinery |
| NRL.KA | 10.0% | Refinery |
| OGDC.KA | 10.0% | Exploration & Production |
| PSO.KA | 10.0% | Marketing & Distribution |
| PPL.KA | 0.0% | Negative Sharpe |
| PRL.KA | 0.0% | Near-zero Sharpe |
| SSGC.KA | 0.0% | Negative Sharpe |
| **Cash** | **50.0%** | **Risk-free (T-bills)** |

---

## Methodology

### 1. Data Collection & Cleaning

- Downloaded adjusted closing prices via **yfinance** (`auto_adjust=True`)
- Removed 68 duplicate date entries (forward-filled weekends/holidays)
- Computed **log returns** (not arithmetic) — required for Kelly's log-wealth maximization

### 2. Data Quality Screening

Identified and excluded **MARI.KA** due to an unadjusted 9:1 stock split on September 5, 2024, which created a spurious -89% daily return:

![MARI Stock Split](images/mari_price_context.png)
<!-- Screenshot of the price context around 2024-09-05 showing the split -->

### 3. Covariance Estimation

Used **Ledoit-Wolf shrinkage** to estimate the covariance matrix:

$$\hat{\Sigma}_{LW} = (1 - \delta)\hat{\Sigma}_{sample} + \delta \cdot F$$

- Shrinkage coefficient: **δ = 0.59%** (sample covariance highly reliable with ~1,500 obs / 8 assets)
- Compared simple vs. shrunk eigenvalue spectra to validate matrix conditioning

![Eigenvalue Spectrum](images/eigenvalue_spectrum.png)
<!-- Screenshot of the eigenvalue spectrum plot -->

### 4. Kelly Optimization

- Solved `max_quadratic_utility(risk_aversion=1.0)` via **pypfopt** for full Kelly weights
- Applied **Half Kelly** (×0.5 scaling) to reduce sensitivity to estimation error
- Imposed **20% position cap** to manage concentration risk

### 5. Portfolio Performance

Computed total portfolio return including cash position earning the risk-free rate:

$$R_p = w^\top \mu + (1 - \mathbf{1}^\top w) \cdot r_f$$

---

## Data Quality

### MARI.KA Exclusion

| Date | Price (PKR) | Event |
|------|-------------|-------|
| 2024-09-04 | 3,351.62 | Pre-split |
| 2024-09-05 | 367.22 | Post-split (~9:1) |
| 2024-09-06 | 371.43 | Stable post-split |

**Impact of the outlier:**
- Inflated MARI annualized volatility to **94.8%** (vs. 28–49% for peers)
- Pushed Ledoit-Wolf shrinkage from **0.59% → 62.5%**
- Zero Kelly weight regardless (negative Sharpe), but distorted covariance for all other stocks

---

## Project Structure

```
kelly-criterion-psx-oil-gas/
│
├── Kelly_Criterion_Implementation.ipynb    # Main Jupyter notebook
├── data_series.csv                         # Pakistan T-bill rate data
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
│
└── images/                                # Visualizations for README
    ├── banner.png                         # Project banner
    ├── growth_of_1pkr.png                 # Cumulative growth chart
    ├── correlation_heatmap.png            # Stock correlation matrix
    ├── eigenvalue_spectrum.png            # Simple vs. Ledoit-Wolf eigenvalues
    ├── mari_price_context.png             # MARI stock split evidence
    └── portfolio_weights.png              # Final allocation chart
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/kelly-criterion-psx-oil-gas.git
cd kelly-criterion-psx-oil-gas

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
pandas>=2.0
numpy>=1.24
yfinance>=0.2.30
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
pypfopt>=1.5
squarify>=0.4
```

---

## Usage

```bash
# Launch Jupyter and open the notebook
jupyter notebook Kelly_Criterion_Implementation.ipynb
```

Run all cells sequentially. The notebook:
1. Downloads price data from Yahoo Finance (requires internet)
2. Performs EDA and data quality screening
3. Estimates covariance using Ledoit-Wolf shrinkage
4. Computes Kelly and Half Kelly allocations
5. Reports portfolio performance metrics

---

## Results & Visualizations

### Growth of 1 PKR (2020–2025)

![Growth of 1 PKR](images/growth_of_1pkr.png)
<!-- Screenshot of the cumulative growth chart from cell [45] -->

ATRL.KA delivered the highest total return, turning 1 PKR into 7.03 PKR over 6 years.

### Correlation Heatmap

![Correlation Heatmap](images/correlation_heatmap.png)
<!-- Screenshot of the correlation heatmap from cell [48] -->

High intra-sector correlations (0.37–0.86) limit diversification within a single-sector universe. OGDC–PPL (0.86) and ATRL–NRL (0.78) are the most correlated pairs.

### Eigenvalue Spectrum

![Eigenvalue Spectrum](images/eigenvalue_spectrum.png)
<!-- Screenshot of the eigenvalue spectrum from cell [56] -->

Near-perfect overlap between simple and Ledoit-Wolf spectra confirms clean, well-conditioned data after removing MARI.

---

## Limitations

| Limitation | Description |
|------------|-------------|
| **In-sample estimation** | Returns and covariance estimated on the same data used for evaluation — no walk-forward validation |
| **Single-sector universe** | All 8 stocks are Oil & Gas — high correlations limit diversification |
| **Static risk-free rate** | Used average T-bill rate (13.51%) despite Pakistan's rate ranging from ~7% to ~22% |
| **No transaction costs** | Kelly assumes frictionless rebalancing |
| **No out-of-sample backtest** | Reported metrics are expected, not realized |
| **Data quality risk** | Other unadjusted corporate actions may exist beyond the identified MARI split |

---

## Future Work

- [ ] Walk-forward backtest with rolling estimation windows
- [ ] Expand universe to multi-sector PSX stocks
- [ ] Time-varying risk-free rate (monthly T-bill series)
- [ ] Transaction cost modeling for discrete rebalancing
- [ ] Kelly fraction sensitivity analysis (Quarter, Half, Three-Quarter, Full)
- [ ] Comparison with equal-weight and minimum-variance benchmarks

---

## References

1. Kelly, J.L. (1956). *A New Interpretation of Information Rate*. Bell System Technical Journal, 35(4), 917–926.
2. Thorp, E.O. (2006). *The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market*. Handbook of Asset and Liability Management.
3. Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis, 88(2), 365–411.
4. MacLean, L.C., Thorp, E.O., & Ziemba, W.T. (2011). *The Kelly Capital Growth Investment Criterion*. World Scientific.

---

## How to Add Images to This README

To insert screenshots from your notebook into this README:

1. **Take screenshots** of the relevant plots/outputs from your Jupyter notebook
2. **Save them** in the `images/` folder with the filenames referenced above
3. The image syntax in Markdown is:

```markdown
![Alt Text](images/filename.png)
```

**Recommended screenshots to capture:**
- `growth_of_1pkr.png` — Cell [45] growth chart
- `correlation_heatmap.png` — Cell [48] heatmap
- `eigenvalue_spectrum.png` — Cell [56] eigenvalue plot
- `mari_price_context.png` — Cell [40] output showing the price drop
- `portfolio_weights.png` — Create a bar chart of final weights (optional addition)
- `banner.png` — A custom banner (optional, can use the growth chart)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Author

**Asad** — MFE Candidate, WorldQuant University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black.svg)](https://github.com/YOUR_USERNAME)
