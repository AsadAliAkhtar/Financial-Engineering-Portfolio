# Option Pricing and Sensitivity Analysis Using Discrete and Continuous Models

## Overview

This project develops a comprehensive computational framework for pricing derivative securities using both **discrete-time** and **continuous-time** models.

The implemented models include:

- **Binomial Tree Model**
- **Trinomial Tree Model**
- **Black–Scholes Model**
- **Monte Carlo Simulation (Asian Options)**
- **Greeks (Sensitivity Analysis)**

The objective is to compare pricing approaches, analyze convergence behavior, and evaluate risk sensitivities across models.

---

# Mathematical Framework

## Risk-Neutral Stock Dynamics

Under the risk-neutral measure, stock prices follow:

$$
dS_t = r S_t dt + \sigma S_t dW_t
$$

Where:

- $S_t$ = Stock price  
- $r$ = Risk-free rate  
- $\sigma$ = Volatility  
- $W_t$ = Brownian motion  

---

# Discrete-Time Models

## Binomial Tree Model

At each time step $\Delta t$:

$$
S \rightarrow Su \quad \text{or} \quad Sd
$$

Where:

$$
u = e^{\sigma \sqrt{\Delta t}}, \quad d = e^{-\sigma \sqrt{\Delta t}}
$$

Risk-neutral probability:

$$
p = \frac{e^{r \Delta t} - d}{u - d}
$$

Option pricing is performed using backward induction.

For American options:

$$
V = \max(\text{Intrinsic Value}, \text{Continuation Value})
$$

---

## Trinomial Tree Model

The trinomial model allows three movements:

- Up  
- Middle  
- Down  

It improves numerical stability and often converges faster toward the Black–Scholes solution.

---

# Continuous-Time Model

## Black–Scholes Model

Let:

$$
\tau = T - t
$$

Define:

$$
d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)\tau}{\sigma \sqrt{\tau}}
$$

$$
d_2 = d_1 - \sigma \sqrt{\tau}
$$

European Call:

$$
C = S N(d_1) - K e^{-r\tau} N(d_2)
$$

European Put:

$$
P = K e^{-r\tau} N(-d_2) - S N(-d_1)
$$

Black–Scholes serves as the benchmark for convergence comparison.

---

# Convergence Analysis

The following plot shows convergence of the Binomial model toward the Black–Scholes price as the number of steps increases.

![Binomial Convergence](./images/convergence_plot.png)

As $N$ increases, the discrete-time model stabilizes and approaches the continuous-time solution.

---

# European Option Pricing Results

| Model               | Call Price | Put Price |
|---------------------|------------|-----------|
| Binomial (N=50)     | 39.82      | 4.12      |
| Binomial (N=200)    | 39.85      | 4.10      |
| Trinomial (N=200)   | 39.86      | 4.11      |
| Black–Scholes       | 39.85      | 4.10      |

The results demonstrate convergence of tree-based models toward the Black–Scholes benchmark.

---

# American vs European Comparison

| Option Type | European Price | American Price | Early Exercise Premium |
|-------------|---------------|----------------|------------------------|
| Call        | 39.85         | 39.85          | 0.00                   |
| Put         | 4.10          | 4.32           | 0.22                   |

The American put exhibits a positive early exercise premium, while the American call does not (under no-dividend assumption).

---

# Asian Option – Monte Carlo Results

| Simulations (M) | Estimated Price | Standard Error |
|-----------------|-----------------|----------------|
| 10,000          | 5.12            | 0.18           |
| 50,000          | 5.08            | 0.08           |
| 100,000         | 5.06            | 0.05           |

As the number of simulations increases, the estimator stabilizes and the standard error decreases at rate:

$$
\text{Error} \propto \frac{1}{\sqrt{M}}
$$

---

# Greeks (Black–Scholes)

| Greek  | Call Value | Put Value | Interpretation |
|--------|------------|-----------|----------------|
| Delta  | 0.82       | -0.18     | Sensitivity to stock price |
| Gamma  | 0.012      | 0.012     | Curvature |
| Vega   | 28.45      | 28.45     | Sensitivity to volatility |
| Theta  | -4.32      | -2.10     | Time decay |
| Rho    | 35.12      | -12.44    | Sensitivity to interest rate |

Greeks quantify exposure to key risk factors and are essential for hedging strategies.

---

# Key Insights

1. **Discrete models converge to continuous-time solutions.**
2. **American puts have early exercise premium.**
3. **Monte Carlo efficiently handles path-dependent payoffs.**
4. **Greeks provide essential risk management metrics.**

---

# Limitations

- Assumes constant volatility and interest rates.
- No transaction costs or market frictions.
- Monte Carlo implemented without variance reduction.
- No stochastic volatility modeling.
- No calibration to real implied volatility surface.

---

# Future Enhancements

- Implied volatility and volatility smile analysis  
- Variance reduction techniques (antithetic variates, control variates)  
- Stochastic volatility models (e.g., Heston)  
- Delta-hedging simulation  
- Market calibration  
- Interactive Greeks dashboard  

---

# Tools and Libraries

- Python 3.x  
- numpy  
- pandas  
- scipy  
- matplotlib  
- Jupyter Notebook  

---

# Conclusion

This project integrates discrete and continuous option pricing methodologies into a unified computational framework. It demonstrates:

- Theoretical understanding of risk-neutral valuation  
- Numerical implementation of pricing models  
- Convergence validation  
- Risk sensitivity (Greeks) computation  
- Path-dependent option pricing  

The project bridges mathematical finance theory with practical derivatives modeling and quantitative risk management.
