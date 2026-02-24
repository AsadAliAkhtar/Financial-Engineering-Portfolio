import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. CORE FUNCTIONS (EXACTLY AS DEFINED IN YOUR PDF) ---

def binomial_eu_option(S_in, K, T, r, sigma, N, opt_type):
    """Prices European Options using Binomial Tree[cite: 28]."""
    dt = T / N # [cite: 30]
    u = np.exp(sigma * np.sqrt(dt)) # [cite: 31]
    d = np.exp(-sigma * np.sqrt(dt)) # [cite: 32]
    p = (np.exp(r * dt) - d) / (u - d) # [cite: 33]
    S = np.zeros([N + 1, N + 1])
    C = np.zeros([N + 1, N + 1])
    for i in range(0, N + 1):
        S[N, i] = S_in * u**i * d**(N - i) # [cite: 40]
        if opt_type == 'C':
            C[N, i] = max(S[N, i] - K, 0) # [cite: 44]
        else:
            C[N, i] = max(K - S[N, i], 0) # [cite: 45]
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            C[j, i] = np.exp(-r * dt) * (p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i]) # [cite: 50]
    return C[0, 0]

def trinomial_eu_option(S_in, K, T, r, sigma, N, opt_type):
    """Prices European Options using Trinomial Tree[cite: 94]."""
    dt = T / N # [cite: 96]
    u = np.exp(sigma * np.sqrt(2 * dt)) # [cite: 97]
    d = 1 / u # [cite: 98]
    pu = ((np.exp(r * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2))) / (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2))))**2 # [cite: 86]
    pd = ((np.exp(sigma * np.sqrt(dt / 2)) - np.exp(r * dt / 2)) / (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2))))**2 # [cite: 88]
    pm = 1 - pu - pd # [cite: 90]
    
    # Stock prices at expiration
    S_final = S_in * (u ** np.arange(N, -N - 1, -1))
    if opt_type == "C":
        nxt_vec_prices = np.maximum(S_final - K, 0) # [cite: 133]
    else:
        nxt_vec_prices = np.maximum(K - S_final, 0) # [cite: 135]
        
    for i in range(1, N + 1):
        expectation = (pd * nxt_vec_prices[:-2] + pm * nxt_vec_prices[1:-1] + pu * nxt_vec_prices[2:]) # [cite: 148-152]
        nxt_vec_prices = np.exp(-r * dt) * expectation # [cite: 155]
    return nxt_vec_prices[0]

def black_scholes(S_in, K, T, t, r, sigma, opt_type):
    """Calculates Fair Value using Continuous-Time Formula[cite: 394]."""
    time_to_maturity = T - t # [cite: 391]
    if time_to_maturity <= 0: return max(S_in - K, 0) if opt_type == "C" else max(K - S_in, 0)
    d1 = (np.log(S_in / K) + (r + sigma**2 / 2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity)) # [cite: 394]
    d2 = d1 - sigma * np.sqrt(time_to_maturity) # [cite: 394]
    if opt_type == "C":
        return (S_in * norm.cdf(d1)) - (K * np.exp(-r * time_to_maturity) * norm.cdf(d2)) # [cite: 392]
    else:
        return (K * np.exp(-r * time_to_maturity) * norm.cdf(-d2)) - (S_in * norm.cdf(-d1)) # [cite: 393]

def black_scholes_greek(S, K, T, t, r, sigma, opt_type):
    """Computes Greeks for Risk Sensitivity Analysis."""
    time_to_maturity = T - t
    if time_to_maturity <= 0: return 0.0, 0.0, 0.0, 0.0, 0.0
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
    d2 = d1 - sigma * np.sqrt(time_to_maturity)
    
    # Shared Gamma and Vega
    Gamma = norm.pdf(d1) / (S * sigma * np.sqrt(time_to_maturity)) # [cite: 400]
    Vega = S * norm.pdf(d1) * np.sqrt(time_to_maturity) # [cite: 401]
    
    if opt_type == "C":
        Delta = norm.cdf(d1) # [cite: 399]
        Theta = -((S * norm.pdf(d1) * sigma) / (2 * np.sqrt(time_to_maturity))) - (r * K * np.exp(-r * time_to_maturity) * norm.cdf(d2)) # [cite: 402]
        Rho = K * time_to_maturity * np.exp(-r * time_to_maturity) * norm.cdf(d2) # 
    else:
        Delta = norm.cdf(d1) - 1 # [cite: 399]
        Theta = -((S * norm.pdf(d1) * sigma) / (2 * np.sqrt(time_to_maturity))) + (r * K * np.exp(-r * time_to_maturity) * norm.cdf(-d2)) # [cite: 402]
        Rho = -(K * time_to_maturity * np.exp(-r * time_to_maturity) * norm.cdf(-d2)) # 
    return Delta, Gamma, Vega, Theta, Rho

# --- 2. DASHBOARD UI SETUP ---

st.set_page_config(layout="wide")
st.title("📊 Project Dashboard: Option Pricing & Greeks")

with st.sidebar:
    st.header("Project Parameters")
    opt_style = st.selectbox("Option Style", ["European", "American", "Asian"])
    method = st.selectbox("Method", ["Binomial", "Trinomial", "Black-Scholes"]) if opt_style == "European" else "Other"
    opt_type_label = st.selectbox("Call/Put", ["Call", "Put"])
    opt_type = 'C' if opt_type_label == "Call" else 'P'
    
    S0 = st.number_input("S0", value=100.0)
    K = st.number_input("K", value=90.0)
    T = st.number_input("T (Years)", value=10.0)
    r = st.number_input("r", value=0.0)
    sigma = st.number_input("σ", value=0.3)
    N = st.slider("Steps (N)", 10, 2000, 100)

# --- 3. CALCULATION AND GREEKS DISPLAY ---

if st.sidebar.button("Run Model"):
    # Calculate Price
    if method == "Black-Scholes":
        price = black_scholes(S0, K, T, 0, r, sigma, opt_type)
        # UPDATED: Display Greeks specifically for Black-Scholes
        delta, gamma, vega, theta, rho = black_scholes_greek(S0, K, T, 0, r, sigma, opt_type)
        
        st.success(f"Black-Scholes {opt_type_label} Price: ${price:.4f}")
        
            st.subheader("Black-Scholes Greeks")
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Delta (Δ)", f"{delta:.4f}")
        g2.metric("Gamma (Γ)", f"{gamma:.4f}")
        g3.metric("Vega (ν)", f"{vega:.4f}")
        g4.metric("Theta (θ)", f"{theta:.4f}")
        g5.metric("Rho (ρ)", f"{rho:.4f}")
        
    elif method == "Binomial":
        price = binomial_eu_option(S0, K, T, r, sigma, N, opt_type)
        st.success(f"Binomial {opt_type_label} Price: ${price:.4f}")
    else:
        price = trinomial_eu_option(S0, K, T, r, sigma, N, opt_type)
        st.success(f"Trinomial {opt_type_label} Price: ${price:.4f}")

    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Payoff Diagram")
        s_range = np.linspace(S0 * 0.5, S0 * 1.5, 100)
        payoff = np.maximum(s_range - K, 0) if opt_type == 'C' else np.maximum(K - s_range, 0)
        fig, ax = plt.subplots(); ax.plot(s_range, payoff); ax.axvline(K, color='r', ls='--'); st.pyplot(fig)

    with col2:
        if method in ["Binomial", "Trinomial"]:
            st.subheader(f"Convergence Plot (up to N={N})")
            test_steps = np.unique(np.linspace(10, N, 12, dtype=int))
            conv = [binomial_eu_option(S0, K, T, r, sigma, s, opt_type) if method == "Binomial" 
                    else trinomial_eu_option(S0, K, T, r, sigma, s, opt_type) for s in test_steps]
            fig2, ax2 = plt.subplots(); ax2.plot(test_steps, conv, marker='o', color='orange'); st.pyplot(fig2)
