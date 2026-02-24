import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. CORE PRICING FUNCTIONS (Directly from Project PDF) ---

def black_scholes(S, K, T, r, sigma, opt_type):
    # Formulas for d1 and d2 [cite: 808]
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "Call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) # [cite: 806]
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) # [cite: 807]

def binomial_tree(S, K, T, r, sigma, N, opt_type, exercise="European"):
    dt = T / N # [cite: 432]
    u = np.exp(sigma * np.sqrt(dt)) # [cite: 429]
    d = np.exp(-sigma * np.sqrt(dt)) # [cite: 430]
    p = (np.exp(r * dt) - d) / (u - d) # [cite: 434]
    
    # Initialize terminal stock prices
    S_tree = S * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1, 1))
    
    # Option value at maturity [cite: 436, 439]
    if opt_type == "Call":
        C = np.maximum(S_tree - K, 0)
    else:
        C = np.maximum(K - S_tree, 0)
        
    # Backward induction [cite: 441]
    for j in range(N - 1, -1, -1):
        C = np.exp(-r * dt) * (p * C[:-1] + (1 - p) * C[1:])
        if exercise == "American":
            S_curr = S * (u ** np.arange(j, -1, -1)) * (d ** np.arange(0, j + 1, 1))
            if opt_type == "Call":
                C = np.maximum(C, S_curr - K) # [cite: 630, 662]
            else:
                C = np.maximum(C, K - S_curr) # [cite: 630, 663]
    return C[0]

def trinomial_tree(S, K, T, r, sigma, N, opt_type, exercise="European"):
    dt = T / N # [cite: 510]
    u = np.exp(sigma * np.sqrt(2 * dt)) # [cite: 498]
    d = 1 / u # [cite: 498]
    
    # Probabilities [cite: 500, 502, 504]
    edr = np.exp(r * dt / 2)
    esig = np.exp(sigma * np.sqrt(dt / 2))
    pu = ((edr - 1/esig) / (esig - 1/esig))**2
    pd = ((esig - edr) / (esig - 1/esig))**2
    pm = 1 - pu - pd
    
    # Grid initialization
    S_tree = S * (u ** np.arange(N, -N - 1, -1))
    if opt_type == "Call":
        C = np.maximum(S_tree - K, 0)
    else:
        C = np.maximum(K - S_tree, 0)
        
    for j in range(N - 1, -1, -1):
        # [cite: 506]
        C = np.exp(-r * dt) * (pu * C[:-2] + pm * C[1:-1] + pd * C[2:])
        if exercise == "American":
            S_curr = S * (u ** np.arange(j, -j - 1, -1))
            if opt_type == "Call":
                C = np.maximum(C, S_curr - K) # [cite: 741]
            else:
                C = np.maximum(C, K - S_curr) # [cite: 741]
    return C[0]

def asian_mc(S, K, T, r, sigma, N, M, opt_type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    
    payoffs = []
    for _ in range(M):
        path = [S]
        for _ in range(N):
            if np.random.rand() < p:
                path.append(path[-1] * u)
            else:
                path.append(path[-1] * d)
        avg_price = np.mean(path) # [cite: 765, 794]
        if opt_type == "Call":
            payoffs.append(max(avg_price - K, 0)) # [cite: 769]
        else:
            payoffs.append(max(K - avg_price, 0)) # [cite: 770]
            
    return np.exp(-r * T) * np.mean(payoffs) # [cite: 771]

# --- 2. STREAMLIT UI SETUP ---

st.set_page_config(layout="wide", page_title="Option Pricing Dashboard")
st.title("📊 Option Pricing & Convergence Analysis")

# Sidebar Configuration
with st.sidebar:
    st.header("Project Parameters")
    
    # Selection logic requested by user
    opt_style = st.selectbox("Option Style", ["European", "American", "Asian"])
    
    if opt_style == "European":
        method = st.selectbox("Method", ["Binomial", "Trinomial", "Black-Scholes"])
    elif opt_style == "American":
        method = st.selectbox("Method", ["Binomial", "Trinomial"])
    else:
        method = "Monte Carlo" # Method hidden for Asian as requested
        st.info("Asian options priced via Monte Carlo simulation.")

    opt_type = st.selectbox("Call/Put", ["Call", "Put"])
    
    # Input Sliders & Values
    S0 = st.number_input("Current Stock Price (S0)", value=100.0)
    K = st.number_input("Strike Price (K)", value=90.0)
    T = st.number_input("Time to Maturity (Years)", value=1.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    sigma = st.number_input("Volatility (σ)", value=0.3)
    
    # UPDATED: Max limit increased to 2000
    if method != "Black-Scholes":
        N = st.slider("Time Steps (N)", 10, 2000, 100)
    else:
        N = 100 # Default if BS selected
        
    if opt_style == "Asian":
        M = st.number_input("Simulations (M)", value=5000)

# --- 3. CALCULATION AND PLOTTING ---

if st.sidebar.button("Run Model"):
    # Calculate current price
    if opt_style == "Asian":
        price = asian_mc(S0, K, T, r, sigma, N, M, opt_type)
    elif method == "Black-Scholes":
        price = black_scholes(S0, K, T, r, sigma, opt_type)
    elif method == "Binomial":
        price = binomial_tree(S0, K, T, r, sigma, N, opt_type, opt_style)
    else:
        price = trinomial_tree(S0, K, T, r, sigma, N, opt_type, opt_style)

    st.success(f"Estimated {opt_style} {opt_type} Price: ${price:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Intrinsic Payoff Diagram")
        s_range = np.linspace(S0 * 0.5, S0 * 1.5, 100)
        payoff = np.maximum(s_range - K, 0) if opt_type == "Call" else np.maximum(K - s_range, 0)
        
        fig, ax = plt.subplots()
        ax.plot(s_range, payoff, label='Intrinsic Value', color='#1f77b4')
        ax.axvline(K, color='red', linestyle='--', label=f'Strike (${K})')
        ax.set_xlabel("Stock Price")
        ax.set_ylabel("Payoff")
        ax.legend()
        st.pyplot(fig)

    with col2:
        # UPDATED: Dynamic Convergence Plot
        if method in ["Binomial", "Trinomial"]:
            st.subheader(f"Convergence Analysis (up to N={N})")
            
            # Dynamically create testing steps up to the user-selected N 
            # We use 12 points to keep it efficient even at N=2000
            test_steps = np.unique(np.linspace(10, N, 12, dtype=int))
            
            conv_prices = []
            progress_bar = st.progress(0)
            for i, s in enumerate(test_steps):
                if method == "Binomial":
                    val = binomial_tree(S0, K, T, r, sigma, s, opt_type, opt_style)
                else:
                    val = trinomial_tree(S0, K, T, r, sigma, s, opt_type, opt_style)
                conv_prices.append(val)
                progress_bar.progress((i + 1) / len(test_steps))
            
            fig2, ax2 = plt.subplots()
            ax2.plot(test_steps, conv_prices, marker='o', linestyle='-', color='orange')
            ax2.set_xlabel("Number of Steps (N)")
            ax2.set_ylabel("Option Price")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            st.caption("As N increases, the model converges to the continuous-time value[cite: 423, 620].")
        else:
            st.info("Convergence analysis is used for discrete tree models.")
