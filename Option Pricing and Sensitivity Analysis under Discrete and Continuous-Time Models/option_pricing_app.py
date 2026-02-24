import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- CORE PRICING FUNCTIONS (Extracted & Cleaned from your Project) ---

def black_scholes(S, K, T, r, sigma, opt_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "Call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def binomial_tree(S, K, T, r, sigma, N, opt_type, exercise="European"):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u [cite: 16]
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    S_tree = S * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1, 1))
    
    # Option value at maturity
    if opt_type == "Call":
        C = np.maximum(S_tree - K, 0)
    else:
        C = np.maximum(K - S_tree, 0)
        
    # Backward induction
    for j in range(N - 1, -1, -1):
        C = np.exp(-r * dt) * (p * C[:-1] + (1 - p) * C[1:]) 
        if exercise == "American":
            S_curr = S * (u ** np.arange(j, -1, -1)) * (d ** np.arange(0, j + 1, 1))
            if opt_type == "Call":
                C = np.maximum(C, S_curr - K)
            else:
                C = np.maximum(C, K - S_curr)
    return C[0]

def trinomial_tree(S, K, T, r, sigma, N, opt_type, exercise="European"):
    dt = T / N 
    u = np.exp(sigma * np.sqrt(2 * dt))
    d = 1 / u 
    
    # Probability calculations based on project logic
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
        C = np.exp(-r * dt) * (pu * C[:-2] + pm * C[1:-1] + pd * C[2:]) 
        if exercise == "American": 
            S_curr = S * (u ** np.arange(j, -j - 1, -1))
            if opt_type == "Call":
                C = np.maximum(C, S_curr - K)
            else:
                C = np.maximum(C, K - S_curr)
    return C[0]

def asian_monte_carlo(S, K, T, r, sigma, N, M, opt_type):
    dt = T / N 
    u = np.exp(sigma * np.sqrt(dt)) 
    d = 1 / u 
    p = (np.exp(r * dt) - d) / (u - d) 
    
    payoffs = []
    for _ in range(M): 
        path = [S]
        for _ in range(N):
            if np.random.rand() < p:
                path.append(path[-1] * u)
            else:
                path.append(path[-1] * d)
        avg_price = np.mean(path) [cite: 380]
        if opt_type == "Call":
            payoffs.append(max(avg_price - K, 0)) 
        else:
            payoffs.append(max(K - avg_price, 0)) 
            
    return np.exp(-r * T) * np.mean(payoffs) 

# --- STREAMLIT UI SETUP ---

st.set_page_config(layout="wide", page_title="Option Pricing Dashboard")
st.title("📊 Option Pricing & Sensitivity Analysis")

# Sidebar Configuration (Inputs)
with st.sidebar:
    st.header("Parameters")
    
    # 1. Option Style
    opt_style = st.selectbox("Option Style", ["European", "American", "Asian"]) 
    
    # 2. Method (Conditional Logic)
    if opt_style == "European":
        method = st.selectbox("Method", ["Binomial", "Trinomial", "Black-Scholes"]) 
    elif opt_style == "American":
        method = st.selectbox("Method", ["Binomial", "Trinomial"]) 
    else:
        method = "Monte Carlo"
        st.info("Asian options priced via Monte Carlo simulation.") 

    # 3. Call/Put
    opt_type = st.selectbox("Type", ["Call", "Put"]) 
    
    # 4. Numerical Inputs
    S0 = st.number_input("Stock Price (S0)", value=100.0)
    K = st.number_input("Strike Price (K)", value=90.0)
    T = st.number_input("Time to Maturity (Years)", value=1.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    sigma = st.number_input("Volatility (σ)", value=0.3)
    
    if method != "Black-Scholes":
        N = st.slider("Time Steps (N)", 10, 500, 100)
    if opt_style == "Asian":
        M = st.number_input("Simulations (M)", value=5000)

# --- CALCULATION LOGIC ---

if st.sidebar.button("Run Analysis"):
    # Calculate Price
    if opt_style == "Asian":
        price = asian_monte_carlo(S0, K, T, r, sigma, N, M, opt_type)
    elif method == "Black-Scholes":
        price = black_scholes(S0, K, T, r, sigma, opt_type)
    elif method == "Binomial":
        price = binomial_tree(S0, K, T, r, sigma, N, opt_type, opt_style)
    else:
        price = trinomial_tree(S0, K, T, r, sigma, N, opt_type, opt_style)

    # Display Result
    st.metric(label=f"{opt_style} {opt_type} Price ({method})", value=f"${price:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Payoff Diagram")
        s_range = np.linspace(S0 * 0.5, S0 * 1.5, 100)
        if opt_type == "Call":
            payoff = np.maximum(s_range - K, 0)
        else:
            payoff = np.maximum(K - s_range, 0)
        
        fig, ax = plt.subplots()
        ax.plot(s_range, payoff, label='Payoff', color='blue')
        ax.axvline(K, color='red', linestyle='--', label='Strike')
        ax.set_xlabel("Stock Price")
        ax.set_ylabel("Profit/Loss")
        ax.legend()
        st.pyplot(fig)

    with col2:
        # Convergence Plot (Only for Tree Methods)
        if method in ["Binomial", "Trinomial"]:
            st.subheader("Convergence Analysis")
            steps = [10, 20, 50, 100, 150, 200] 
            prices = []
            for s in steps:
                if method == "Binomial":
                    prices.append(binomial_tree(S0, K, T, r, sigma, s, opt_type, opt_style))
                else:
                    prices.append(trinomial_tree(S0, K, T, r, sigma, s, opt_type, opt_style))
            
            fig2, ax2 = plt.subplots()
            ax2.plot(steps, prices, marker='o', linestyle='-') 
            ax2.set_xlabel("Number of Steps (N)") 
            ax2.set_ylabel("Option Price") 
            st.pyplot(fig2)
        else:
            st.info("Convergence plot is only applicable for discrete tree models.")
