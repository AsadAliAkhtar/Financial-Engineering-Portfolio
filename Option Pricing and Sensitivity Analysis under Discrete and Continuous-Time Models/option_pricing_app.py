import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

st.set_page_config(
    page_title="Option Pricing Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #2d3748; }
    .price-card {
        background: linear-gradient(135deg, #1a2332, #243447);
        border: 1px solid #3d5a80; border-radius: 12px;
        padding: 24px; text-align: center; margin-bottom: 16px;
    }
    .price-card h2 { color: #63b3ed; font-size: 1rem; margin: 0 0 8px; letter-spacing: 0.08em; text-transform: uppercase; }
    .price-card h1 { color: #f0f4f8; font-size: 2.8rem; margin: 0; font-weight: 700; }
    .greek-card {
        background: #161b27; border: 1px solid #2d3748;
        border-radius: 10px; padding: 16px; text-align: center; margin-bottom: 12px;
    }
    .greek-card .greek-name { color: #a0aec0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; }
    .greek-card .greek-value { color: #68d391; font-size: 1.5rem; font-weight: 700; }
    .info-box {
        background: #1a2332; border-left: 4px solid #3d5a80;
        border-radius: 0 8px 8px 0; padding: 12px 16px;
        margin: 8px 0; font-size: 0.85rem; color: #a0aec0;
    }
    .info-box strong { color: #90cdf4; }
    .section-header {
        color: #90cdf4; font-size: 1.1rem; font-weight: 600;
        margin: 20px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #2d3748;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Pricing Functions
# ─────────────────────────────────────────────

def binomial_eu_option(S_in, K, T, r, sigma, N, opt_type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    C = np.zeros([N + 1, N + 1])
    for i in range(N + 1):
        S_T = S_in * u ** i * d ** (N - i)
        C[N, i] = max(S_T - K, 0) if opt_type == "C" else max(K - S_T, 0)
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            C[j, i] = np.exp(-r * dt) * (p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i])
    return C[0, 0]


def trinomial_eu_option(S_in, K, T, r, sigma, N, opt_type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(2 * dt))
    d = 1 / u
    pu = ((np.exp(r * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2))) /
          (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
    pd = ((-np.exp(r * dt / 2) + np.exp(sigma * np.sqrt(dt / 2))) /
          (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
    pm = 1 - pu - pd

    def asset_price(nb):
        if nb == 0:
            return np.array([S_in])
        vec_u = np.cumprod(u * np.ones(nb))
        vec_d = np.cumprod(d * np.ones(nb))
        return np.concatenate((vec_d[::-1], [1.0], vec_u)) * S_in

    payoff = np.maximum(asset_price(N) - K, 0) if opt_type == "C" else np.maximum(K - asset_price(N), 0)
    nxt = payoff
    for i in range(1, N + 1):
        S_cur = asset_price(N - i)
        exp_val = np.array([pd * nxt[j] + pm * nxt[j+1] + pu * nxt[j+2] for j in range(len(S_cur))])
        nxt = np.exp(-r * dt) * exp_val
    return nxt[0]


def binomial_american_options(S_in, K, T, r, sigma, N, opt_type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    S = np.zeros([N + 1, N + 1])
    C = np.zeros([N + 1, N + 1])
    for i in range(N + 1):
        S[N, i] = S_in * u ** i * d ** (N - i)
        C[N, i] = max(S[N, i] - K, 0) if opt_type == "C" else max(K - S[N, i], 0)
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            S[j, i] = S_in * u ** i * d ** (j - i)
            C[j, i] = np.exp(-r * dt) * (p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i])
            intrinsic = S[j, i] - K if opt_type == "C" else K - S[j, i]
            C[j, i] = max(C[j, i], intrinsic)
    return C[0, 0]


def trinomial_american_option(S_in, K, T, r, sigma, N, opt_type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(2 * dt))
    d = 1 / u
    pu = ((np.exp(r * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2))) /
          (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
    pd = ((-np.exp(r * dt / 2) + np.exp(sigma * np.sqrt(dt / 2))) /
          (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
    pm = 1 - pu - pd

    def asset_price(nb):
        if nb == 0:
            return np.array([S_in])
        vec_u = np.cumprod(u * np.ones(nb))
        vec_d = np.cumprod(d * np.ones(nb))
        return np.concatenate((vec_d[::-1], [1.0], vec_u)) * S_in

    S_final = asset_price(N)
    payoff = np.maximum(S_final - K, 0) if opt_type == "C" else np.maximum(K - S_final, 0)
    nxt = payoff
    for i in range(1, N + 1):
        S_cur = asset_price(N - i)
        continuation = np.exp(-r * dt) * np.array([
            pd * nxt[j] + pm * nxt[j+1] + pu * nxt[j+2] for j in range(len(S_cur))
        ])
        intrinsic = np.maximum(S_cur - K, 0) if opt_type == "C" else np.maximum(K - S_cur, 0)
        nxt = np.maximum(continuation, intrinsic)
    return nxt[0]


def black_scholes(S_in, K, T, r, sigma, opt_type):
    if T <= 0:
        return max(S_in - K, 0) if opt_type == "C" else max(K - S_in, 0)
    d1 = (np.log(S_in / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "C":
        return S_in * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S_in * norm.cdf(-d1)


def black_scholes_greeks(S, K, T, r, sigma, opt_type):
    if T <= 0:
        return {"Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0}
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    if opt_type == "C":
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}


def american_numerical_greeks(S, K, T, r, sigma, N, opt_type, method):
    """Finite-difference Greeks for American options — no closed form exists."""
    pricer = binomial_american_options if method == "Binomial" else trinomial_american_option
    dS   = S * 0.01
    dsig = sigma * 0.01
    dr   = 0.0001
    dT   = T * 0.01
    p0   = pricer(S, K, T, r, sigma, N, opt_type)
    p_up = pricer(S + dS, K, T, r, sigma, N, opt_type)
    p_dn = pricer(S - dS, K, T, r, sigma, N, opt_type)
    delta = (p_up - p_dn) / (2 * dS)
    gamma = (p_up - 2 * p0 + p_dn) / (dS ** 2)
    vega  = (pricer(S, K, T, r, sigma + dsig, N, opt_type) -
             pricer(S, K, T, r, sigma - dsig, N, opt_type)) / (2 * dsig)
    theta = (pricer(S, K, max(T - dT, 1e-6), r, sigma, N, opt_type) - p0) / dT
    rho   = (pricer(S, K, T, r + dr, sigma, N, opt_type) -
             pricer(S, K, T, r - dr, sigma, N, opt_type)) / (2 * dr)
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}


def asian_option_mc(S_in, K, T, r, sigma, N, M, opt_type, seed=42):
    np.random.seed(seed)
    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = np.exp(-sigma * np.sqrt(dt))
    p  = (np.exp(r * dt) - d) / (u - d)
    payoffs, sample_paths = [], []
    for j in range(M):
        path, S_cur = [S_in], S_in
        for _ in range(N):
            S_cur = S_cur * (u if np.random.binomial(1, p) else d)
            path.append(S_cur)
        avg = np.mean(path)
        payoff = np.exp(-r * T) * (max(avg - K, 0) if opt_type == "C" else max(K - avg, 0))
        payoffs.append(payoff)
        if j < 50:
            sample_paths.append(path)
    return np.mean(payoffs), np.std(payoffs) / np.sqrt(M), sample_paths


# ─────────────────────────────────────────────
# Convergence Helpers
# ─────────────────────────────────────────────

def get_convergence_european(S, K, T, r, sigma, opt_type, method):
    steps = [2, 5, 10, 25, 50, 100, 200, 300, 500, 750, 1000]
    pricer = binomial_eu_option if method == "Binomial" else trinomial_eu_option
    prices = []
    for n in steps:
        try:
            prices.append(pricer(S, K, T, r, sigma, n, opt_type))
        except Exception:
            prices.append(np.nan)
    return steps, prices, black_scholes(S, K, T, r, sigma, opt_type)


def get_convergence_american(S, K, T, r, sigma, opt_type):
    steps = [2, 5, 10, 25, 50, 100, 200, 300, 500, 750, 1000]
    bin_prices, tri_prices = [], []
    for n in steps:
        try:
            bin_prices.append(binomial_american_options(S, K, T, r, sigma, n, opt_type))
        except Exception:
            bin_prices.append(np.nan)
        try:
            tri_prices.append(trinomial_american_option(S, K, T, r, sigma, n, opt_type))
        except Exception:
            tri_prices.append(np.nan)
    return steps, bin_prices, tri_prices


# ─────────────────────────────────────────────
# Plot Helpers
# ─────────────────────────────────────────────

def dark_fig(figsize=(9, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1117")
    ax.set_facecolor("#161b27")
    for spine in ax.spines.values():
        spine.set_color("#2d3748")
    ax.tick_params(colors="#a0aec0")
    ax.xaxis.label.set_color("#a0aec0")
    ax.yaxis.label.set_color("#a0aec0")
    ax.title.set_color("#90cdf4")
    ax.grid(True, color="#2d3748", linewidth=0.6, linestyle="--")
    return fig, ax


def plot_convergence_european(steps, prices, bs_price, method, opt_type_label):
    fig, ax = dark_fig((9, 4))
    ax.plot(steps, prices, color="#63b3ed", linewidth=2, marker="o", markersize=5, label=f"{method} Tree")
    ax.axhline(bs_price, color="#f6ad55", linestyle="--", linewidth=1.5, label="Black-Scholes")
    ax.set_xlabel("Number of Steps (N)")
    ax.set_ylabel("Option Price")
    ax.set_title(f"European {opt_type_label} — {method} Convergence vs Black-Scholes")
    ax.legend(facecolor="#1a2332", edgecolor="#3d5a80", labelcolor="#e2e8f0")
    plt.tight_layout()
    return fig


def plot_convergence_american(steps, bin_prices, tri_prices, opt_type_label):
    fig, ax = dark_fig((9, 4))
    ax.plot(steps, bin_prices, color="#63b3ed", linewidth=2, marker="o", markersize=5, label="Binomial")
    ax.plot(steps, tri_prices, color="#68d391", linewidth=2, marker="s", markersize=5, label="Trinomial")
    ax.set_xlabel("Number of Steps (N)")
    ax.set_ylabel("Option Price")
    ax.set_title(f"American {opt_type_label} — Binomial vs Trinomial Convergence")
    ax.legend(facecolor="#1a2332", edgecolor="#3d5a80", labelcolor="#e2e8f0")
    plt.tight_layout()
    return fig


def plot_payoff_diagram(S_in, K, price, opt_type_label):
    S_range = np.linspace(max(1, S_in * 0.4), S_in * 1.8, 300)
    payoff  = np.maximum(S_range - K, 0) if opt_type_label == "Call" else np.maximum(K - S_range, 0)
    profit  = payoff - price
    fig, ax = dark_fig((9, 4))
    ax.plot(S_range, payoff, color="#63b3ed", linewidth=2, label="Payoff at Expiry")
    ax.plot(S_range, profit, color="#68d391", linewidth=2, linestyle="--", label="Profit / Loss")
    ax.axhline(0, color="#4a5568", linewidth=1)
    ax.axvline(K,    color="#f6ad55", linewidth=1, linestyle=":", label=f"Strike K = {K}")
    ax.axvline(S_in, color="#fc8181", linewidth=1, linestyle=":", label=f"Current S = {S_in}")
    ax.fill_between(S_range, profit, 0, where=(profit > 0), alpha=0.15, color="#68d391")
    ax.fill_between(S_range, profit, 0, where=(profit < 0), alpha=0.15, color="#fc8181")
    ax.set_xlabel("Stock Price at Expiry")
    ax.set_ylabel("Value ($)")
    ax.set_title(f"{opt_type_label} Option — Payoff & Profit Diagram")
    ax.legend(facecolor="#1a2332", edgecolor="#3d5a80", labelcolor="#e2e8f0")
    plt.tight_layout()
    return fig


def plot_greeks_sensitivity(S_in, K, T, r, sigma, opt_type, deriv_type, method, N):
    # Fewer points for American (numerical Greeks are slower)
    n_pts = 40 if deriv_type == "American" else 200
    S_range = np.linspace(max(1, S_in * 0.4), S_in * 1.8, n_pts)
    metrics = {"Delta": [], "Gamma": [], "Vega": [], "Theta": []}

    for s in S_range:
        if deriv_type == "European":
            g = black_scholes_greeks(s, K, T, r, sigma, opt_type)
        else:
            g = american_numerical_greeks(s, K, T, r, sigma, N, opt_type, method)
        for key in metrics:
            metrics[key].append(g[key])

    colors = {"Delta": "#63b3ed", "Gamma": "#68d391", "Vega": "#f6ad55", "Theta": "#fc8181"}
    fig = plt.figure(figsize=(12, 7), facecolor="#0f1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    for idx, (greek, vals) in enumerate(metrics.items()):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor("#161b27")
        for spine in ax.spines.values():
            spine.set_color("#2d3748")
        ax.tick_params(colors="#a0aec0")
        ax.xaxis.label.set_color("#a0aec0")
        ax.yaxis.label.set_color("#a0aec0")
        ax.title.set_color("#90cdf4")
        ax.grid(True, color="#2d3748", linewidth=0.6, linestyle="--")
        ax.plot(S_range, vals, color=colors[greek], linewidth=2)
        ax.axvline(S_in, color="#718096", linewidth=1, linestyle=":")
        ax.set_xlabel("Stock Price (S)")
        ax.set_ylabel(greek)
        ax.set_title(greek)

    method_note = f"Numerical Finite Differences — {method}" if deriv_type == "American" else "Black-Scholes Closed Form"
    plt.suptitle(f"Greeks Sensitivity vs. Stock Price  [{method_note}]",
                 color="#90cdf4", fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def plot_mc_paths(sample_paths, T, K, opt_type_label):
    fig, ax = dark_fig((10, 5))
    time_axis = np.linspace(0, T, len(sample_paths[0]))
    for path in sample_paths[:30]:
        ax.plot(time_axis, path, alpha=0.25, linewidth=0.8, color="#63b3ed")
    avg_path = np.mean(sample_paths, axis=0)
    ax.plot(time_axis, avg_path, color="#f6ad55", linewidth=2.5, label="Average Path")
    ax.axhline(K, color="#fc8181", linewidth=1.5, linestyle="--", label=f"Strike K = {K}")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"Monte Carlo Simulated Paths — Asian {opt_type_label}")
    ax.legend(facecolor="#1a2332", edgecolor="#3d5a80", labelcolor="#e2e8f0")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown('<div class="section-header">Derivative Type</div>', unsafe_allow_html=True)
    deriv_type = st.selectbox(
        "Select derivative",
        ["European", "American", "Asian (Monte Carlo)"],
        help="Choose the type of option contract to price."
    )

    method = "Binomial"
    if deriv_type in ["European", "American"]:
        st.markdown('<div class="section-header">Pricing Method</div>', unsafe_allow_html=True)
        method = st.radio(
            "Tree method", ["Binomial", "Trinomial"],
            help="Binomial: 2 branches per node. Trinomial: 3 branches — often faster convergence."
        )

    st.markdown('<div class="section-header">Option Parameters</div>', unsafe_allow_html=True)
    opt_label = st.radio("Option type", ["Call", "Put"],
                         help="Call = right to BUY at strike. Put = right to SELL at strike.")
    opt_type = "C" if opt_label == "Call" else "P"

    S_in  = st.number_input("Spot Price (S₀)",           min_value=0.01,  max_value=100_000.0, value=100.0, step=1.0,
                            help="Current market price of the underlying asset. Must be > 0.")
    K     = st.number_input("Strike Price (K)",           min_value=0.01,  max_value=100_000.0, value=90.0,  step=1.0,
                            help="The price at which the option can be exercised. Must be > 0.")
    T     = st.number_input("Time to Maturity (years)",   min_value=0.01,  max_value=50.0,      value=10.0,  step=0.25,
                            help="Years until expiry (e.g. 0.5 = 6 months). Must be > 0.")
    r     = st.number_input("Risk-Free Rate (r)",         min_value=0.0,   max_value=1.0,       value=0.05,  step=0.005, format="%.4f",
                            help="Annual continuously-compounded rate as a decimal (0.05 = 5%). Must be ≥ 0.")
    sigma = st.number_input("Volatility (σ)",             min_value=0.001, max_value=5.0,       value=0.30,  step=0.01,  format="%.3f",
                            help="Annual volatility of the underlying as a decimal (0.30 = 30%). Must be > 0.")

    if deriv_type in ["European", "American"]:
        N = st.slider("Tree Steps (N)", min_value=5, max_value=500, value=100, step=5,
                      help="Number of time steps in the tree. Recommended: 100–300.")
    else:
        N = st.slider("Steps per Path (N)", min_value=10, max_value=500, value=100, step=10,
                      help="Time steps per simulated path.")
        M = st.slider("Simulations (M)", min_value=100, max_value=50_000, value=10_000, step=500,
                      help="Number of Monte Carlo paths. Recommended ≥ 5,000.")

    st.markdown('<div class="section-header">Charts to Display</div>', unsafe_allow_html=True)
    show_payoff = st.checkbox("Payoff & Profit Diagram", value=True)

    if deriv_type in ["European", "American"]:
        show_conv         = st.checkbox("Convergence Plot",          value=True)
        show_greeks_chart = st.checkbox("Greeks Sensitivity Charts", value=True)
    else:
        show_conv = show_greeks_chart = False

    show_paths = st.checkbox("Simulated Paths", value=True) if deriv_type == "Asian (Monte Carlo)" else False

    run = st.button("▶  Price Option", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# Main Area
# ─────────────────────────────────────────────
st.markdown("# 📈 Option Pricing Dashboard")
st.markdown("Binomial · Trinomial · Black–Scholes · Monte Carlo — with Greeks & sensitivity analysis.")
st.markdown("---")

with st.expander("📖 How to Use This Dashboard", expanded=False):
    st.markdown("""
### Quick Start
1. **Select a derivative type** in the sidebar (European, American, Asian).
2. **Choose a pricing method** — Binomial or Trinomial tree (applies to European and American).
3. **Enter the option parameters** — every input has a tooltip (hover **?**) explaining valid ranges.
4. **Click "▶ Price Option"** to run the model and display results.

### Parameter Guide
| Parameter | Symbol | Typical Range | Notes |
|-----------|--------|--------------|-------|
| Spot Price | S₀ | > 0 | Current underlying price |
| Strike Price | K | > 0 | Exercise price |
| Time to Maturity | T | 0.01 – 30 | In years (0.5 = 6 months) |
| Risk-Free Rate | r | 0 – 0.20 | Annual, decimal (0.05 = 5%) |
| Volatility | σ | 0.05 – 1.50 | Annual, decimal (0.30 = 30%) |
| Tree Steps | N | 50 – 500 | Higher N = more accurate, slower |
| Simulations | M | 1,000 – 50,000 | Higher M = lower Monte Carlo error |

### Derivative Types
- **European**: Exercisable only at expiry. Tree price is benchmarked against the Black–Scholes closed form.
- **American**: Exercisable any time before expiry. Both Binomial and Trinomial prices are displayed. Greeks are computed via finite differences since no closed form exists.
- **Asian (Monte Carlo)**: Payoff depends on the *average* stock price over the option's life, not just the final price.

### Greeks — American Options
Since no closed-form solution exists for American options, Greeks are estimated by **finite differences** —
bumping each input slightly and measuring how the price responds. This is standard industry practice.
""")

# ─────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────
if run:
    with st.spinner("Running model..."):
        price = bin_price = tri_price = bs_price = error = None
        greeks = {}

        try:
            if deriv_type == "European":
                if method == "Binomial":
                    price = binomial_eu_option(S_in, K, T, r, sigma, N, opt_type)
                else:
                    price = trinomial_eu_option(S_in, K, T, r, sigma, N, opt_type)
                bs_price = black_scholes(S_in, K, T, r, sigma, opt_type)
                greeks   = black_scholes_greeks(S_in, K, T, r, sigma, opt_type)

            elif deriv_type == "American":
                bin_price = binomial_american_options(S_in, K, T, r, sigma, N, opt_type)
                tri_price = trinomial_american_option(S_in, K, T, r, sigma, N, opt_type)
                price     = bin_price if method == "Binomial" else tri_price
                greeks    = american_numerical_greeks(S_in, K, T, r, sigma, N, opt_type, method)

            elif deriv_type == "Asian (Monte Carlo)":
                price, std_err, sample_paths = asian_option_mc(S_in, K, T, r, sigma, N, M, opt_type)
                error = std_err

        except Exception as e:
            st.error(f"Pricing error: {e}")
            st.stop()

    # ── Price Cards ──
    st.markdown("## Results")

    if deriv_type == "European":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="price-card"><h2>{method} Tree Price</h2><h1>${price:,.4f}</h1></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="price-card"><h2>Black–Scholes Price</h2><h1>${bs_price:,.4f}</h1></div>',
                        unsafe_allow_html=True)
        with c3:
            diff = abs(price - bs_price)
            st.markdown(f'<div class="price-card"><h2>Absolute Difference</h2><h1>${diff:,.4f}</h1></div>',
                        unsafe_allow_html=True)

    elif deriv_type == "American":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="price-card"><h2>Binomial Price</h2><h1>${bin_price:,.4f}</h1></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="price-card"><h2>Trinomial Price</h2><h1>${tri_price:,.4f}</h1></div>',
                        unsafe_allow_html=True)
        with c3:
            diff = abs(bin_price - tri_price)
            st.markdown(f'<div class="price-card"><h2>Difference (Bin vs Tri)</h2><h1>${diff:,.4f}</h1></div>',
                        unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
            <strong>Early Exercise:</strong> At each node the model compares the continuation value to the
            intrinsic value (S − K for calls, K − S for puts) and takes the maximum, capturing the early
            exercise premium. When r = 0, early exercise is rarely optimal for calls so American ≈ European.
            Greeks are estimated via <strong>finite differences</strong> on the selected tree method.
        </div>""", unsafe_allow_html=True)

    elif deriv_type == "Asian (Monte Carlo)":
        c1, c2, c3 = st.columns(3)
        ci_lo, ci_hi = price - 1.96 * error, price + 1.96 * error
        with c1:
            st.markdown(f'<div class="price-card"><h2>Asian {opt_label} Price</h2><h1>${price:,.4f}</h1></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="price-card"><h2>Std. Error</h2><h1>${error:,.4f}</h1></div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="price-card"><h2>95% Confidence Interval</h2>'
                        f'<h1 style="font-size:1.6rem;">[{ci_lo:.3f}, {ci_hi:.3f}]</h1></div>',
                        unsafe_allow_html=True)

    # ── Greeks ──
    if deriv_type in ["European", "American"] and greeks:
        st.markdown("---")
        label_note = " *(numerical — finite differences)*" if deriv_type == "American" else " *(Black-Scholes closed form)*"
        st.markdown(f"## 🔬 Option Greeks{label_note}")
        greek_meta = {
            "Delta": ("Δ", "Price sensitivity to S₀"),
            "Gamma": ("Γ", "Delta sensitivity to S₀"),
            "Vega":  ("ν", "Price sensitivity to σ"),
            "Theta": ("Θ", "Price decay over time"),
            "Rho":   ("ρ", "Sensitivity to r"),
        }
        for col, (name, (sym, desc)) in zip(st.columns(5), greek_meta.items()):
            with col:
                st.markdown(f"""<div class="greek-card">
                    <div class="greek-name">{name} ({sym})</div>
                    <div class="greek-value">{greeks[name]:.4f}</div>
                    <div style="color:#718096;font-size:0.75rem;margin-top:4px;">{desc}</div>
                </div>""", unsafe_allow_html=True)

    # ── Plots ──
    st.markdown("---")
    st.markdown("## 📊 Visualisations")

    if show_payoff:
        st.markdown("### Payoff & Profit Diagram")
        st.pyplot(plot_payoff_diagram(S_in, K, price, opt_label))
        plt.close()

    if show_conv and deriv_type == "European":
        st.markdown("### Convergence Plot")
        with st.spinner("Computing convergence..."):
            steps, prices_conv, bs_ref = get_convergence_european(S_in, K, T, r, sigma, opt_type, method)
        st.pyplot(plot_convergence_european(steps, prices_conv, bs_ref, method, opt_label))
        plt.close()

    if show_conv and deriv_type == "American":
        st.markdown("### Convergence Plot — Binomial vs Trinomial")
        with st.spinner("Computing convergence (this takes a moment)..."):
            steps, bin_conv, tri_conv = get_convergence_american(S_in, K, T, r, sigma, opt_type)
        st.pyplot(plot_convergence_american(steps, bin_conv, tri_conv, opt_label))
        plt.close()

    if show_greeks_chart and deriv_type in ["European", "American"]:
        note = " *(computed numerically across stock price range — may take a few seconds)*" if deriv_type == "American" else ""
        st.markdown(f"### Greeks Sensitivity vs. Stock Price{note}")
        with st.spinner("Computing Greeks sensitivity..."):
            fig = plot_greeks_sensitivity(S_in, K, T, r, sigma, opt_type, deriv_type, method, N)
        st.pyplot(fig)
        plt.close()

    if show_paths and deriv_type == "Asian (Monte Carlo)":
        st.markdown("### Simulated Monte Carlo Paths")
        st.pyplot(plot_mc_paths(sample_paths, T, K, opt_label))
        plt.close()

else:
    st.markdown("""
    <div style="text-align:center; padding:80px 40px;">
        <div style="font-size:4rem;">📈</div>
        <h2 style="color:#718096; margin-top:16px;">Configure your option in the sidebar</h2>
        <p style="color:#4a5568; max-width:500px; margin:0 auto; font-size:0.95rem;">
            Select a derivative type, enter your parameters, and click
            <strong style="color:#63b3ed;">▶ Price Option</strong> to run the model.
        </p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#4a5568;font-size:0.8rem;'>"
            "Option Pricing Dashboard · Binomial · Trinomial · Black–Scholes · Monte Carlo</p>",
            unsafe_allow_html=True)
