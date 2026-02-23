import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Option Pricing Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #2d3748; }

    /* Price card */
    .price-card {
        background: linear-gradient(135deg, #1a2332, #243447);
        border: 1px solid #3d5a80;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 16px;
    }
    .price-card h2 { color: #63b3ed; font-size: 1rem; margin: 0 0 8px; letter-spacing: 0.08em; text-transform: uppercase; }
    .price-card h1 { color: #f0f4f8; font-size: 2.8rem; margin: 0; font-weight: 700; }

    /* Greek card */
    .greek-card {
        background: #161b27;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .greek-card .greek-name { color: #a0aec0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; }
    .greek-card .greek-value { color: #68d391; font-size: 1.5rem; font-weight: 700; }

    /* Info box */
    .info-box {
        background: #1a2332;
        border-left: 4px solid #3d5a80;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #a0aec0;
    }
    .info-box strong { color: #90cdf4; }

    /* Coming soon */
    .coming-soon {
        background: #1a2332;
        border: 2px dashed #4a5568;
        border-radius: 12px;
        padding: 60px;
        text-align: center;
        color: #718096;
    }
    .coming-soon h2 { color: #a0aec0; }

    /* Section header */
    .section-header {
        color: #90cdf4;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 20px 0 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #2d3748;
    }

    /* Metric label override */
    [data-testid="metric-container"] { background: #161b27; border-radius: 8px; border: 1px solid #2d3748; padding: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Pricing functions (from the original project)
# ─────────────────────────────────────────────

def binomial_eu_option(S_in, K, T, r, sigma, N, opt_type):
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

    S_final = asset_price(N)
    payoff = np.maximum(S_final - K, 0) if opt_type == "C" else np.maximum(K - S_final, 0)
    nxt_vec_prices = payoff
    for i in range(1, N + 1):
        S_current = asset_price(N - i)
        expectation = np.zeros(len(S_current))
        for j in range(len(S_current)):
            expectation[j] = (pd * nxt_vec_prices[j] +
                              pm * nxt_vec_prices[j + 1] +
                              pu * nxt_vec_prices[j + 2])
        nxt_vec_prices = np.exp(-r * dt) * expectation
    return nxt_vec_prices[0]


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
    if opt_type == "C":
        delta = norm.cdf(d1)
        theta = (-((S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) -
                 r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = (-((S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) +
                 r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}


def asian_option_mc(S_in, K, T, r, sigma, N, M, opt_type, seed=42):
    np.random.seed(seed)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    payoffs = []
    sample_paths = []
    for j in range(M):
        path = [S_in]
        S_cur = S_in
        for _ in range(N):
            move = np.random.binomial(1, p)
            S_cur = S_cur * u if move == 1 else S_cur * d
            path.append(S_cur)
        avg_price = np.mean(path)
        payoff = np.exp(-r * T) * max(avg_price - K, 0) if opt_type == "C" else np.exp(-r * T) * max(K - avg_price, 0)
        payoffs.append(payoff)
        if j < 50:
            sample_paths.append(path)
    return np.mean(payoffs), np.std(payoffs) / np.sqrt(M), sample_paths


# ─────────────────────────────────────────────
# Convergence data helper
# ─────────────────────────────────────────────
def get_convergence_data(S, K, T, r, sigma, opt_type, method):
    steps = [2, 5, 10, 25, 50, 100, 200, 300, 500, 750, 1000]
    prices = []
    for n in steps:
        try:
            if method == "Binomial":
                price = binomial_eu_option(S, K, T, r, sigma, n, opt_type)
            else:
                price = trinomial_eu_option(S, K, T, r, sigma, n, opt_type)
        except Exception:
            price = np.nan
        prices.append(price)
    bs_price = black_scholes(S, K, T, r, sigma, opt_type)
    return steps, prices, bs_price


# ─────────────────────────────────────────────
# Plot helpers
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


def plot_convergence(steps, prices, bs_price, method, opt_type_label):
    fig, ax = dark_fig((9, 4))
    ax.plot(steps, prices, color="#63b3ed", linewidth=2, marker="o", markersize=5, label=f"{method} Tree")
    ax.axhline(bs_price, color="#f6ad55", linestyle="--", linewidth=1.5, label="Black-Scholes")
    ax.set_xlabel("Number of Steps (N)")
    ax.set_ylabel("Option Price")
    ax.set_title(f"{method} Tree Convergence — {opt_type_label}")
    ax.legend(facecolor="#1a2332", edgecolor="#3d5a80", labelcolor="#e2e8f0")
    plt.tight_layout()
    return fig


def plot_payoff_diagram(S_in, K, price, opt_type_label):
    S_range = np.linspace(max(1, S_in * 0.4), S_in * 1.8, 300)
    if opt_type_label == "Call":
        payoff = np.maximum(S_range - K, 0)
        profit = payoff - price
    else:
        payoff = np.maximum(K - S_range, 0)
        profit = payoff - price

    fig, ax = dark_fig((9, 4))
    ax.plot(S_range, payoff, color="#63b3ed", linewidth=2, label="Payoff at Expiry")
    ax.plot(S_range, profit, color="#68d391", linewidth=2, linestyle="--", label="Profit / Loss")
    ax.axhline(0, color="#4a5568", linewidth=1)
    ax.axvline(K, color="#f6ad55", linewidth=1, linestyle=":", label=f"Strike K = {K}")
    ax.axvline(S_in, color="#fc8181", linewidth=1, linestyle=":", label=f"Current S = {S_in}")
    ax.fill_between(S_range, profit, 0, where=(profit > 0), alpha=0.15, color="#68d391")
    ax.fill_between(S_range, profit, 0, where=(profit < 0), alpha=0.15, color="#fc8181")
    ax.set_xlabel("Stock Price at Expiry")
    ax.set_ylabel("Value ($)")
    ax.set_title(f"{opt_type_label} Option — Payoff & Profit Diagram")
    ax.legend(facecolor="#1a2332", edgecolor="#3d5a80", labelcolor="#e2e8f0")
    plt.tight_layout()
    return fig


def plot_greeks_sensitivity(S_in, K, T, r, sigma, opt_type):
    S_range = np.linspace(max(1, S_in * 0.4), S_in * 1.8, 200)
    metrics = {"Delta": [], "Gamma": [], "Vega": [], "Theta": []}
    for s in S_range:
        g = black_scholes_greeks(s, K, T, r, sigma, opt_type)
        for key in metrics:
            metrics[key].append(g[key])

    colors = {"Delta": "#63b3ed", "Gamma": "#68d391", "Vega": "#f6ad55", "Theta": "#fc8181"}
    fig = plt.figure(figsize=(12, 7), facecolor="#0f1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
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
    plt.suptitle("Greeks Sensitivity vs. Stock Price", color="#90cdf4", fontsize=13, y=1.01)
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
# Sidebar — Input Controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # ── Derivative type ──
    st.markdown('<div class="section-header">Derivative Type</div>', unsafe_allow_html=True)
    deriv_type = st.selectbox(
        "Select derivative",
        ["European", "American", "Asian (Monte Carlo)"],
        help="Choose the type of option contract to price."
    )

    # ── Method selection (only for European) ──
    method = "Binomial"
    if deriv_type == "European":
        st.markdown('<div class="section-header">Pricing Method</div>', unsafe_allow_html=True)
        method = st.radio(
            "Tree method",
            ["Binomial", "Trinomial"],
            help="Binomial: 2 branches per node. Trinomial: 3 branches — faster convergence."
        )

    # ── Option type ──
    st.markdown('<div class="section-header">Option Parameters</div>', unsafe_allow_html=True)
    opt_label = st.radio("Option type", ["Call", "Put"],
                         help="Call = right to BUY. Put = right to SELL.")
    opt_type = "C" if opt_label == "Call" else "P"

    # ── Core parameters ──
    S_in = st.number_input(
        "Spot Price (S₀)", min_value=0.01, max_value=100_000.0, value=100.0, step=1.0,
        help="Current market price of the underlying asset. Must be > 0."
    )
    K = st.number_input(
        "Strike Price (K)", min_value=0.01, max_value=100_000.0, value=90.0, step=1.0,
        help="The price at which the option can be exercised. Must be > 0."
    )
    T = st.number_input(
        "Time to Maturity (T, years)", min_value=0.01, max_value=50.0, value=10.0, step=0.25,
        help="Time until the option expires, in years (e.g. 0.5 = 6 months). Must be > 0."
    )
    r = st.number_input(
        "Risk-Free Rate (r)", min_value=0.0, max_value=1.0, value=0.00, step=0.005,
        format="%.4f",
        help="Annual continuously-compounded risk-free rate as a decimal (e.g. 0.05 = 5%). Must be ≥ 0."
    )
    sigma = st.number_input(
        "Volatility (σ)", min_value=0.001, max_value=5.0, value=0.30, step=0.01,
        format="%.3f",
        help="Annual volatility of the underlying asset as a decimal (e.g. 0.3 = 30%). Must be > 0."
    )

    # ── Tree steps / MC sims ──
    if deriv_type in ["European", "American"]:
        N = st.slider(
            "Tree Steps (N)", min_value=5, max_value=500, value=100, step=5,
            help="Number of time steps in the tree. More steps → more accurate but slower. Recommended: 100–300."
        )
    elif deriv_type == "Asian (Monte Carlo)":
        N = st.slider(
            "Steps per Path (N)", min_value=10, max_value=500, value=100, step=10,
            help="Number of time steps along each simulated price path."
        )
        M = st.slider(
            "Simulations (M)", min_value=100, max_value=50_000, value=10_000, step=500,
            help="Number of Monte Carlo paths. More paths → lower standard error. Recommended ≥ 5,000."
        )

    # ── Visuals ──
    st.markdown('<div class="section-header">Charts to Display</div>', unsafe_allow_html=True)
    show_payoff = st.checkbox("Payoff & Profit Diagram", value=True)
    show_conv = st.checkbox("Convergence Plot", value=True,
                            disabled=deriv_type not in ["European"]) if deriv_type == "European" else False
    show_greeks_chart = st.checkbox("Greeks Sensitivity Charts", value=True,
                                    disabled=deriv_type != "European") if deriv_type == "European" else False
    show_paths = st.checkbox("Simulated Paths", value=True,
                             disabled=deriv_type != "Asian (Monte Carlo)") if deriv_type == "Asian (Monte Carlo)" else False

    run = st.button("▶  Price Option", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# Main Area
# ─────────────────────────────────────────────
st.markdown("# 📈 Option Pricing Dashboard")
st.markdown("Implementations of Binomial, Trinomial, Black–Scholes and Monte Carlo methods with sensitivity analysis.")
st.markdown("---")

# Instructions panel
with st.expander("📖 How to Use This Dashboard", expanded=False):
    st.markdown("""
    ### Quick Start
    1. **Select a derivative type** in the sidebar (European, American, Asian).
    2. **Choose a pricing method** (Binomial/Trinomial for European).
    3. **Enter the option parameters** — all inputs have tooltips explaining valid ranges.
    4. **Click "Price Option"** to run the model.

    ### Parameter Guide
    | Parameter | Symbol | Typical Range | Notes |
    |-----------|--------|--------------|-------|
    | Spot Price | S₀ | > 0 | Current underlying price |
    | Strike Price | K | > 0 | Exercise price |
    | Time to Maturity | T | 0.01 – 30 | In years (1 = 1 year) |
    | Risk-Free Rate | r | 0 – 0.20 | Annual, decimal (e.g. 0.05) |
    | Volatility | σ | 0.05 – 1.50 | Annual, decimal (e.g. 0.30) |
    | Tree Steps | N | 50 – 500 | Higher N = more accurate |
    | Simulations | M | 1,000 – 50,000 | Higher M = lower MC error |

    ### Derivative Types
    - **European**: Can only be exercised at expiry. Priced via Binomial or Trinomial tree, compared with Black–Scholes closed form.
    - **American**: Can be exercised any time before expiry — priced via Binomial tree with early-exercise check at every node.
    - **Asian (Monte Carlo)**: Payoff depends on the *average* stock price over the life of the option, not just the final price.
    """)

# ─────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────
if run:
    with st.spinner("Running model..."):

        # ── Compute price ──
        price = None
        bs_price = None
        error = None

        try:
            if deriv_type == "European":
                if method == "Binomial":
                    price = binomial_eu_option(S_in, K, T, r, sigma, N, opt_type)
                else:
                    price = trinomial_eu_option(S_in, K, T, r, sigma, N, opt_type)
                bs_price = black_scholes(S_in, K, T, r, sigma, opt_type)
                greeks = black_scholes_greeks(S_in, K, T, r, sigma, opt_type)

            elif deriv_type == "American":
                price = binomial_american_options(S_in, K, T, r, sigma, N, opt_type)

            elif deriv_type == "Asian (Monte Carlo)":
                price, std_err, sample_paths = asian_option_mc(S_in, K, T, r, sigma, N, M, opt_type)
                error = std_err

        except Exception as e:
            st.error(f"Pricing error: {e}")
            st.stop()

        # ── Price cards ──
        st.markdown("## Results")

        if deriv_type == "European":
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="price-card">
                    <h2>{method} Tree Price</h2>
                    <h1>${price:,.4f}</h1>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="price-card">
                    <h2>Black–Scholes Price</h2>
                    <h1>${bs_price:,.4f}</h1>
                </div>""", unsafe_allow_html=True)
            with c3:
                diff = abs(price - bs_price)
                st.markdown(f"""
                <div class="price-card">
                    <h2>Absolute Difference</h2>
                    <h1>${diff:,.4f}</h1>
                </div>""", unsafe_allow_html=True)

        elif deriv_type == "American":
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"""
                <div class="price-card">
                    <h2>American {opt_label} Price</h2>
                    <h1>${price:,.4f}</h1>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div class="info-box">
                    <strong>Early Exercise:</strong> The American option value accounts for the possibility of
                    exercising before expiry. At each node, the model takes the maximum of the continuation value
                    and the intrinsic value (S − K for calls, K − S for puts).
                    When r = 0, early exercise is typically not optimal for calls, so American ≈ European.
                </div>""", unsafe_allow_html=True)

        elif deriv_type == "Asian (Monte Carlo)":
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="price-card">
                    <h2>Asian {opt_label} Price</h2>
                    <h1>${price:,.4f}</h1>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="price-card">
                    <h2>Std. Error</h2>
                    <h1>${error:,.4f}</h1>
                </div>""", unsafe_allow_html=True)
            with c3:
                ci_lo = price - 1.96 * error
                ci_hi = price + 1.96 * error
                st.markdown(f"""
                <div class="price-card">
                    <h2>95% CI</h2>
                    <h1style="font-size:1.4rem;">[{ci_lo:.3f}, {ci_hi:.3f}]</h1>
                </div>""", unsafe_allow_html=True)

        # ── Greeks ──
        if deriv_type == "European":
            st.markdown("---")
            st.markdown("## 🔬 Option Greeks")
            greek_cols = st.columns(5)
            greek_meta = {
                "Delta": ("Δ", "Price sensitivity to S₀"),
                "Gamma": ("Γ", "Delta sensitivity to S₀"),
                "Vega":  ("ν", "Price sensitivity to σ"),
                "Theta": ("Θ", "Price decay per day"),
                "Rho":   ("ρ", "Sensitivity to r"),
            }
            for col, (name, (sym, desc)) in zip(greek_cols, greek_meta.items()):
                with col:
                    val = greeks[name]
                    st.markdown(f"""
                    <div class="greek-card">
                        <div class="greek-name">{name} ({sym})</div>
                        <div class="greek-value">{val:.4f}</div>
                        <div style="color:#718096; font-size:0.75rem; margin-top:4px;">{desc}</div>
                    </div>""", unsafe_allow_html=True)

        # ── Plots ──
        st.markdown("---")
        st.markdown("## 📊 Visualisations")

        if show_payoff:
            st.markdown("### Payoff & Profit Diagram")
            fig = plot_payoff_diagram(S_in, K, price, opt_label)
            st.pyplot(fig)
            plt.close(fig)

        if deriv_type == "European" and show_conv:
            st.markdown("### Convergence Plot")
            with st.spinner("Computing convergence..."):
                steps, prices_conv, bs_ref = get_convergence_data(S_in, K, T, r, sigma, opt_type, method)
            fig = plot_convergence(steps, prices_conv, bs_ref, method, opt_label)
            st.pyplot(fig)
            plt.close(fig)

        if deriv_type == "European" and show_greeks_chart:
            st.markdown("### Greeks Sensitivity vs. Stock Price")
            fig = plot_greeks_sensitivity(S_in, K, T, r, sigma, opt_type)
            st.pyplot(fig)
            plt.close(fig)

        if deriv_type == "Asian (Monte Carlo)" and show_paths:
            st.markdown("### Simulated Monte Carlo Paths")
            fig = plot_mc_paths(sample_paths, T, K, opt_label)
            st.pyplot(fig)
            plt.close(fig)

else:
    # Placeholder state
    st.markdown("""
    <div style="text-align:center; padding: 80px 40px; color: #4a5568;">
        <div style="font-size: 4rem;">📈</div>
        <h2 style="color: #718096; margin-top: 16px;">Configure your option in the sidebar</h2>
        <p style="color: #4a5568; max-width: 500px; margin: 0 auto; font-size: 0.95rem;">
            Select a derivative type, enter your parameters, and click <strong style="color:#63b3ed;">▶ Price Option</strong> to run the model.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4a5568; font-size:0.8rem;'>"
    "Option Pricing Dashboard · Binomial · Trinomial · Black–Scholes · Monte Carlo"
    "</p>",
    unsafe_allow_html=True,
)
