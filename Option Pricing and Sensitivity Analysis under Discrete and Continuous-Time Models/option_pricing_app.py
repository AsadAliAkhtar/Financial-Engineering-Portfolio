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
    /* ── Global base ── */
    html, body, [class*="css"] {
        color: #0f172a !important;
    }

    /* ── Main background ── */
    .stApp { background-color: #f8fafc; }
    .main .block-container { background-color: #f8fafc; padding-top: 2rem; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #e2e8f0;
    }
    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    /* Radio and checkbox option text */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #1e293b !important;
        font-weight: 400 !important;
    }
    /* Selectbox — fix visible text in dropdown button and popup */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    div[data-baseweb="select"] span {
        color: #0f172a !important;
    }
    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    div[data-baseweb="popover"] * {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #eff6ff !important;
    }
    /* Slider value label */
    [data-testid="stSidebar"] .stSlider p {
        color: #374151 !important;
    }
    /* Number input text */
    [data-testid="stSidebar"] input {
        color: #0f172a !important;
        background-color: #f8fafc !important;
    }

    /* ── Section header (sidebar) ── */
    .section-header {
        color: #1e40af !important;
        font-size: 1.0rem;
        font-weight: 700;
        margin: 18px 0 8px;
        padding-bottom: 5px;
        border-bottom: 2px solid #bfdbfe;
    }

    /* ── Main content text ── */
    .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
        color: #1e293b !important;
    }
    h1 { color: #0f172a !important; }
    h2 { color: #1e293b !important; }
    h3 { color: #334155 !important; }
    p  { color: #1e293b !important; }

    /* ── Expander — force white background and dark text ── */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary {
        color: #1e293b !important;
        font-weight: 600;
        background-color: #ffffff !important;
    }
    [data-testid="stExpander"] summary:hover {
        background-color: #f1f5f9 !important;
    }
    [data-testid="stExpander"] > div {
        background-color: #ffffff !important;
    }
    [data-testid="stExpander"] * {
        color: #1e293b !important;
        background-color: transparent !important;
    }
    /* Expander table */
    [data-testid="stExpander"] table {
        background-color: #ffffff !important;
    }
    [data-testid="stExpander"] th {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }
    [data-testid="stExpander"] td {
        color: #1e293b !important;
    }

    /* ── Result block ── */
    .result-block {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 14px;
    }
    .result-label {
        color: #475569 !important;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .result-value {
        color: #0f172a !important;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    /* ── Greek row ── */
    .greek-row {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .greek-left { color: #1e293b !important; font-weight: 600; font-size: 0.95rem; }
    .greek-desc { color: #64748b !important; font-size: 0.8rem; }
    .greek-val  { color: #1d4ed8 !important; font-size: 1.2rem; font-weight: 700; }

    /* ── Info box ── */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.87rem;
        color: #1e3a5f !important;
    }
    .info-box strong { color: #1d4ed8 !important; }

    /* ── Divider ── */
    hr { border-color: #cbd5e1 !important; }

    /* ── Spinner / caption / small text ── */
    .stCaption, small, .stCaption p {
        color: #64748b !important;
    }

    /* ── Checkbox label in main area ── */
    .stCheckbox label { color: #1e293b !important; }

    /* ── st.spinner text ── */
    .stSpinner p { color: #1e293b !important; }

    /* ── Placeholder text ── */
    ::placeholder { color: #94a3b8 !important; }
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
            C[j, i] = np.exp(-r * dt) * (p * C[j+1, i+1] + (1-p) * C[j+1, i])
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
            C[j, i] = np.exp(-r * dt) * (p * C[j+1, i+1] + (1-p) * C[j+1, i])
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
# Plot Helpers  (light theme)
# ─────────────────────────────────────────────

def light_fig(figsize=(9, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    ax.tick_params(colors="#475569")
    ax.xaxis.label.set_color("#475569")
    ax.yaxis.label.set_color("#475569")
    ax.title.set_color("#0f172a")
    ax.grid(True, color="#e2e8f0", linewidth=0.7, linestyle="--")
    return fig, ax


def plot_convergence_european(steps, prices, bs_price, method, opt_type_label):
    fig, ax = light_fig((9, 4))
    ax.plot(steps, prices, color="#2563eb", linewidth=2, marker="o", markersize=5, label=f"{method} Tree")
    ax.axhline(bs_price, color="#dc2626", linestyle="--", linewidth=1.5, label="Black-Scholes")
    ax.set_xlabel("Number of Steps (N)")
    ax.set_ylabel("Option Price")
    ax.set_title(f"European {opt_type_label} — {method} Convergence vs Black-Scholes")
    ax.legend(facecolor="#ffffff", edgecolor="#e2e8f0")
    plt.tight_layout()
    return fig


def plot_convergence_american(steps, bin_prices, tri_prices, opt_type_label):
    fig, ax = light_fig((9, 4))
    ax.plot(steps, bin_prices, color="#2563eb", linewidth=2, marker="o", markersize=5, label="Binomial")
    ax.plot(steps, tri_prices, color="#16a34a", linewidth=2, marker="s", markersize=5, label="Trinomial")
    ax.set_xlabel("Number of Steps (N)")
    ax.set_ylabel("Option Price")
    ax.set_title(f"American {opt_type_label} — Binomial vs Trinomial Convergence")
    ax.legend(facecolor="#ffffff", edgecolor="#e2e8f0")
    plt.tight_layout()
    return fig


def plot_payoff_diagram(S_in, K, price, opt_type_label):
    S_range = np.linspace(max(1, S_in * 0.4), S_in * 1.8, 300)
    payoff  = np.maximum(S_range - K, 0) if opt_type_label == "Call" else np.maximum(K - S_range, 0)
    profit  = payoff - price
    fig, ax = light_fig((9, 4))
    ax.plot(S_range, payoff, color="#2563eb", linewidth=2, label="Payoff at Expiry")
    ax.plot(S_range, profit, color="#16a34a", linewidth=2, linestyle="--", label="Profit / Loss")
    ax.axhline(0, color="#94a3b8", linewidth=1)
    ax.axvline(K,    color="#f59e0b", linewidth=1.2, linestyle=":", label=f"Strike K = {K}")
    ax.axvline(S_in, color="#dc2626", linewidth=1.2, linestyle=":", label=f"Current S = {S_in}")
    ax.fill_between(S_range, profit, 0, where=(profit > 0), alpha=0.10, color="#16a34a")
    ax.fill_between(S_range, profit, 0, where=(profit < 0), alpha=0.10, color="#dc2626")
    ax.set_xlabel("Stock Price at Expiry")
    ax.set_ylabel("Value ($)")
    ax.set_title(f"{opt_type_label} Option — Payoff & Profit Diagram")
    ax.legend(facecolor="#ffffff", edgecolor="#e2e8f0")
    plt.tight_layout()
    return fig


def plot_mc_paths(sample_paths, T, K, opt_type_label):
    fig, ax = light_fig((10, 5))
    time_axis = np.linspace(0, T, len(sample_paths[0]))
    for path in sample_paths[:30]:
        ax.plot(time_axis, path, alpha=0.25, linewidth=0.8, color="#2563eb")
    avg_path = np.mean(sample_paths, axis=0)
    ax.plot(time_axis, avg_path, color="#f59e0b", linewidth=2.5, label="Average Path")
    ax.axhline(K, color="#dc2626", linewidth=1.5, linestyle="--", label=f"Strike K = {K}")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"Monte Carlo Simulated Paths — Asian {opt_type_label}")
    ax.legend(facecolor="#ffffff", edgecolor="#e2e8f0")
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
        ["European", "American", "Asian (Monte Carlo)", "Black-Scholes"],
        help="European / American use tree models. Asian uses Monte Carlo. Black-Scholes gives the analytical closed-form price and Greeks."
    )

    # Binomial / Trinomial only for European and American
    method = None
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

    S_in  = st.number_input("Spot Price (S₀)",         min_value=0.01,  max_value=100_000.0, value=100.0, step=1.0,
                            help="Current market price of the underlying asset. Must be > 0.")
    K     = st.number_input("Strike Price (K)",         min_value=0.01,  max_value=100_000.0, value=90.0,  step=1.0,
                            help="The price at which the option can be exercised. Must be > 0.")
    T     = st.number_input("Time to Maturity (years)", min_value=0.01,  max_value=50.0,      value=10.0,  step=0.25,
                            help="Years until expiry (e.g. 0.5 = 6 months). Must be > 0.")
    r     = st.number_input("Risk-Free Rate (r)",       min_value=0.0,   max_value=1.0,       value=0.05,  step=0.005, format="%.4f",
                            help="Annual continuously-compounded rate as decimal (0.05 = 5%). Must be ≥ 0.")
    sigma = st.number_input("Volatility (σ)",           min_value=0.001, max_value=5.0,       value=0.30,  step=0.01,  format="%.3f",
                            help="Annual volatility as decimal (0.30 = 30%). Must be > 0.")

    if deriv_type in ["European", "American"]:
        N = st.slider("Tree Steps (N)", min_value=5, max_value=500, value=100, step=5,
                      help="Number of time steps in the tree. Recommended: 100–300.")
    elif deriv_type == "Asian (Monte Carlo)":
        N = st.slider("Steps per Path (N)", min_value=10, max_value=500, value=100, step=10,
                      help="Time steps along each simulated price path.")
        M = st.slider("Simulations (M)", min_value=100, max_value=50_000, value=10_000, step=500,
                      help="Number of Monte Carlo paths. Recommended ≥ 5,000.")

    st.markdown('<div class="section-header">Charts to Display</div>', unsafe_allow_html=True)
    show_payoff = st.checkbox("Payoff & Profit Diagram", value=True)

    if deriv_type in ["European", "American"]:
        show_conv = st.checkbox("Convergence Plot", value=True)
    else:
        show_conv = False

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
1. **Select a derivative type** in the sidebar.
2. **Choose Binomial or Trinomial** — only shown for European and American options.
3. **Enter parameters** — every input has a tooltip (hover **?**) explaining valid ranges.
4. **Click "▶ Price Option"** to run the model.

### Derivative Types
- **European**: Exercisable only at expiry. Priced via tree model, benchmarked against Black–Scholes.
- **American**: Exercisable any time before expiry. Both Binomial and Trinomial prices shown.
- **Asian (Monte Carlo)**: Payoff depends on the *average* stock price over the option's life.
- **Black-Scholes**: Analytical closed-form model for European options. Also shows all five Greeks.

### Parameter Guide
| Parameter | Symbol | Typical Range | Notes |
|-----------|--------|--------------|-------|
| Spot Price | S₀ | > 0 | Current underlying price |
| Strike Price | K | > 0 | Exercise price |
| Time to Maturity | T | 0.01 – 30 | In years (0.5 = 6 months) |
| Risk-Free Rate | r | 0 – 0.20 | Annual, decimal (0.05 = 5%) |
| Volatility | σ | 0.05 – 1.50 | Annual, decimal (0.30 = 30%) |
| Tree Steps | N | 50 – 500 | Higher N = more accurate, slower |
| Simulations | M | 1,000 – 50,000 | Higher M = lower MC error |
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

            elif deriv_type == "American":
                bin_price = binomial_american_options(S_in, K, T, r, sigma, N, opt_type)
                tri_price = trinomial_american_option(S_in, K, T, r, sigma, N, opt_type)
                price     = bin_price if method == "Binomial" else tri_price

            elif deriv_type == "Asian (Monte Carlo)":
                price, std_err, sample_paths = asian_option_mc(S_in, K, T, r, sigma, N, M, opt_type)
                error = std_err

            elif deriv_type == "Black-Scholes":
                price    = black_scholes(S_in, K, T, r, sigma, opt_type)
                greeks   = black_scholes_greeks(S_in, K, T, r, sigma, opt_type)

        except Exception as e:
            st.error(f"Pricing error: {e}")
            st.stop()

    # ── Results ──────────────────────────────────
    st.markdown("## Results")

    if deriv_type == "European":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="result-block">
                <div class="result-label">{method} Tree — {opt_label}</div>
                <div class="result-value">${price:,.4f}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="result-block">
                <div class="result-label">Black–Scholes Benchmark</div>
                <div class="result-value">${bs_price:,.4f}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            diff = abs(price - bs_price)
            st.markdown(f"""<div class="result-block">
                <div class="result-label">Absolute Difference</div>
                <div class="result-value">${diff:,.4f}</div>
            </div>""", unsafe_allow_html=True)

    elif deriv_type == "American":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="result-block">
                <div class="result-label">Binomial — {opt_label}</div>
                <div class="result-value">${bin_price:,.4f}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="result-block">
                <div class="result-label">Trinomial — {opt_label}</div>
                <div class="result-value">${tri_price:,.4f}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            diff = abs(bin_price - tri_price)
            st.markdown(f"""<div class="result-block">
                <div class="result-label">Difference (Bin vs Tri)</div>
                <div class="result-value">${diff:,.4f}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
            <strong>Early Exercise:</strong> At each node the model takes the maximum of the continuation value
            and intrinsic value (S − K for calls, K − S for puts), capturing the early exercise premium.
            When r = 0, early exercise is rarely optimal for calls so American ≈ European.
        </div>""", unsafe_allow_html=True)

    elif deriv_type == "Asian (Monte Carlo)":
        ci_lo, ci_hi = price - 1.96 * error, price + 1.96 * error
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="result-block">
                <div class="result-label">Asian {opt_label} Price</div>
                <div class="result-value">${price:,.4f}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="result-block">
                <div class="result-label">Standard Error</div>
                <div class="result-value">${error:,.4f}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="result-block">
                <div class="result-label">95% Confidence Interval</div>
                <div class="result-value" style="font-size:1.4rem;">[{ci_lo:.3f}, {ci_hi:.3f}]</div>
            </div>""", unsafe_allow_html=True)

    elif deriv_type == "Black-Scholes":
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""<div class="result-block">
                <div class="result-label">Black–Scholes {opt_label} Price</div>
                <div class="result-value">${price:,.4f}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="info-box">
                <strong>Black–Scholes</strong> assumes the stock follows Geometric Brownian Motion with
                constant volatility and risk-free rate. It gives an exact analytical price for
                <em>European</em> options. Use this as the theoretical benchmark — real markets
                deviate due to volatility smiles, jumps, and discrete dividends.
            </div>""", unsafe_allow_html=True)

    # ── Greeks (Black-Scholes only) ──────────────
    if deriv_type == "Black-Scholes" and greeks:
        st.markdown("---")
        st.markdown("## 🔬 Option Greeks")
        st.caption("Analytical sensitivities derived from the Black-Scholes closed-form solution.")

        greek_meta = {
            "Delta": ("Δ", "Price sensitivity to stock price (S₀)",        "∂V/∂S"),
            "Gamma": ("Γ", "Rate of change of Delta w.r.t. stock price",    "∂²V/∂S²"),
            "Vega":  ("ν", "Price sensitivity to volatility (σ)",           "∂V/∂σ"),
            "Theta": ("Θ", "Price decay per unit time",                     "∂V/∂t"),
            "Rho":   ("ρ", "Price sensitivity to risk-free rate (r)",       "∂V/∂r"),
        }

        for name, (sym, desc, formula) in greek_meta.items():
            val = greeks[name]
            st.markdown(f"""
            <div class="greek-row">
                <div class="greek-left">
                    {name} &nbsp;<span style="color:#94a3b8;">({sym})</span>
                    &nbsp;&nbsp;<span class="greek-desc">{desc}</span>
                </div>
                <div style="text-align:right;">
                    <span style="color:#94a3b8; font-size:0.8rem; margin-right:16px;">{formula}</span>
                    <span class="greek-val">{val:.4f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Plots ────────────────────────────────────
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
        with st.spinner("Computing convergence (this may take a moment)..."):
            steps, bin_conv, tri_conv = get_convergence_american(S_in, K, T, r, sigma, opt_type)
        st.pyplot(plot_convergence_american(steps, bin_conv, tri_conv, opt_label))
        plt.close()

    if show_paths and deriv_type == "Asian (Monte Carlo)":
        st.markdown("### Simulated Monte Carlo Paths")
        st.pyplot(plot_mc_paths(sample_paths, T, K, opt_label))
        plt.close()

else:
    st.markdown("""
    <div style="text-align:center; padding:80px 40px;">
        <div style="font-size:4rem;">📈</div>
        <h2 style="color:#64748b; margin-top:16px;">Configure your option in the sidebar</h2>
        <p style="color:#94a3b8; max-width:480px; margin:0 auto; font-size:0.95rem;">
            Select a derivative type, enter your parameters, and click
            <strong style="color:#2563eb;">▶ Price Option</strong> to run the model.
        </p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#94a3b8; font-size:0.8rem;'>"
            "Option Pricing Dashboard · Binomial · Trinomial · Black–Scholes · Monte Carlo</p>",
            unsafe_allow_html=True)
