"""
Mean-Variance Optimization — Streamlit App
==========================================
Run this app with:
    streamlit run mvo_streamlit_app.py

Required libraries:
    pip install streamlit yfinance pypfopt plotly
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
from pypfopt import expected_returns, risk_models, EfficientFrontier

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration — must be the FIRST streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mean-Variance Optimizer",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# A representative list of large S&P 500 stocks across different sectors
# ─────────────────────────────────────────────────────────────────────────────
SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM",  "JNJ",  "PG",   "XOM",  "BAC",  "HD",   "CVX",  "LLY",
    "AVGO", "MRK",  "ABBV", "KO",   "PEP",  "COST", "MCD",  "TMO",
    "ACN",  "V",    "MA",   "UNH",  "AMD",  "CRM",  "WMT",  "GE",
    "DIS",  "NFLX", "INTC", "PYPL", "T",    "VZ",   "PFE",  "IBM",
]

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for a clean, dark, professional look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Overall background */
    .stApp { background-color: #0d1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #161b27;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 14px 18px;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #0b0f1a;
        border-right: 1px solid #21262d;
    }

    /* Heading colours */
    h1, h2, h3 { color: #f0f6fc !important; }

    /* Button */
    .stButton > button {
        background-color: #F59E0B;
        color: #0a0e1a;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        width: 100%;
        padding: 12px 0;
        font-size: 15px;
    }
    .stButton > button:hover {
        background-color: #d97706;
        color: #0a0e1a;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: download price data and compute mu + sigma
# We cache this so the app doesn't re-download on every interaction
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    df = yf.download(list(tickers), start=start, end=end, auto_adjust=True)["Close"]
    # If only one ticker is selected, yfinance returns a Series — fix that
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    df.dropna(how="all", inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run the optimization and return weights + performance stats
# ─────────────────────────────────────────────────────────────────────────────
def run_optimization(prices: pd.DataFrame, method: str, cap: float | None):
    """
    Runs either Minimum Variance or Maximum Sharpe optimization.

    Parameters
    ----------
    prices  : DataFrame of adjusted close prices
    method  : "mvp" or "max_sharpe"
    cap     : upper bound on each weight (e.g. 0.15 for 15%), or None

    Returns
    -------
    weights : pd.Series of cleaned portfolio weights
    ret     : expected annual return (float)
    vol     : annual volatility (float)
    sharpe  : Sharpe ratio assuming 4% risk-free rate (float)
    """
    # Arithmetic mean returns (same approach as your notebook's PyPortfolioOpt section)
    mu = expected_returns.mean_historical_return(prices, compounding=False)

    # Sample covariance matrix, annualised
    sigma = risk_models.sample_cov(prices)

    # Build the EfficientFrontier object with optional weight bounds
    weight_bounds = (0, cap) if cap else (0, 1)   # no short selling in either case
    ef = EfficientFrontier(mu, sigma, weight_bounds=weight_bounds)

    if method == "mvp":
        ef.min_volatility()
    else:
        ef.max_sharpe(risk_free_rate=0.04)         # 4% risk-free rate (US T-bill approx)

    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.04)
    weights = pd.Series(ef.clean_weights())

    return weights, ret, vol, sharpe


# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Portfolio Settings")
    st.markdown("---")

    # 1. Stock selection
    st.markdown("**Select Stocks** *(min 2)*")
    selected_tickers = st.multiselect(
        label="Choose from S&P 500",
        options=SP500_TICKERS,
        default=["AAPL", "NVDA", "JPM", "AMZN", "GOOGL", "PG", "XOM", "JNJ"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # 2. Date range
    st.markdown("**Date Range**")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=pd.to_datetime("2016-01-01"))
    with col2:
        end_date = st.date_input("To", value=pd.to_datetime("2024-12-31"))

    st.markdown("---")

    # 3. Portfolio type
    st.markdown("**Portfolio Type**")
    method = st.radio(
        label="Portfolio Type",
        options=["Minimum Variance Portfolio", "Maximum Sharpe Ratio Portfolio"],
        label_visibility="collapsed",
    )
    method_key = "mvp" if "Minimum" in method else "max_sharpe"

    # Small explanation under the radio
    if method_key == "mvp":
        st.caption("🔵 Minimizes portfolio risk (volatility), regardless of return.")
    else:
        st.caption("🟡 Maximizes return per unit of risk (Sharpe Ratio). Uses 4% risk-free rate.")

    st.markdown("---")

    # 4. Cap constraint with a tooltip-style help text
    st.markdown(
        "**Max Weight per Stock (%)**  "
        '<span title="This limits how much of your portfolio can be placed in any single stock. '
        'For example, entering 15 means no stock will exceed 15% allocation. '
        'Leave blank for no limit.">❓</span>',
        unsafe_allow_html=True,
    )
    cap_input = st.number_input(
        label="Cap (%)",
        min_value=1,
        max_value=100,
        value=None,            # empty by default → no constraint
        step=1,
        placeholder="e.g. 15  (leave blank = no limit)",
        label_visibility="collapsed",
        help=(
            "This limits how much of your portfolio can be placed in any single stock. "
            "For example, entering 15 means no stock will exceed 15% allocation. "
            "Leave blank for no limit."
        ),
    )

    st.markdown("---")

    # 5. Run button
    run = st.button("▶ Run Optimization")


# ─────────────────────────────────────────────────────────────────────────────
# ── MAIN PANEL ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# Mean-Variance Optimization")
st.markdown(
    "Built on **Markowitz (1952)** modern portfolio theory. "
    "Select your stocks and settings in the sidebar, then click **Run Optimization**."
)
st.markdown("---")

if not run:
    # Landing state — show a friendly prompt
    st.info("👈 Configure your portfolio in the sidebar and click **Run Optimization** to get started.")
    st.stop()

# ── Input validation ─────────────────────────────────────────────────────────
if len(selected_tickers) < 2:
    st.error("Please select at least **2 stocks** to build a portfolio.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Convert cap from percentage to decimal (e.g. 15 → 0.15)
cap_decimal = (cap_input / 100) if cap_input else None

# Validate cap is not mathematically impossible
if cap_decimal and cap_decimal < 1 / len(selected_tickers):
    st.error(
        f"With **{len(selected_tickers)} stocks**, the cap cannot be lower than "
        f"**{100 / len(selected_tickers):.1f}%** (otherwise weights can't sum to 100%)."
    )
    st.stop()

# ── Download data ─────────────────────────────────────────────────────────────
with st.spinner("Downloading price data from Yahoo Finance..."):
    try:
        prices = get_data(
            tickers=tuple(selected_tickers),
            start=str(start_date),
            end=str(end_date),
        )
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

# Drop any tickers that returned all NaN (sometimes happens with delisted stocks)
prices.dropna(axis=1, how="all", inplace=True)
valid_tickers = list(prices.columns)

if len(valid_tickers) < 2:
    st.error("Not enough valid price data returned. Try different tickers or dates.")
    st.stop()

if len(valid_tickers) < len(selected_tickers):
    dropped = set(selected_tickers) - set(valid_tickers)
    st.warning(f"No data found for: {', '.join(dropped)}. Proceeding with remaining stocks.")

# ── Run optimization ──────────────────────────────────────────────────────────
with st.spinner("Running optimization..."):
    try:
        weights, ret, vol, sharpe = run_optimization(prices, method_key, cap_decimal)
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        st.stop()

# ── Display metrics ───────────────────────────────────────────────────────────
st.markdown(f"### Results — {method}")

col1, col2, col3 = st.columns(3)
col1.metric("📈 Expected Annual Return", f"{ret*100:.2f}%")
col2.metric("📉 Annual Volatility (Risk)", f"{vol*100:.2f}%")
col3.metric("⚡ Sharpe Ratio", f"{sharpe:.3f}")

st.markdown("---")

# ── Pie chart ─────────────────────────────────────────────────────────────────
# Filter out near-zero weights for a cleaner chart
weights_clean = weights[weights > 0.001].sort_values(ascending=False)
weights_df = weights_clean.reset_index()
weights_df.columns = ["Ticker", "Weight"]
weights_df["Weight (%)"] = (weights_df["Weight"] * 100).round(2)

col_chart, col_table = st.columns([3, 2])

with col_chart:
    st.markdown("#### Portfolio Allocation")
    fig = px.pie(
        weights_df,
        names="Ticker",
        values="Weight (%)",
        hole=0.38,                           # donut style — looks more modern
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        textfont_size=12,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.markdown("#### Weight Breakdown")
    # Format the table nicely for display
    display_df = weights_df[["Ticker", "Weight (%)"]].copy()
    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        column_config={
            "Weight (%)": st.column_config.ProgressColumn(
                "Weight (%)",
                min_value=0,
                max_value=100,
                format="%.2f%%",
            )
        }
    )

# ── Footer note ───────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️Returns are based on historical data and do not guarantee future performance. "
    "Data sourced from Yahoo Finance."
)