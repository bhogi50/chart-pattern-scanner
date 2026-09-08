import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from detector import detect_patterns

st.set_page_config(
    page_title="NSE Chart Pattern Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 NSE Chart Pattern Scanner")
st.caption("Personal technical-analysis scanner • heuristic pattern detection")

NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL",
    "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "INDUSINDBK", "INFY", "ITC", "JIOFIN", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND",
    "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE",
    "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATAMOTORS",
    "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT",
    "ULTRACEMCO",
]

with st.sidebar:
    st.header("Scanner")

    lookback = st.slider(
        "Lookback candles",
        min_value=60,
        max_value=250,
        value=120,
    )

    max_stocks = st.slider(
        "Stocks to scan",
        min_value=5,
        max_value=len(NIFTY_50),
        value=len(NIFTY_50),
    )

    scan_button = st.button(
        "🔎 Scan NIFTY 50",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.caption("Daily data • Yahoo Finance symbols use .NS")

@st.cache_data(ttl=900, show_spinner=False)
def load_data(symbol):
    import yfinance as yf

    data = yf.download(
        symbol,
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna()


if scan_button:
    rows = []
    symbols = NIFTY_50[:max_stocks]
    progress = st.progress(0)

    for index, symbol in enumerate(symbols, start=1):
        try:
            data = load_data(symbol + ".NS")
            patterns = detect_patterns(data, lookback)

            for pattern in patterns:
                rows.append({
                    "Stock": symbol,
                    **pattern,
                })
        except Exception:
            pass

        progress.progress(index / len(symbols))

    st.session_state["scan_results"] = pd.DataFrame(rows)

if "scan_results" not in st.session_state:
    st.info("Choose the lookback and press **Scan NIFTY 50**.")
    st.markdown(
        """
        ### Currently detected

        - Symmetrical Triangle
        - Ascending Triangle
        - Descending Triangle
        - Rising Wedge
        - Falling Wedge
        - Rising Channel
        - Falling Channel
        - Double Top
        - Double Bottom
        - Bullish Flag
        - Bearish Flag
        """
    )
    st.stop()

results = st.session_state["scan_results"]

if results.empty:
    st.warning("No qualifying patterns were found.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Stocks scanned", max_stocks)
c2.metric("Patterns found", len(results))
c3.metric(
    "High confidence ≥ 85",
    int((results["Confidence"] >= 85).sum()),
)

st.subheader("Detected patterns")

display = results.sort_values(
    ["Confidence", "Stock"],
    ascending=[False, True],
)

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "⬇️ Download scan CSV",
    display.to_csv(index=False),
    "pattern_scan.csv",
    "text/csv",
)

st.divider()
st.subheader("Stock chart")

selected = st.selectbox(
    "Select stock",
    sorted(results["Stock"].unique()),
)

data = load_data(selected + ".NS")

fig = go.Figure(
    go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name=selected,
    )
)

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    title=f"{selected} — Daily chart",
)

st.plotly_chart(fig, use_container_width=True)

stock_patterns = display[display["Stock"] == selected]

st.subheader("Detected patterns for " + selected)
st.dataframe(
    stock_patterns,
    use_container_width=True,
    hide_index=True,
)

st.warning(
    "Confidence is a pattern-quality ranking score, not a probability "
    "and not a trading recommendation."
)
