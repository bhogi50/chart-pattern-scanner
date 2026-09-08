
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from detector import detect_patterns

st.set_page_config(page_title="Chart Pattern Scanner", layout="wide")
st.title("📈 Chart Pattern Scanner")
st.caption("Rule-based technical pattern detection. Educational/research use; not investment advice.")

with st.sidebar:
    st.header("Data")
    source = st.radio("Input", ["CSV", "Yahoo Finance"])
    lookback = st.slider("Pattern lookback", 60, 250, 120)
    if source == "CSV":
        uploaded = st.file_uploader("Upload OHLCV CSV", type=["csv"])
        st.info("CSV columns required: Date, Open, High, Low, Close. Volume is optional.")
    else:
        ticker = st.text_input("NSE ticker", "TCS.NS")
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)

df = None
symbol = "CSV"

if source == "CSV" and uploaded:
    df = pd.read_csv(uploaded)
    symbol = uploaded.name
elif source == "Yahoo Finance":
    try:
        import yfinance as yf
        data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        df = data.reset_index()
        symbol = ticker
    except Exception as e:
        st.error(f"Could not download data: {e}")

if df is not None:
    df.columns = [str(c).strip().title() for c in df.columns]
    if "Date" in df:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
    required = {"Open","High","Low","Close"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing columns: {', '.join(sorted(missing))}")
        st.stop()

    df = df.dropna(subset=list(required))
    results = detect_patterns(df, lookback=lookback)

    c1,c2,c3 = st.columns(3)
    c1.metric("Symbol", symbol)
    c2.metric("Detected patterns", len(results))
    c3.metric("Latest close", f"₹{float(df['Close'].iloc[-1]):,.2f}")

    if results:
        out = pd.DataFrame(results, columns=["Pattern","Direction","Confidence"])
        st.subheader("Detected patterns")
        st.dataframe(out, use_container_width=True, hide_index=True)
    else:
        st.warning("No qualifying pattern found in the selected lookback window.")

    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=symbol
    )])
    fig.update_layout(height=600, xaxis_rangeslider_visible=False,
                      title=f"{symbol} — price chart")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("""
### Quick start

1. Upload an OHLCV CSV, or choose Yahoo Finance.
2. Select an NSE ticker such as `TCS.NS`, `RELIANCE.NS`, or `INFY.NS`.
3. The engine searches for triangles, wedges, channels, flags and double tops/bottoms.
4. Each match receives a confidence score.

**Next upgrade:** add a full NSE universe scanner, breakout confirmation, support/resistance levels, volume confirmation, and chart overlays for every detected pattern.
""")
