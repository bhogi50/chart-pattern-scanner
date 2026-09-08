import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from detector import detect_patterns

st.set_page_config(page_title="Chart Pattern Scanner",page_icon="📈",layout="wide")

# Constituent lists can be expanded without changing the scanner engine.
UNIVERSES={
"NIFTY 50":["ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BEL","BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DRREDDY","EICHERMOT","ETERNAL","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC","JIOFIN","JSWSTEEL","KOTAKBANK","LT","M&M","MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBILIFE","SBIN","SHRIRAMFIN","SUNPHARMA","TATACONSUM","TATAMOTORS","TATASTEEL","TCS","TECHM","TITAN","TRENT","ULTRACEMCO"],
"NIFTY Next 50":[],"NIFTY Midcap 150":[],"NIFTY Smallcap 250":[]}

REL={"Daily":("1d","4h","1y"),"Weekly":("1wk","1d","5y"),"Monthly":("1mo","1wk","10y")}

@st.cache_data(ttl=900,show_spinner=False)
def data(symbol,period,interval):
    import yfinance as yf
    x=yf.download(symbol,period=period,interval=interval,auto_adjust=False,progress=False)
    if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
    return x.dropna()

with st.sidebar:
    st.header("Scanner")
    universe=st.selectbox("Universe",list(UNIVERSES))
    tide=st.selectbox("Candle / Tide",["Daily","Weekly","Monthly"])
    view=st.radio("View",["All","Bullish","Bearish"])
    st.markdown("**Tide → Wave**")
    st.write("Daily → 4H")
    st.write("Weekly → Daily")
    st.write("Monthly → Weekly")
    scan=st.button("🔎 SCAN",type="primary",use_container_width=True)

if scan:
    symbols=UNIVERSES[universe]
    if not symbols:
        st.warning("This universe is enabled in the UI, but its constituent list is not bundled yet. NIFTY 50 is ready.")
        st.stop()
    ti,wi,tp=REL[tide]; rows=[]; bar=st.progress(0)
    for n,s in enumerate(symbols,1):
        try:
            td=data(s+".NS",tp,ti)
            wp=data(s+".NS","60d" if tide=="Daily" else ("2y" if tide=="Weekly" else "5y"),wi)
            for p in detect_patterns(td,td,wp):
                if view=="All" or p["Direction"]==view:rows.append({"Stock":s,"Tide":tide,**p})
        except Exception:pass
        bar.progress(n/len(symbols))
    st.session_state.results=pd.DataFrame(rows)
    st.session_state.filters=(universe,tide,view)

if "results" not in st.session_state:
    st.info("Select the Universe, Candle/Tide and View, then press SCAN.")
    st.stop()

r=st.session_state.results
if r.empty:
    st.warning("No qualifying patterns found.")
    st.stop()

a,b,c=st.columns(3)
a.metric("Stocks scanned",len(UNIVERSES[st.session_state.filters[0]]))
b.metric("Patterns found",len(r))
c.metric("Confidence ≥ 80",int((r.Confidence>=80).sum()))

st.subheader("Scan results")
st.dataframe(r.sort_values(["Confidence","Stock"],ascending=[False,True]),use_container_width=True,hide_index=True)
st.download_button("⬇️ Download CSV",r.to_csv(index=False),"pattern_scan.csv","text/csv")

st.divider()
selected=st.selectbox("Stock chart",sorted(r.Stock.unique()))
ti,wi,tp=REL[st.session_state.filters[1]]
d=data(selected+".NS",tp,ti)
fig=go.Figure(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close))
fig.update_layout(title=f"{selected} — {st.session_state.filters[1]} Tide",height=600,xaxis_rangeslider_visible=False)
st.plotly_chart(fig,use_container_width=True)

st.subheader("Confidence calculation")
st.markdown("""
**Confidence is a technical-quality score, not a probability of profit.**

- **30% Geometry:** trendline fit, convergence and pivot structure.
- **25% Tide alignment:** higher-timeframe trend agrees with the pattern.
- **20% Wave alignment:** lower-timeframe trend confirms the Tide.
- **10% Volume:** latest volume versus recent average.
- **10% Breakout:** currently a neutral placeholder; will become breakout/proximity logic.
- **5% Cleanliness:** structural pivot/touch quality.

This makes the score reflect **pattern + Tide + Wave**, rather than pattern shape alone.
""")
st.warning("The Daily → 4H leg uses Yahoo Finance in this prototype. Intraday history/availability is provider-dependent; a proper NSE intraday provider should be used before relying on long historical 4H scans.")
