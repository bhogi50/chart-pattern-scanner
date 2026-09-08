
import io,requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from detector import scan_setups

st.set_page_config(page_title="ASTA Tide-Wave Scanner",page_icon="📈",layout="wide")
URLS={
"NIFTY 50":"https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
"NIFTY Next 50":"https://www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
"NIFTY Midcap 150":"https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
"NIFTY Smallcap 250":"https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv"}
REL={"Daily":("1d","4h","1y","60d"),"Weekly":("1wk","1d","5y","2y"),"Monthly":("1mo","1wk","10y","5y")}

@st.cache_data(ttl=86400)
def symbols(name):
    r=requests.get(URLS[name],headers={"User-Agent":"Mozilla/5.0"},timeout=15); r.raise_for_status()
    x=pd.read_csv(io.BytesIO(r.content)); c=next(c for c in x.columns if c.upper()=="SYMBOL")
    return x[c].dropna().astype(str).str.strip().tolist()

@st.cache_data(ttl=900)
def data(sym,period,interval):
    import yfinance as yf
    x=yf.download(sym,period=period,interval=interval,auto_adjust=False,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x.dropna()

st.title("📈 ASTA Tide → Wave → Setup Scanner")
st.caption("Based on the three ASTA checklists in GEO PAN - SETUPS.pdf")

with st.sidebar:
    st.header("Scanner")
    universe=st.selectbox("Universe",list(URLS))
    tide=st.selectbox("Candle / Tide",["Daily","Weekly","Monthly"])
    view=st.radio("View",["All","Bullish","Bearish"])
    setups=st.multiselect("Setup",["ASTA Triple Screen","ASTA Swing Trader","ASTA Momentum Trader"],default=["ASTA Triple Screen","ASTA Swing Trader","ASTA Momentum Trader"])
    minimum=st.slider("Minimum confidence",0,100,0,5)
    scan=st.button("🔎 SCAN",type="primary",use_container_width=True)
    st.divider(); st.markdown("**Tide → Wave**")
    st.write("Daily → 4H"); st.write("Weekly → Daily"); st.write("Monthly → Weekly")

if scan:
    try: syms=symbols(universe)
    except Exception as e: st.error(f"Could not load {universe} constituents: {e}"); st.stop()
    ti,wi,tp,wp=REL[tide]; rows=[]; bar=st.progress(0)
    for i,s in enumerate(syms,1):
        try:
            td=data(s+".NS",tp,ti); wd=data(s+".NS",wp,wi)
            if len(td)>=60 and len(wd)>=40:
                for p in scan_setups(td,wd):
                    if p["Setup"] in setups and (view=="All" or p["Direction"]==view) and p["Confidence"]>=minimum:
                        rows.append({"Stock":s,"Tide":tide,**p})
        except Exception: pass
        bar.progress(i/len(syms))
    st.session_state.results=pd.DataFrame(rows); st.session_state.filter=(universe,tide,view)

if "results" not in st.session_state:
    st.info("Select Universe + Tide + Direction, then press SCAN."); st.stop()
r=st.session_state.results
if r.empty: st.warning("No qualifying ASTA setups found."); st.stop()
r=r.sort_values(["Confidence","Stock"],ascending=[False,True])
a,b,c,d=st.columns(4); a.metric("Universe",st.session_state.filter[0]); b.metric("Stocks",r.Stock.nunique()); c.metric("Setups",len(r)); d.metric("≥80",int((r.Confidence>=80).sum()))
st.subheader("Scan results")
st.dataframe(r[["Stock","Tide","Setup","Direction","Eligible","Mandatory","Supporting","Confidence","Checklist"]],use_container_width=True,hide_index=True)
st.download_button("⬇️ Download CSV",r.to_csv(index=False),"asta_scan.csv","text/csv")
st.divider()
sel=st.selectbox("Stock chart",sorted(r.Stock.unique()))
ti,wi,tp,wp=REL[st.session_state.filter[1]]; ch=data(sel+".NS",tp,ti)
# ---------- ASTA annotated chart ----------
from detector import geometry, indicators, trend, ema

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=ch.index, open=ch.Open, high=ch.High, low=ch.Low, close=ch.Close,
    name="Candles"
))

pats, geom_score, H, L = geometry(ch)
latest = ch.Close.iloc[-1]

# Draw pivot points.
if H:
    fig.add_trace(go.Scatter(
        x=ch.index[H], y=ch.High.iloc[H], mode="markers",
        marker=dict(symbol="triangle-down", size=9),
        name="Swing High"
    ))
if L:
    fig.add_trace(go.Scatter(
        x=ch.index[L], y=ch.Low.iloc[L], mode="markers",
        marker=dict(symbol="triangle-up", size=9),
        name="Swing Low"
    ))

# Draw the most recent chart-pattern geometry.
if len(H) >= 2 and len(L) >= 2:
    hidx = H[-5:] if len(H) >= 5 else H
    lidx = L[-5:] if len(L) >= 5 else L
    import numpy as np
    mh, bh = np.polyfit(hidx, ch.High.iloc[hidx].to_numpy(float), 1)
    ml, bl = np.polyfit(lidx, ch.Low.iloc[lidx].to_numpy(float), 1)
    start_i = max(0, min(hidx[0], lidx[0]) - 5)
    end_i = len(ch) - 1
    xline = np.array([start_i, end_i])
    highline = mh*xline + bh
    lowline = ml*xline + bl

    # Only show the lines when the detector found a geometrical pattern.
    if pats:
        fig.add_trace(go.Scatter(
            x=ch.index[xline], y=highline, mode="lines",
            line=dict(dash="dash", width=2), name="Pattern resistance"
        ))
        fig.add_trace(go.Scatter(
            x=ch.index[xline], y=lowline, mode="lines",
            line=dict(dash="dash", width=2), name="Pattern support"
        ))

# Add the selected setup's reference levels.
selected_rows = r[r.Stock == sel]
if not selected_rows.empty:
    best = selected_rows.iloc[0]
    direction = best["Direction"]
    entry = float(latest)
    if direction == "Bullish":
        stop = float(ch.Low.iloc[-1])
        target = float(ch.High.iloc[-1])
    else:
        stop = float(ch.High.iloc[-1])
        target = float(ch.Low.iloc[-1])

    fig.add_hline(y=entry, line_dash="dot", annotation_text="Entry / current price",
                  annotation_position="top left")
    fig.add_hline(y=stop, line_dash="dash", annotation_text="Stop reference",
                  annotation_position="bottom left")
    fig.add_hline(y=target, line_dash="dash", annotation_text="Target reference",
                  annotation_position="top right")

    # Highlight the current setup in the title.
    title = (
        f"{sel} — {st.session_state.filter[1]} Tide | "
        f"{best['Setup']} • {direction} • Confidence {best['Confidence']}/100"
    )
else:
    title = f"{sel} — {st.session_state.filter[1]} Tide"

fig.update_layout(
    title=title,
    height=680,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02, x=0)
)
st.plotly_chart(fig, use_container_width=True)

# Pattern explanation directly below the chart.
pattern_names = ", ".join(f"{p[0]} ({p[1]})" for p in pats) if pats else "No qualifying geometric pattern detected"
st.caption(f"Detected geometry: {pattern_names}. Geometry score: {geom_score:.1f}/100.")
st.subheader("Confidence & checklist")
st.markdown("""
**Confidence is setup quality, not probability of profit.**

Triple Screen = 35% mandatory structure + 65% supporting checks.
Swing Trader = 45% mandatory structure + 55% supporting checks.
Momentum Trader = 70% mandatory structure + 30% supporting checks.

The PDF's exact formulas for proprietary/undefined terms such as **TI, TLBO/TLBD,
ATM PE/CE-TMJ and TMJ** are not specified on the three pages. The implementation
therefore uses clearly labelled technical proxies where practical and does not
fabricate derivatives/OI data.
""")
st.info("The constituent lists are loaded from Nifty Indices CSV endpoints. Yahoo Finance supplies OHLCV. Daily → 4H is subject to Yahoo intraday history limits; use a dedicated NSE intraday provider before relying on it for research/live decisions.")
