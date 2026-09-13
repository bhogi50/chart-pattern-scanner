"""Nifty Chart Pattern Scanner — human-like structure pipeline."""
import streamlit as st
import pandas as pd
from constituents import get_index_constituents
from scanner import run_scan
from data import fetch_ohlc
from patterns import analyze_chart
from chart_utils import build_chart_payload

try:
    from streamlit_lightweight_charts import renderLightweightCharts
    LWC_AVAILABLE=True
except Exception:
    LWC_AVAILABLE=False

st.set_page_config(page_title="Nifty Pattern Scanner",page_icon="📈",layout="wide")
INDEX_OPTIONS=["Nifty 50","Nifty Next 50","Nifty Midcap 150","Nifty Smallcap 250"]
TIMEFRAME_OPTIONS=["Daily","Weekly","Monthly"]
PATTERN_FILTER_OPTIONS=["All","Bullish","Bearish","Neutral"]

st.title("📈 Nifty Pattern Scanner")
st.caption("Structure → Location → Relevant Pattern → Confirmation")

with st.sidebar:
    st.header("Scan Settings")
    index_choice=st.selectbox("Index",INDEX_OPTIONS)
    timeframe=st.selectbox("Timeframe",TIMEFRAME_OPTIONS)
    num_candles=st.slider("Number of candles",50,400,150,10)
    pattern_filter=st.selectbox("View",PATTERN_FILTER_OPTIONS)
    check_confluence=st.checkbox("Check higher-timeframe confluence",True)
    scan_clicked=st.button("🔍 Scan",type="primary",use_container_width=True)

if "scan_results" not in st.session_state: st.session_state.scan_results=None
if "selected_ticker" not in st.session_state: st.session_state.selected_ticker=None

if scan_clicked:
    constituents=get_index_constituents(index_choice)
    if constituents.empty: st.error("Could not load index constituents.")
    else:
        bar=st.progress(0.0,text="Starting scan...")
        def progress(done,total,symbol,stage="scanning"):
            label="Scanning" if stage=="scanning" else "Checking higher timeframe"
            bar.progress(done/max(total,1),text=f"{label} {symbol} ({done}/{total})")
        with st.spinner("Analyzing structure, location and patterns..."):
            st.session_state.scan_results=run_scan(constituents,timeframe,num_candles,pattern_filter,progress,check_confluence)
        bar.empty();st.session_state.selected_ticker=None

results=st.session_state.scan_results
if results is None:
    st.info("Set scan parameters and click **Scan**.")
elif results.empty:
    st.warning("No relevant setups found for this selection.")
else:
    st.subheader("Scan Results")
    cols=["Symbol","Company","Structure","Location","Pattern","Direction","Confidence","Status","LastPrice","Confluence","Note"]
    display=results[cols].copy();display["Confidence"]=display["Confidence"].map(lambda x:f"{x:.1f}")
    event=st.dataframe(display,use_container_width=True,hide_index=True,on_select="rerun",selection_mode="single-row",key="results_table")
    selected=event.selection.rows if hasattr(event,"selection") else []
    if selected: st.session_state.selected_ticker=results.iloc[selected[0]]["Ticker"]
    else:
        pick=st.selectbox("Or select a stock",results["Symbol"].astype(str)+" — "+results["Pattern"].astype(str),index=None)
        if pick: st.session_state.selected_ticker=results.loc[results["Symbol"]==pick.split(" — ")[0],"Ticker"].iloc[0]

ticker=st.session_state.selected_ticker
if ticker:
    symbol=ticker.replace(".NS","")
    st.divider();st.subheader(f"📊 {symbol} — {timeframe}")
    df=fetch_ohlc(ticker,timeframe,num_candles)
    if df.empty: st.error("Could not load price data.")
    else:
        analysis=analyze_chart(df)
        s=analysis["structure"];loc=analysis["location"]
        st.markdown(f"### Human interpretation")
        direction_phrase={"Channel Up":"rising channel","Channel Down":"falling channel","Ascending Triangle":"ascending triangle","Descending Triangle":"descending triangle","Flat / Range":"flat/range","Unclear":"unclear structure"}[s.name]
        st.write(f"Price is currently in a **{direction_phrase}**. It is **{loc.zone.lower()}**, with the structure position **{loc.structure_position.lower()}** and the latest candle showing **{loc.candle_behavior.lower()}**.")
        if analysis["patterns"]:
            top=analysis["patterns"][0]
            st.write(f"The strongest relevant interpretation is **{top.name}** ({top.confidence:.1f}% confidence), status **{top.status}**. {top.note}.")
        c1,c2,c3=st.columns(3)
        c1.metric("Last Close",f"₹{df.Close.iloc[-1]:.2f}")
        c2.metric("Structure",s.name)
        c3.metric("Location",loc.zone)
        st.markdown("**Resistance**")
        st.write(" · ".join(f"R{i+1}: ₹{z['price']:.2f} ({z['touches']} touches)" for i,z in enumerate(analysis["resistances"])) or "None")
        st.markdown("**Support**")
        st.write(" · ".join(f"S{i+1}: ₹{z['price']:.2f} ({z['touches']} touches)" for i,z in enumerate(analysis["supports"])) or "None")
        st.markdown("**Relevant patterns**")
        for m in analysis["patterns"]:
            st.write(f"**{m.name}** — {m.direction} · {m.confidence:.1f} · {m.status}")
            st.caption(f"{m.note} | Structure: {m.structure} | Location: {m.location}")
        if LWC_AVAILABLE:
            best=analysis["patterns"][0] if analysis["patterns"] else None
            payload=build_chart_payload(df,analysis["supports"],analysis["resistances"],best)
            renderLightweightCharts(payload,key=f"chart_{ticker}_{timeframe}_{num_candles}")
        else: st.line_chart(df["Close"])
