
import io, requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from detector import geometry, pattern_target

st.set_page_config(page_title="Chart Pattern Scanner", page_icon="📈", layout="wide")

URLS = {
    "NIFTY 50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "NIFTY Next 50": "https://www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
    "NIFTY Midcap 150": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    "NIFTY Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

# Pattern detection timeframe is now explicit.
# We do NOT mix Tide/Wave/ASTA conditions.
TF = {
    "Daily": ("1d", "2y"),
    "Weekly": ("1wk", "7y"),
    "Monthly": ("1mo", "15y"),
}

@st.cache_data(ttl=86400)
def symbols(name):
    r = requests.get(URLS[name], headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    x = pd.read_csv(io.BytesIO(r.content))
    col = next(c for c in x.columns if c.upper() == "SYMBOL")
    return x[col].dropna().astype(str).str.strip().tolist()

@st.cache_data(ttl=900)
def data(sym, period, interval):
    import yfinance as yf
    x = yf.download(sym, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    return x.dropna()

st.title("📈 Chart Pattern Scanner")
st.caption("Pure price-action pattern detection — no ASTA setup scoring")

with st.sidebar:
    st.header("Scanner")
    universe = st.selectbox("Universe", list(URLS))
    timeframe = st.selectbox("Pattern timeframe", ["Daily", "Weekly", "Monthly"])
    view = st.radio("View", ["All", "Bullish", "Bearish", "Neutral"])
    patterns = st.multiselect(
        "Patterns",
        [
            "Symmetrical Triangle", "Ascending Triangle", "Descending Triangle",
            "Rising Wedge", "Falling Wedge", "Rising Channel", "Falling Channel",
            "Double Bottom", "Double Top", "Bullish Flag", "Bearish Flag",
            "Head and Shoulders", "Inverse Head and Shoulders", "Rectangle",
            "Cup and Handle", "Bullish Pennant", "Bearish Pennant",
            "Resistance Breakout", "Support Breakdown"
        ],
        default=[
            "Symmetrical Triangle", "Ascending Triangle", "Descending Triangle",
            "Rising Wedge", "Falling Wedge", "Rising Channel", "Falling Channel",
            "Double Bottom", "Double Top", "Bullish Flag", "Bearish Flag",
            "Head and Shoulders", "Inverse Head and Shoulders", "Rectangle",
            "Cup and Handle", "Bullish Pennant", "Bearish Pennant",
            "Resistance Breakout", "Support Breakdown"
        ]
    )
    scan = st.button("🔎 SCAN PATTERNS", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**How timeframe works**")
    st.write("Daily = patterns from daily candles")
    st.write("Weekly = patterns from weekly candles")
    st.write("Monthly = patterns from monthly candles")
    st.caption("No 4H/1D wave confirmation is used in this pattern-only version.")

if scan:
    try:
        syms = symbols(universe)
    except Exception as e:
        st.error(f"Could not load {universe} constituents: {e}")
        st.stop()

    interval, period = TF[timeframe]
    rows = []
    bar = st.progress(0)

    for i, s in enumerate(syms, 1):
        try:
            df = data(s + ".NS", period, interval)
            if len(df) < 60:
                continue

            found, geometry_score, H, L = geometry(df)

            for name, direction in found:
                if name not in patterns:
                    continue
                if view != "All" and direction != view:
                    continue

                rows.append({
                    "Stock": s,
                    "Timeframe": timeframe,
                    "Pattern": name,
                    "Direction": direction,
                    "Pattern Strength": round(float(geometry_score), 1),
                    "Detected On": df.index[-1].strftime("%Y-%m-%d"),
                })
        except Exception:
            pass
        bar.progress(i / len(syms))

    st.session_state.results = pd.DataFrame(rows)
    st.session_state.scan_filter = (universe, timeframe, view)

if "results" not in st.session_state:
    st.info("Select a universe and pattern timeframe, then press SCAN PATTERNS.")
    st.stop()

r = st.session_state.results
if r.empty:
    st.warning("No selected chart patterns found.")
    st.stop()

r = r.sort_values(["Pattern Strength", "Stock"], ascending=[False, True])

a, b, c, d = st.columns(4)
a.metric("Universe", st.session_state.scan_filter[0])
b.metric("Timeframe", st.session_state.scan_filter[1])
c.metric("Stocks", r.Stock.nunique())
d.metric("Patterns", len(r))

st.subheader("Detected chart patterns")
st.dataframe(
    r[["Stock", "Timeframe", "Pattern", "Direction", "Pattern Strength", "Detected On"]],
    use_container_width=True,
    hide_index=True
)
st.download_button(
    "⬇️ Download CSV", r.to_csv(index=False),
    "chart_pattern_scan.csv", "text/csv"
)

st.divider()
st.subheader("Stocks")
st.caption("Charts are hidden from the scan page. Click View to open the stock details.")
if "selected_stock" not in st.session_state: st.session_state.selected_stock=None
if "selected_pattern" not in st.session_state: st.session_state.selected_pattern=None

for i,row in r.reset_index(drop=True).iterrows():
    c1,c2,c3,c4,c5,c6=st.columns([1.4,1.0,2.5,1.0,1.1,.7])
    c1.write(f"**{row.Stock}**"); c2.write(row.Timeframe); c3.write(row.Pattern); c4.write(row.Direction); c5.write(f"{row['Pattern Strength']:.1f}")
    if c6.button("View",key=f"view_{i}_{row.Stock}_{row.Pattern}"):
        st.session_state.selected_stock=row.Stock; st.session_state.selected_pattern=row.Pattern; st.rerun()

if st.session_state.selected_stock:
    sel=st.session_state.selected_stock
    sr=r[r.Stock==sel].reset_index(drop=True)
    opts=sr.Pattern.tolist()
    if st.session_state.selected_pattern not in opts: st.session_state.selected_pattern=opts[0]
    pat=st.selectbox("Detected pattern",opts,index=opts.index(st.session_state.selected_pattern))
    st.session_state.selected_pattern=pat
    selected=sr[sr.Pattern==pat].iloc[0]
    interval,period=TF[selected.Timeframe]; ch=data(sel+'.NS',period,interval)
    found,geometry_score,H,L=geometry(ch); details=pattern_target(ch,pat,selected.Direction,H,L)
    st.divider(); st.subheader(f"{sel} — {pat}")
    st.caption(f"{selected.Timeframe} candles · {selected.Direction} · strength {selected['Pattern Strength']:.1f}/100")
    a,b,c,d,e=st.columns(5)
    a.metric("Current",f"₹{details['current']:,.2f}"); b.metric("Entry / Breakout",f"₹{details['entry']:,.2f}")
    c.metric("Stop / Invalidation",f"₹{details['stop']:,.2f}" if details['stop'] is not None else '—')
    d.metric("Target 1",f"₹{details['target1']:,.2f}" if details['target1'] is not None else '—')
    e.metric("Target 2",f"₹{details['target2']:,.2f}" if details['target2'] is not None else '—')
    if details['stop'] is not None:
        risk=abs(details['entry']-details['stop']); reward=abs(details['target1']-details['entry']); rr=reward/risk if risk else None
        st.info(f"Target method: **{details['method']}**"+(f" · Risk/Reward to T1: **1:{rr:.2f}**" if rr else ''))
    else: st.info(f"Target method: **{details['method']}**")
    import numpy as np
    fig=go.Figure(go.Candlestick(x=ch.index,open=ch.Open,high=ch.High,low=ch.Low,close=ch.Close,name='Candles'))
    if H: fig.add_trace(go.Scatter(x=ch.index[H],y=ch.High.iloc[H],mode='markers',marker=dict(symbol='triangle-down',size=8),name='Swing High'))
    if L: fig.add_trace(go.Scatter(x=ch.index[L],y=ch.Low.iloc[L],mode='markers',marker=dict(symbol='triangle-up',size=8),name='Swing Low'))
    if len(H)>=2 and len(L)>=2:
        hi=H[-5:]; lo=L[-5:]; mh,bh=np.polyfit(hi,ch.High.iloc[hi].to_numpy(float),1); ml,bl=np.polyfit(lo,ch.Low.iloc[lo].to_numpy(float),1); xx=np.array([max(0,min(hi[0],lo[0])-5),len(ch)-1])
        fig.add_trace(go.Scatter(x=ch.index[xx],y=mh*xx+bh,mode='lines',line=dict(dash='dash',width=2),name='Resistance'))
        fig.add_trace(go.Scatter(x=ch.index[xx],y=ml*xx+bl,mode='lines',line=dict(dash='dash',width=2),name='Support'))
    for y,label,dash in [(details['entry'],'Entry / breakout','dot'),(details['target1'],'Target 1','dash'),(details['target2'],'Target 2','dash'),(details['stop'],'Stop / invalidation','dash')]:
        if y is not None: fig.add_hline(y=y,line_dash=dash,annotation_text=label,annotation_position='top left')
    fig.update_layout(title=f"{sel} — {pat} | {selected.Direction}",height=680,xaxis_rangeslider_visible=False,hovermode='x unified')
    st.plotly_chart(fig,use_container_width=True)
    if st.button('← Back to scan results'):
        st.session_state.selected_stock=None; st.session_state.selected_pattern=None; st.rerun()

st.subheader("What the detector is doing")
st.markdown("""
**1. Select the candle timeframe.** The scanner detects patterns only on that
timeframe; there is no ASTA Tide/Wave/Setup layer.

**2. Find swing highs and swing lows.** These pivots form the structural points
used to construct pattern boundaries.

**3. Fit structural trendlines.** Recent swing highs form the resistance side;
recent swing lows form the support side.

**4. Classify geometry.** The slope and convergence/divergence of those lines
are used to identify triangles, wedges and channels.

**5. Detect repeated extrema.** Double Top/Bottom requires at least 8 candles
between the two swing points and a price difference within 3%.

**6. Detect flags.** A strong prior move followed by a relatively tight
consolidation is classified as a bullish or bearish flag.

**Pattern Strength is geometry quality, not probability of success.**
""")
