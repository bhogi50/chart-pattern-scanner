
import io, requests, pandas as pd, streamlit as st, plotly.graph_objects as go
from detector import geometry, pattern_target

st.set_page_config(page_title="Chart Pattern Scanner", page_icon="📈", layout="wide")

UNIVERSES={
"NIFTY 50":"https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
"NIFTY Next 50":"https://www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
"NIFTY Midcap 150":"https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
"NIFTY Smallcap 250":"https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv"}

CANDLE_OPTIONS={
"Daily":[50,100,150,200,250,300],
"Weekly":[26,52,78,104,156],
"Monthly":[12,24,36,60,120]}

@st.cache_data(ttl=86400)
def get_symbols(name):
    r=requests.get(UNIVERSES[name],headers={"User-Agent":"Mozilla/5.0"},timeout=20)
    r.raise_for_status()
    x=pd.read_csv(io.BytesIO(r.content))
    col=next(c for c in x.columns if c.upper()=="SYMBOL")
    return x[col].dropna().astype(str).str.strip().tolist()

def period_for(timeframe,n):
    return {"Daily":max(2,int(n*1.7)), "Weekly":max(2,int(n*1.7)), "Monthly":max(2,int(n*1.7))}[timeframe]

@st.cache_data(ttl=900)
def get_prices(symbol,timeframe,candle_count):
    import yfinance as yf
    interval={"Daily":"1d","Weekly":"1wk","Monthly":"1mo"}[timeframe]
    period=f"{period_for(timeframe,candle_count)}d" if timeframe=="Daily" else (
        f"{max(2,int(candle_count*1.7/52)+1)}y" if timeframe=="Weekly" else
        f"{max(2,int(candle_count*1.7/12)+1)}y")
    x=yf.download(symbol,period=period,interval=interval,auto_adjust=False,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    needed={"Open","High","Low","Close"}
    if x.empty or not needed.issubset(x.columns): return pd.DataFrame()
    return x.dropna(subset=list(needed)).tail(candle_count)

st.title("📈 Chart Pattern Scanner")
st.caption("Pure pattern detection • one result row per stock • details/chart appear only after View")

with st.sidebar:
    st.header("Scan")
    universe=st.selectbox("Universe",list(UNIVERSES))
    timeframe=st.selectbox("Pattern timeframe",list(CANDLE_OPTIONS))
    candles=st.selectbox("Number of candles",CANDLE_OPTIONS[timeframe],index=CANDLE_OPTIONS[timeframe].index(
        {"Daily":100,"Weekly":52,"Monthly":36}[timeframe]))
    view=st.radio("Direction filter",["All","Bullish","Bearish","Neutral"])
    scan=st.button("🔎 SCAN PATTERNS",type="primary",use_container_width=True)
    st.caption(f"{candles} {timeframe.lower()} candles will be used for the scan.")

if scan:
    try: syms=get_symbols(universe)
    except Exception as e:
        st.error(f"Unable to load {universe}: {e}"); st.stop()
    rows=[]; bar=st.progress(0)
    for i,sym in enumerate(syms,1):
        try:
            df=get_prices(sym+".NS",timeframe,candles)
            if len(df)<max(30,min(candles,45)): continue
            found,strength,_,_=geometry(df)
            selected=[(p,d) for p,d in found if view=="All" or d==view]
            if selected:
                dirs=sorted(set(d for _,d in selected))
                rows.append({
                    "Stock":sym,
                    "Timeframe":timeframe,
                    "Candles":len(df),
                    "Patterns Detected":", ".join(p for p,_ in selected),
                    "Direction":", ".join(dirs),
                    "Pattern Strength":strength})
        except Exception:
            continue
        bar.progress(i/max(1,len(syms)))
    st.session_state.results=pd.DataFrame(rows)
    st.session_state.scan_config=(universe,timeframe,candles,view)
    st.session_state.selected=None

if "results" not in st.session_state:
    st.info("Choose the scan settings and press SCAN PATTERNS.")
    st.stop()

r=st.session_state.results
if r.empty:
    st.warning("No matching patterns found for the selected settings.")
    st.stop()

st.subheader("Detected patterns")
st.caption("One row per stock. Multiple detected patterns are listed in the Patterns Detected column.")

show=r[["Stock","Timeframe","Candles","Patterns Detected","Direction","Pattern Strength"]].copy()
show["Pattern Strength"]=show["Pattern Strength"].map(lambda x:f"{x:.1f}")
show=show.sort_values("Stock")

st.dataframe(
    show,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Stock": st.column_config.TextColumn("Stock", width="small"),
        "Timeframe": st.column_config.TextColumn("Timeframe", width="small"),
        "Candles": st.column_config.NumberColumn("Candles", width="small"),
        "Patterns Detected": st.column_config.TextColumn("Patterns Detected", width="large"),
        "Direction": st.column_config.TextColumn("Direction", width="medium"),
        "Pattern Strength": st.column_config.TextColumn("Pattern Strength", width="medium"),
    },
)

stock=st.selectbox("Select a stock to view details",["— Select —"]+show.Stock.tolist())
if stock!="— Select —":
    st.session_state.selected=stock

if st.session_state.get("selected"):
    sel=st.session_state.selected
    row=r[r.Stock==sel].iloc[0]
    df=get_prices(sel+".NS",row.Timeframe,int(row.Candles))
    if df.empty:
        st.error("Price data could not be loaded for this stock. Please scan again.")
        st.stop()

    found,strength,H,L=geometry(df)
    available=[(p,d) for p,d in found if p in row["Patterns Detected"].split(", ")]
    if not available:
        st.warning("The pattern is no longer detected on the latest data. Please scan again.")
        st.stop()

    st.divider()
    st.subheader(f"{sel} — Pattern Details")
    st.write("**Patterns detected:** "+", ".join(p for p,_ in available))

    labels=[p for p,_ in available]
    pat=st.selectbox("Pattern to inspect",labels)
    direction=dict(available)[pat]
    details=pattern_target(df,pat,direction,H,L)
    if details is None:
        st.error("Not enough data to calculate targets.")
        st.stop()

    def money(x):
        return "—" if x is None else f"₹{float(x):,.2f}"

    levels = pd.DataFrame([
        ["Current", money(details["current"])],
        ["Entry / Breakout", money(details["entry"])],
        ["Stop / Invalidation", money(details["stop"])],
        ["Target 1", money(details["target1"])],
        ["Target 2", money(details["target2"])],
    ], columns=["Level", "Price"])

    st.table(levels)

    risk=abs(details["entry"]-details["stop"]) if details["stop"] is not None else 0
    reward=abs(details["target1"]-details["entry"])
    rr=reward/risk if risk else None
    st.info(f"Target method: **{details['method']}**"+(f" · R/R to T1: **1:{rr:.2f}**" if rr else ""))

    st.caption(
        f"Exact levels — Current: {money(details['current'])} · "
        f"Entry: {money(details['entry'])} · Stop: {money(details['stop'])} · "
        f"T1: {money(details['target1'])} · T2: {money(details['target2'])}"
    )

    fig=go.Figure(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="Candles"))
    if H: fig.add_trace(go.Scatter(x=df.index[H],y=df.High.iloc[H],mode="markers",name="Swing High"))
    if L: fig.add_trace(go.Scatter(x=df.index[L],y=df.Low.iloc[L],mode="markers",name="Swing Low"))
    for y,label,dash in [(details["entry"],"Entry / breakout","dot"),
                         (details["target1"],"Target 1","dash"),
                         (details["target2"],"Target 2","dash"),
                         (details["stop"],"Stop / invalidation","dash")]:
        if y is not None: fig.add_hline(y=y,line_dash=dash,annotation_text=label)
    fig.update_layout(height=680,xaxis_rangeslider_visible=False,hovermode="x unified",
                      title=f"{sel} — {pat} ({direction})")
    st.plotly_chart(fig,use_container_width=True)
    if st.button("← Close details"):
        st.session_state.selected=None
        st.rerun()
