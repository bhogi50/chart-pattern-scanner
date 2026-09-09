
import io, requests, pandas as pd, streamlit as st, plotly.graph_objects as go
from detector import geometry, pattern_target

st.set_page_config(page_title="Chart Pattern Scanner",page_icon="📈",layout="wide")
UNIVERSES={
"NIFTY 50":"https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
"NIFTY Next 50":"https://www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
"NIFTY Midcap 150":"https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
"NIFTY Smallcap 250":"https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv"}
TF={"Daily":("1d","2y"),"Weekly":("1wk","7y"),"Monthly":("1mo","15y")}

@st.cache_data(ttl=86400)
def symbols(name):
    x=pd.read_csv(io.BytesIO(requests.get(UNIVERSES[name],headers={"User-Agent":"Mozilla/5.0"},timeout=20).content))
    col=next(c for c in x.columns if c.upper()=="SYMBOL")
    return x[col].dropna().astype(str).str.strip().tolist()

@st.cache_data(ttl=900)
def prices(sym,period,interval):
    import yfinance as yf
    x=yf.download(sym,period=period,interval=interval,auto_adjust=False,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x.dropna()

st.title("📈 Chart Pattern Scanner")
st.caption("Pattern-only scanner • charts appear only after selecting a stock")
with st.sidebar:
    universe=st.selectbox("Universe",list(UNIVERSES))
    timeframe=st.selectbox("Timeframe",list(TF))
    view=st.radio("Direction",["All","Bullish","Bearish","Neutral"])
    scan=st.button("🔎 SCAN PATTERNS",type="primary",use_container_width=True)

if scan:
    rows=[]; syms=symbols(universe); interval,period=TF[timeframe]; bar=st.progress(0)
    for i,sym in enumerate(syms,1):
        try:
            df=prices(sym+".NS",period,interval)
            if len(df)<60: continue
            found,strength,_,_=geometry(df)
            for pat,direction in found:
                if view=="All" or direction==view:
                    rows.append({"Stock":sym,"Timeframe":timeframe,"Pattern":pat,"Direction":direction,"Pattern Strength":strength})
        except Exception: pass
        bar.progress(i/len(syms))
    st.session_state.results=pd.DataFrame(rows); st.session_state.selected=None

if "results" not in st.session_state:
    st.info("Select the universe/timeframe and press SCAN PATTERNS.")
    st.stop()

r=st.session_state.results
if r.empty: st.warning("No patterns found."); st.stop()
st.metric("Detected patterns",len(r))
st.subheader("Scan results")
st.caption("No charts are shown here. Click View to open the stock's targets and chart.")

for i,row in r.sort_values("Pattern Strength",ascending=False).reset_index(drop=True).iterrows():
    a,b,c,d,e,f=st.columns([1.5,1,2.4,1,1,.7])
    a.write(f"**{row.Stock}**"); b.write(row.Timeframe); c.write(row.Pattern); d.write(row.Direction); e.write(f"{row['Pattern Strength']:.1f}")
    if f.button("View",key=f"v{i}"):
        st.session_state.selected=(row.Stock,row.Pattern,row.Timeframe)
        st.rerun()

if st.session_state.get("selected"):
    sel,pat,tf=st.session_state.selected
    df=prices(sel+".NS",*TF[tf]); found,strength,H,L=geometry(df)
    details=pattern_target(df,pat,found[0][1] if found else "Bullish",H,L)
    st.divider(); st.subheader(f"{sel} — {pat}")
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("Current",f"₹{details['current']:,.2f}")
    m2.metric("Entry / Breakout",f"₹{details['entry']:,.2f}")
    m3.metric("Stop / Invalidation",f"₹{details['stop']:,.2f}")
    m4.metric("Target 1",f"₹{details['target1']:,.2f}")
    m5.metric("Target 2",f"₹{details['target2']:,.2f}")
    st.info(f"Target method: **{details['method']}**")
    fig=go.Figure(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="Candles"))
    if H: fig.add_trace(go.Scatter(x=df.index[H],y=df.High.iloc[H],mode="markers",name="Swing High"))
    if L: fig.add_trace(go.Scatter(x=df.index[L],y=df.Low.iloc[L],mode="markers",name="Swing Low"))
    for y,label,dash in [(details["entry"],"Entry / breakout","dot"),(details["target1"],"Target 1","dash"),(details["target2"],"Target 2","dash"),(details["stop"],"Stop / invalidation","dash")]:
        if y is not None: fig.add_hline(y=y,line_dash=dash,annotation_text=label)
    fig.update_layout(height=680,xaxis_rangeslider_visible=False,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)
    if st.button("← Close details"):
        st.session_state.selected=None; st.rerun()
