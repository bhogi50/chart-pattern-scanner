
import io, json, html, requests, pandas as pd, streamlit as st
import streamlit.components.v1 as components
from detector import geometry, pattern_target, pattern_confidence, pattern_status, pattern_points

def major_support_resistance(df, H, L):
    cmp=float(df["Close"].iloc[-1])
    atr=float((df["High"]-df["Low"]).rolling(14).mean().iloc[-1]) if len(df)>=14 else float((df["High"]-df["Low"]).mean())
    tol=max(cmp*0.0075, atr*0.45)
    def best(indices, col):
        vals=sorted(float(df[col].iloc[i]) for i in indices[-30:])
        zones=[]
        for v in vals:
            if not zones or abs(v-zones[-1]["center"])>tol:
                zones.append({"vals":[v],"center":v})
            else:
                zones[-1]["vals"].append(v)
                zones[-1]["center"]=sum(zones[-1]["vals"])/len(zones[-1]["vals"])
        for z in zones:
            z["low"]=min(z["vals"])-.35*tol
            z["high"]=max(z["vals"])+.35*tol
            z["touches"]=len(z["vals"])
            z["strength"]=min(100,30+min(5,z["touches"])*12)
        return max(zones,key=lambda z:(z["strength"],z["touches"])) if zones else None
    return best(L,"Low"), best(H,"High")


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


def render_lightweight_chart(df, pattern, direction, details, H, L, confidence, points=None):
    candles=[]
    for idx,row in df.iterrows():
        try:
            t=int(pd.Timestamp(idx).timestamp())
            candles.append({
                "time":t,
                "open":float(row["Open"]),
                "high":float(row["High"]),
                "low":float(row["Low"]),
                "close":float(row["Close"])
            })
        except Exception:
            pass

    # Swing points are used for the small H/L circle markers on the chart.
    highs=[{"time":int(pd.Timestamp(df.index[i]).timestamp()),"value":float(df["High"].iloc[i])}
           for i in H if i < len(df)]
    lows=[{"time":int(pd.Timestamp(df.index[i]).timestamp()),"value":float(df["Low"].iloc[i])}
          for i in L if i < len(df)]

    def pt(i, col):
        i=int(i)
        return {"time":int(pd.Timestamp(df.index[i]).timestamp()), "value":float(df[col].iloc[i])}

    geometry_lines=[]

    # The pattern-drawing geometry now comes directly from `points`, which is
    # produced by detector.pattern_points() using the *exact same candidate*
    # that pattern_confidence()/pattern_target() scored — so the chart can no
    # longer show a different pair of swings than the ones actually evaluated.
    if points:
        ptype=points.get("type")
        if ptype=="double":
            i1,i2=points["indices"]
            col="High" if pattern=="Double Top" else "Low"
            geometry_lines.append({"type":"zigzag","points":[pt(i1,col),pt(i2,col)],"label":pattern})
            ni=points["neck_idx"]
            geometry_lines.append({"type":"horizontal",
                                    "points":[{"time":pt(ni,"High")["time"],"value":points["neckline"]},
                                              {"time":pt(i2,col)["time"],"value":points["neckline"]}],
                                    "label":"Neckline"})
        elif ptype=="hs":
            l_i,h_i,r_i=points["indices"]
            col="High" if pattern=="Head and Shoulders" else "Low"
            geometry_lines.append({"type":"zigzag","points":[pt(l_i,col),pt(h_i,col),pt(r_i,col)],"label":pattern})
            n1,n2=points["neck_idx"]
            ncol="Low" if pattern=="Head and Shoulders" else "High"
            geometry_lines.append({"type":"trend",
                                    "points":[pt(n1,ncol),pt(n2,ncol)],
                                    "label":"Neckline"})
        elif ptype=="channel":
            hi=points["hi"]; lo=points["lo"]
            if len(hi)>=2:
                geometry_lines.append({"type":"trend","points":[pt(i,"High") for i in hi],"label":"Resistance"})
            if len(lo)>=2:
                geometry_lines.append({"type":"trend","points":[pt(i,"Low") for i in lo],"label":"Support"})
        elif ptype=="flag":
            ps,pe=points["pole_start"],points["pole_end"]
            cs,ce=points["consolidation_start"],points["consolidation_end"]
            pole_col="Low" if direction=="Bullish" else "High"
            geometry_lines.append({"type":"trend","points":[pt(ps,pole_col),pt(pe,pole_col)],"label":"Impulse"})
            cons_col="High" if direction=="Bullish" else "Low"
            geometry_lines.append({"type":"trend","points":[pt(cs,cons_col),pt(ce,cons_col)],"label":"Consolidation"})

    payload={
        "candles":candles,
        "highs":highs,
        "lows":lows,
        "geometry":geometry_lines,
        "levels":{
            "current":details["current"],
            "entry":details["entry"],
            "stop":details["stop"],
            "target1":details["target1"],
            "target2":details["target2"],
            "support":(details.get("support") or {}).get("center"),
            "resistance":(details.get("resistance") or {}).get("center")
        },
        "pattern":pattern,
        "direction":direction,
        "confidence":confidence
    }
    data=json.dumps(payload, separators=(",",":"))

    chart_html="""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
<style>
html,body{margin:0;padding:0;background:#0f131a;color:#d7dce5;font-family:Arial,sans-serif;overflow:hidden}
#wrap{height:720px;width:100%;position:relative}
#chart{height:680px;width:100%}
#legend{height:40px;display:flex;align-items:center;gap:16px;padding:0 12px;font-size:13px;box-sizing:border-box;border-top:1px solid #242a34}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
</style>
</head>
<body>
<div id="wrap">
<div id="chart"></div>
<div id="legend">
<span><b>__PATTERN__</b></span>
<span>Confidence: <b>__CONFIDENCE__</b></span>
<span>__DIRECTION__</span>
</div>
</div>
<script>
const D=__DATA__;
const el=document.getElementById('chart');
const chart=LightweightCharts.createChart(el,{
    autoSize:true,
    layout:{background:{type:'solid',color:'#0f131a'},textColor:'#c9d1d9'},
    grid:{vertLines:{color:'#1d232d'},horzLines:{color:'#1d232d'}},
    crosshair:{mode:LightweightCharts.CrosshairMode.Normal},
    rightPriceScale:{borderColor:'#303744'},
    timeScale:{borderColor:'#303744',timeVisible:true,secondsVisible:false},
    localization:{priceFormatter:p=>p.toFixed(2)}
});
const series=chart.addSeries(LightweightCharts.CandlestickSeries,{
    upColor:'#26a69a',downColor:'#ef5350',
    borderUpColor:'#26a69a',borderDownColor:'#ef5350',
    wickUpColor:'#26a69a',wickDownColor:'#ef5350'
});
series.setData(D.candles);

function addLine(name,color,width,style,price){
    if(price===null || price===undefined || !Number.isFinite(price)) return;
    series.createPriceLine({
        price:price,color:color,lineWidth:width,lineStyle:style,
        axisLabelVisible:true,title:name
    });
}
addLine('Entry', '#4da3ff', 2, LightweightCharts.LineStyle.Dashed, D.levels.entry);
addLine('Stop', '#ef5350', 2, LightweightCharts.LineStyle.Dashed, D.levels.stop);
addLine('Target 1', '#26a69a', 2, LightweightCharts.LineStyle.Dashed, D.levels.target1);
addLine('Target 2', '#9ccc65', 2, LightweightCharts.LineStyle.Dotted, D.levels.target2);
addLine('Major Support', '#42a5f5', 2, LightweightCharts.LineStyle.Solid, D.levels.support);
addLine('Major Resistance', '#ffb74d', 2, LightweightCharts.LineStyle.Solid, D.levels.resistance);

function regression(points){
    if(points.length<2) return null;
    let n=points.length,sx=0,sy=0,sxx=0,sxy=0;
    for(const p of points){sx+=p.time;sy+=p.value;sxx+=p.time*p.time;sxy+=p.time*p.value}
    const den=n*sxx-sx*sx;
    if(!den) return null;
    const m=(n*sxy-sx*sy)/den,b=(sy-m*sx)/n;
    return [
      {time:points[0].time,value:m*points[0].time+b},
      {time:points[points.length-1].time,value:m*points[points.length-1].time+b}
    ];
}

const lineColors=['#f5c542','#8ab4f8','#bb86fc','#ff8a65'];
let colorIndex=0;
for(const g of D.geometry){
    if(g.type==='horizontal'){
        const vals=g.points.map(p=>p.value);
        const avg=vals.reduce((a,b)=>a+b,0)/vals.length;
        const ls=series;
        ls.createPriceLine({price:avg,color:lineColors[colorIndex++%lineColors.length],
            lineWidth:2,lineStyle:LightweightCharts.LineStyle.Solid,axisLabelVisible:true,title:g.label});
    } else if(g.type==='zigzag'){
        if(!g.points || g.points.length<2) continue;
        const ls=chart.addSeries(LightweightCharts.LineSeries,{
            color:lineColors[colorIndex++%lineColors.length],lineWidth:2,
            lineStyle:LightweightCharts.LineStyle.Solid,
            lastValueVisible:false,priceLineVisible:false
        });
        ls.setData(g.points.map(p=>({time:p.time,value:p.value})));
    } else {
        const pts=regression(g.points);
        if(!pts) continue;
        const ls=chart.addSeries(LightweightCharts.LineSeries,{
            color:lineColors[colorIndex++%lineColors.length],lineWidth:2,
            lineStyle:LightweightCharts.LineStyle.Solid,
            lastValueVisible:false,priceLineVisible:false
        });
        ls.setData(pts);
    }
}

const markers=[];
for(const p of D.highs.slice(-5)){
    markers.push({time:p.time,position:'aboveBar',color:'#f5c542',shape:'circle',text:'H'});
}
for(const p of D.lows.slice(-5)){
    markers.push({time:p.time,position:'belowBar',color:'#8ab4f8',shape:'circle',text:'L'});
}
if(D.geometry.some(g=>g.type==='zigzag')){
    const g=D.geometry.find(g=>g.type==='zigzag');
    g.points.forEach((p,i)=>markers.push({
        time:p.time,
        position:(i%2===0?'belowBar':'aboveBar'),
        color:'#f5c542',shape:'circle',text:(i===0?'0':String(i))
    }));
}
markers.sort((a,b)=>a.time-b.time);
if(markers.length) LightweightCharts.createSeriesMarkers(series,markers);
chart.timeScale().fitContent();
</script>
</body>
</html>
"""
    chart_html=chart_html.replace("__DATA__", data).replace("__PATTERN__", html.escape(pattern)).replace("__CONFIDENCE__", f"{confidence:.1f}%").replace("__DIRECTION__", html.escape(direction))
    components.html(chart_html,height=725,scrolling=False)


st.title("📈 Chart Pattern Scanner")
st.caption("Pure pattern detection • one result row per stock • details/chart appear only after View")

with st.sidebar:
    st.header("Scan")
    universe=st.selectbox("Universe",list(UNIVERSES))
    timeframe=st.selectbox("Pattern timeframe",list(CANDLE_OPTIONS))
    candles=st.selectbox("Number of candles",CANDLE_OPTIONS[timeframe],index=CANDLE_OPTIONS[timeframe].index(
        {"Daily":100,"Weekly":52,"Monthly":36}[timeframe]))
    view=st.radio("Direction filter",["All","Bullish","Bearish"])
    min_rr_filter=st.checkbox("Only setups with minimum R/R 1:2",value=True)
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
            found,strength,H,L=geometry(df)
            support,resistance=major_support_resistance(df,H,L)
            selected=[(p,d) for p,d in found if view=="All" or d==view]
            if selected:
                eligible=[]
                for p,d in selected:
                    score=pattern_confidence(df,p,d,H,L)
                    details=pattern_target(df,p,d,H,L)
                    if isinstance(details,dict):
                        details["support"]=support
                        details["resistance"]=resistance
                    rr=None
                    if details and details.get("stop") is not None and details.get("target1") is not None:
                        risk=abs(float(details["entry"])-float(details["stop"]))
                        reward=abs(float(details["target1"])-float(details["entry"]))
                        rr=(reward/risk) if risk>0 else None
                    if (not min_rr_filter) or (rr is not None and rr >= 2.0):
                        status=pattern_status(df,p,d,H,L)
                        eligible.append({"pattern":p,"status":status,"bias":d,"confidence":score,"rr":rr})

                if eligible:
                    dirs=sorted(set(x["bias"] for x in eligible))
                    rows.append({
                        "Stock":sym, "Timeframe":timeframe, "Candles":len(df),
                        "Pattern":", ".join(x["pattern"] for x in eligible),
                        "Status":", ".join(x["status"] for x in eligible),
                        "Bias":", ".join(x["bias"] for x in eligible),
                        "Confidence":", ".join(f'{x["confidence"]:.1f}%' for x in eligible),
                        "Pattern Confidence":max(x["confidence"] for x in eligible),
                        "Best R/R to T1":max((x["rr"] for x in eligible if x["rr"] is not None), default=None)
                    })
        except Exception:
            continue
        bar.progress(i/max(1,len(syms)))
    result_df=pd.DataFrame(rows)
    if not result_df.empty:
        result_df=result_df.sort_values(
            by=["Pattern Confidence", "Stock"],
            ascending=[False, True],
            kind="stable"
        ).reset_index(drop=True)
    st.session_state.results=result_df
    st.session_state.scan_config=(universe,timeframe,candles,view,min_rr_filter)
    st.session_state.selected=None

if "results" not in st.session_state:
    st.info("Choose the scan settings and press SCAN PATTERNS.")
    st.stop()

r=st.session_state.results
if r.empty:
    st.warning("No matching patterns found for the selected settings.")
    st.stop()

st.subheader("Detected patterns")
st.caption("One row per stock. Multiple detected patterns are combined in the same row. With the 1:2 filter enabled, only setups with R/R to Target 1 of at least 1:2 are shown.")

show=r[["Stock","Timeframe","Candles","Pattern","Status","Bias","Confidence"]].copy()
# Sort by the best detected pattern confidence without exposing the helper column.
if "Pattern Confidence" in r.columns:
    _confidence_order = pd.to_numeric(r["Pattern Confidence"], errors="coerce")
    show["_confidence_order"] = _confidence_order.to_numpy()
    show=show.sort_values(by=["_confidence_order","Stock"], ascending=[False,True], kind="stable")
    show=show.drop(columns=["_confidence_order"]).reset_index(drop=True)
else:
    show=show.sort_values(by=["Stock"], ascending=[True], kind="stable").reset_index(drop=True)

# Keep confidence numeric while sorting so the displayed table is truly
# ordered from strongest pattern-quality score to weakest.

st.dataframe(
    show,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Stock": st.column_config.TextColumn("Stock", width="small"),
        "Timeframe": st.column_config.TextColumn("Timeframe", width="small"),
        "Candles": st.column_config.NumberColumn("Candles", width="small"),
        "Pattern": st.column_config.TextColumn("Pattern", width="large"),
        "Status": st.column_config.TextColumn("Status", width="medium"),
        "Bias": st.column_config.TextColumn("Bias", width="medium"),
        "Confidence": st.column_config.TextColumn("Confidence", width="medium"),
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
    available=[(p,d) for p,d in found if p in [x.strip() for x in str(row["Pattern"]).split(",")]]
    if not available:
        st.warning("The pattern is no longer detected on the latest data. Please scan again.")
        st.stop()

    st.divider()
    st.subheader(f"{sel} — Pattern Details")
    st.write("**Patterns detected:** "+", ".join(p for p,_ in available))

    labels=[p for p,_ in available]
    pat=st.selectbox("Pattern to inspect",labels)
    direction=dict(available)[pat]
    confidence=pattern_confidence(df,pat,direction,H,L)
    status=pattern_status(df,pat,direction,H,L)
    details=pattern_target(df,pat,direction,H,L)
    support,resistance=major_support_resistance(df,H,L)
    if isinstance(details,dict):
        details["support"]=support
        details["resistance"]=resistance
    if details is None:
        st.error("Not enough data to calculate targets.")
        st.stop()

    def money(x):
        return "—" if x is None else f"₹{float(x):,.2f}"

    risk=abs(details["entry"]-details["stop"]) if details["stop"] is not None else 0.0
    reward=abs(details["target1"]-details["entry"])
    rr=reward/risk if risk else None

    levels = pd.DataFrame([
        ["Entry / Breakout", money(details["entry"])],
        ["CMP", money(details["current"])],
        ["Target 1", money(details["target1"])],
        ["Target 2", money(details["target2"])],
        ["Stop loss", money(details["stop"])],
        ["R:R", (f"1:{rr:.2f}" if rr is not None else "—")],
    ], columns=["Level", "Value"])
    st.table(levels)
    st.info(
        f"Pattern: **{pat}** · Status: **{status}** · Bias: **{direction}** · "
        f"Pattern confidence: **{confidence:.1f}%** · "
        f"Target method: **{details['method']}**" +
        (f" · R/R to T1: **1:{rr:.2f}**" if rr else "")
    )
    st.caption("Confidence measures pattern quality; R/R is a reward-to-risk ratio, not a probability of hitting the target.")

    vc = details.get("volume_confirmed")
    vr = details.get("volume_ratio")
    if vc is None:
        st.caption("Volume confirmation: not available for this data.")
    else:
        vtxt = "confirmed" if vc else "not confirmed"
        st.caption(f"Volume confirmation: **{vtxt}** (latest candle at {vr:.2f}× its 20-period average).")

    st.caption(
        f"Exact levels — Current: {money(details['current'])} · "
        f"Entry: {money(details['entry'])} · Stop: {money(details['stop'])} · "
        f"T1: {money(details['target1'])} · T2: {money(details['target2'])}"
    )

    chart_points = pattern_points(df, pat, direction, H, L)
    render_lightweight_chart(df, pat, direction, details, H, L, confidence, chart_points)
    if st.button("← Close details"):
        st.session_state.selected=None
        st.rerun()
