import pandas as pd
import streamlit as st
from data import fetch_ohlc
from patterns import analyze_chart, detect_all_patterns

DIRECTION_FILTER_MAP={"All":{"Bullish","Bearish","Neutral"},"Bullish":{"Bullish"},"Bearish":{"Bearish"},"Neutral":{"Neutral"}}
MIN_CONFIDENCE=40.0
ALL_TIMEFRAMES=["Daily","Weekly","Monthly"]


def _higher_tf(primary):
    return {"Daily":"Weekly","Weekly":"Monthly","Monthly":"Weekly"}[primary]


def _tf_context(ticker,timeframe,num_candles):
    df=fetch_ohlc(ticker,timeframe,num_candles)
    if df.empty or len(df)<20: return {}
    a=analyze_chart(df)
    s=a["structure"]
    pats=[p for p in a["patterns"] if p.direction in {"Bullish","Bearish"} and p.confidence>=MIN_CONFIDENCE]
    direction=pats[0].direction if pats else None
    return {"timeframe":timeframe,"structure":s.name,"structure_confidence":s.confidence,"direction":direction,
            "location":a["location"].zone,"supports":a["supports"],"resistances":a["resistances"]}


def compute_confluence(ticker,primary_timeframe,primary_direction,num_candles):
    agree=1; total=1; details=[f"{primary_timeframe}: {primary_direction}"]
    for tf in ALL_TIMEFRAMES:
        if tf==primary_timeframe: continue
        c=_tf_context(ticker,tf,num_candles)
        total+=1
        if c.get("direction"):
            details.append(f"{tf}: {c['direction']} ({c['structure']})")
            if c["direction"]==primary_direction: agree+=1
        else: details.append(f"{tf}: {c.get('structure','Unclear')} / no directional pattern")
    return f"{agree}/{total}","; ".join(details)


def run_scan(constituents,timeframe,num_candles,pattern_filter,progress_callback=None,check_confluence=True):
    allowed=DIRECTION_FILTER_MAP.get(pattern_filter,DIRECTION_FILTER_MAP["All"])
    rows=[]; total=len(constituents)
    for i,row in enumerate(constituents.itertuples(index=False)):
        if progress_callback: progress_callback(i+1,total,row.Symbol,stage="scanning")
        df=fetch_ohlc(row.Ticker,timeframe,num_candles)
        if df.empty or len(df)<20: continue
        htf=_tf_context(row.Ticker,_higher_tf(timeframe),num_candles)
        analysis=analyze_chart(df,higher_tf=htf)
        for m in analysis["patterns"]:
            if m.direction not in allowed or m.confidence<MIN_CONFIDENCE: continue
            rows.append({"Symbol":row.Symbol,"Company":row.CompanyName,"Pattern":m.name,"Direction":m.direction,
                         "Confidence":m.confidence,"Status":m.status,"LastPrice":round(float(df.Close.iloc[-1]),2),
                         "PatternDate":df.index[-1].strftime("%Y-%m-%d"),"Note":m.note,"Ticker":row.Ticker,
                         "Structure":analysis["structure"].name,"Location":analysis["location"].zone,
                         "StructureConfidence":analysis["structure"].confidence,
                         "R1":analysis["resistances"][0]["price"] if analysis["resistances"] else None,
                         "S1":analysis["supports"][0]["price"] if analysis["supports"] else None})
    cols=["Symbol","Company","Pattern","Direction","Confidence","Status","LastPrice","PatternDate","Note","Confluence","ConfluenceDetail","Ticker","Structure","Location","StructureConfidence","R1","S1"]
    if not rows: return pd.DataFrame(columns=cols)
    out=pd.DataFrame(rows).sort_values(["Confidence","StructureConfidence"],ascending=False).reset_index(drop=True)
    if check_confluence:
        cv=[];dv=[]
        for j,r in enumerate(out.itertuples(index=False)):
            if progress_callback: progress_callback(j+1,len(out),r.Symbol,stage="confluence")
            c,d=compute_confluence(r.Ticker,timeframe,r.Direction,num_candles);cv.append(c);dv.append(d)
        out["Confluence"]=cv;out["ConfluenceDetail"]=dv
    else:
        out["Confluence"]="—";out["ConfluenceDetail"]=""
    return out
