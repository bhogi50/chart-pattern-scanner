
"""
Pattern detector built around PatternPy as the candidate-generation engine.

PatternPy is used for OHLC pattern recognition (tops/bottoms, H&S and
channels). Our validation layer converts those raw candidates into the
scanner's stricter six requested families plus flags, then calculates
status, confidence, geometry and targets.

PatternPy: https://github.com/keithorange/PatternPy
License: CC BY-NC-SA 4.0 (non-commercial/share-alike).
"""

import numpy as np
import pandas as pd

from tradingpatterns.tradingpatterns import (
    detect_head_shoulder,
    detect_multiple_tops_bottoms,
    detect_channel,
    detect_double_top_bottom,
)

ALLOWED_PATTERNS = (
    "Double Top",
    "Double Bottom",
    "Head and Shoulders",
    "Inverse Head and Shoulders",
    "Bullish Flag",
    "Bearish Flag",
    "Rising Channel",
    "Falling Channel",
)

def _clamp(x, lo=0.0, hi=100.0):
    return float(max(lo, min(hi, x)))

def _atr(df, period=14):
    if len(df) < 2:
        return float((df["High"] - df["Low"]).mean()) if len(df) else 0.0
    h=df["High"].to_numpy(float)
    l=df["Low"].to_numpy(float)
    c=df["Close"].to_numpy(float)
    tr=np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    return float(np.nanmean(tr[-min(period, len(tr)):])) if len(tr) else 0.0

def pivots(df, left=3, right=3):
    if df is None or len(df) < left + right + 3:
        return [], []
    h=df["High"].to_numpy(float)
    l=df["Low"].to_numpy(float)
    H,L=[],[]
    for i in range(left, len(df)-right):
        if h[i] >= np.max(h[i-left:i]) and h[i] >= np.max(h[i+1:i+right+1]):
            H.append(i)
        if l[i] <= np.min(l[i-left:i]) and l[i] <= np.min(l[i+1:i+right+1]):
            L.append(i)
    return H,L

def _near(a,b,t=0.04):
    return abs(a-b)/max(abs(a),abs(b),1e-9) <= t

def _slope(y,x):
    return float(np.polyfit(np.asarray(x,float),np.asarray(y,float),1)[0]) if len(x)>=2 else 0.0

def _r2(values, indices):
    if len(indices)<2:
        return 0.0
    x=np.asarray(indices,float); y=np.asarray(values,float)
    if np.std(x)==0 or np.std(y)==0:
        return 1.0
    r=np.corrcoef(x,y)[0,1]
    return float(r*r) if np.isfinite(r) else 0.0

def _copy_for_patternpy(df):
    x=df.copy()
    # PatternPy mutates its input and expects these OHLC column names.
    return x[["Open","High","Low","Close"] + (["Volume"] if "Volume" in x.columns else [])].copy()

def _patternpy_frame(df, kind):
    x=_copy_for_patternpy(df)
    try:
        if kind=="hs":
            return detect_head_shoulder(x, window=3)
        if kind=="multi":
            return detect_multiple_tops_bottoms(x, window=3)
        if kind=="channel":
            return detect_channel(x, window=3)
        if kind=="double":
            return detect_double_top_bottom(x, window=3, threshold=0.05)
    except Exception:
        return None
    return None

def _patternpy_indices(df, column, label):
    x=_patternpy_frame(df, {"hs":"hs","multi":"multi","channel":"channel","double":"double"}[
        {"head_shoulder_pattern":"hs","multiple_top_bottom_pattern":"multi",
         "channel_pattern":"channel","double_pattern":"double"}[column]
    ])
    if x is None or column not in x.columns:
        return []
    vals=x[column].astype(str)
    return [i for i,v in enumerate(vals) if v==label]

def _double_candidates(df, side, H, L):
    # PatternPy supplies local top/bottom candidates. We then pair them and
    # apply the scanner's structural validation so a single local pivot is
    # never mistaken for a double.
    col="double_pattern"
    label="Double Top" if side=="top" else "Double Bottom"
    pp=_patternpy_indices(df,col,label)
    pivot=list(H if side=="top" else L)
    candidates=sorted(set(pp + pivot))
    if len(candidates)<2:
        return []

    vals=df["High" if side=="top" else "Low"].to_numpy(float)
    atr=max(_atr(df),1e-9)
    out=[]
    for a_pos in range(max(0,len(candidates)-12),len(candidates)-1):
        for b_pos in range(a_pos+1,len(candidates)):
            a,b=candidates[a_pos],candidates[b_pos]
            gap=b-a
            if gap<5 or gap>90:
                continue
            va,vb=vals[a],vals[b]
            tol=max(0.012, min(0.055, 1.25*atr/max((va+vb)/2,1e-9)))
            if abs(va-vb)/max((va+vb)/2,1e-9)>tol:
                continue

            mid_vals=df["Low"].to_numpy(float)[a:b+1] if side=="top" else df["High"].to_numpy(float)[a:b+1]
            neckline=float(np.min(mid_vals) if side=="top" else np.max(mid_vals))
            reaction=abs(neckline-(va+vb)/2)
            if reaction < max(1.0*atr, 0.012*abs((va+vb)/2)):
                continue

            # The two extrema must be the dominant opposite-side reactions
            # between the candidate and neckline.
            out.append({
                "first":a, "second":b,
                "first_value":va, "second_value":vb,
                "neckline":neckline,
                "height":reaction,
                "tolerance":tol,
                "patternpy": a in pp or b in pp,
            })
    return out

def _best_double(df, side, H, L):
    cs=_double_candidates(df,side,H,L)
    if not cs:
        return None
    def score(c):
        mid=(c["first_value"]+c["second_value"])/2
        similarity=abs(c["first_value"]-c["second_value"])/max(abs(mid),1e-9)
        sim=_clamp(100*(1-similarity/max(c["tolerance"],1e-9)))
        reaction=_clamp(c["height"]/max(1.5*_atr(df),0.02*abs(mid))*100)
        spacing=_clamp(c["second"]-c["first"],5,45)
        return .50*sim+.35*reaction+.15*(spacing/45*100)+(5 if c["patternpy"] else 0)
    return max(cs,key=score)

def _hs_candidates(df, inverse, H, L):
    # PatternPy gives the H&S candidate trigger; pivot geometry supplies the
    # exact shoulders/head/neckline used by our scanner.
    x=_patternpy_frame(df,"hs")
    col="head_shoulder_pattern"
    label="Inverse Head and Shoulder" if inverse else "Head and Shoulder"
    pp=[] if x is None or col not in x.columns else [i for i,v in enumerate(x[col].astype(str)) if v==label]
    extrema=L if inverse else H
    opposite=H if inverse else L
    if len(extrema)<3 or len(opposite)<2:
        return []

    vals=df["Low" if inverse else "High"].to_numpy(float)
    atr=max(_atr(df),1e-9)
    out=[]
    for a,b,c in zip(extrema[-8:-2],extrema[-7:-1],extrema[-6:]):
        lv, hv, rv=vals[a], vals[b], vals[c]
        shoulder_tol=max(.02,min(.08,1.4*atr/max((abs(lv)+abs(rv))/2,1e-9)))
        if abs(lv-rv)/max((abs(lv)+abs(rv))/2,1e-9)>shoulder_tol:
            continue
        if inverse:
            prominence=min(lv,rv)-hv
        else:
            prominence=hv-max(lv,rv)
        if prominence < max(1.0*atr, .015*abs(hv)):
            continue

        between1=[i for i in opposite if a<i<b]
        between2=[i for i in opposite if b<i<c]
        if not between1 or not between2:
            continue
        n1=max(df["High"].iloc[between1]) if inverse else min(df["Low"].iloc[between1])
        n2=max(df["High"].iloc[between2]) if inverse else min(df["Low"].iloc[between2])
        out.append({
            "left":a,"head":b,"right":c,
            "left_value":lv,"head_value":hv,"right_value":rv,
            "neck1":float(n1),"neck2":float(n2),
            "prominence":float(prominence),
            "tolerance":shoulder_tol,
            "patternpy": bool(pp),
        })
    return out

def _best_hs(df,inverse,H,L):
    cs=_hs_candidates(df,inverse,H,L)
    if not cs:
        return None
    def score(c):
        mid=(c["left_value"]+c["right_value"])/2
        sym=abs(c["left_value"]-c["right_value"])/max(abs(mid),1e-9)
        sym_score=_clamp(100*(1-sym/max(c["tolerance"],1e-9)))
        prom=_clamp(c["prominence"]/max(1.5*_atr(df),.03*abs(c["head_value"]))*100)
        return .50*sym_score+.40*prom+.10*(5 if c["patternpy"] else 0)
    return max(cs,key=score)

def _channel_candidate(df, kind, H, L):
    # PatternPy identifies the channel direction; our validator then rejects
    # channels with weak fits, poor containment or recent decisive breaks.
    x=_patternpy_frame(df,"channel")
    col="channel_pattern"
    label="Channel Up" if kind=="Rising Channel" else "Channel Down"
    pp=bool(x is not None and col in x.columns and str(x[col].iloc[-1])==label)
    if len(H)<3 or len(L)<3:
        return None
    hi=H[-6:]; lo=L[-6:]
    hv=df["High"].to_numpy(float)[hi]
    lv=df["Low"].to_numpy(float)[lo]
    hs=_slope(hv,hi); ls=_slope(lv,lo)
    if kind=="Rising Channel" and not (hs>0 and ls>0):
        return None
    if kind=="Falling Channel" and not (hs<0 and ls<0):
        return None

    hs_abs=max(abs(hs),1e-9); ls_abs=max(abs(ls),1e-9)
    slope_ratio=min(hs_abs,ls_abs)/max(hs_abs,ls_abs)
    if slope_ratio<0.45:
        return None

    hi_fit=_r2(hv,hi); lo_fit=_r2(lv,lo)
    if min(hi_fit,lo_fit)<0.55:
        return None

    xall=np.arange(len(df),dtype=float)
    hcoef=np.polyfit(hi,hv,1); lcoef=np.polyfit(lo,lv,1)
    upper=np.polyval(hcoef,xall); lower=np.polyval(lcoef,xall)
    width=upper-lower
    width_med=float(np.nanmedian(np.abs(width[-min(40,len(width)):])) if len(width) else 0)
    width_cv=float(np.nanstd(width[-min(40,len(width)):])/max(width_med,1e-9))
    if width_cv>0.45:
        return None

    c=df["Close"].to_numpy(float)
    atr=max(_atr(df),1e-9)
    start=max(0,len(df)-min(50,max(20,len(df)//2)))
    inside=(c[start:]<=upper[start:]+1.1*atr)&(c[start:]>=lower[start:]-1.1*atr)
    containment=float(np.mean(inside)) if len(inside) else 0
    if containment<0.62:
        return None

    recent_break=False
    for i in range(max(start,len(df)-5),len(df)):
        if kind=="Rising Channel" and c[i]<lower[i]-1.25*atr:
            recent_break=True
        if kind=="Falling Channel" and c[i]>upper[i]+1.25*atr:
            recent_break=True

    return {"hi":hi,"lo":lo,"upper":upper,"lower":lower,
            "slope_ratio":slope_ratio,"hi_fit":hi_fit,"lo_fit":lo_fit,
            "width_cv":width_cv,"containment":containment,
            "recent_break":recent_break,"patternpy":pp}

def _flag_candidate(df, bullish):
    if len(df)<25:
        return None
    c=df["Close"].to_numpy(float); h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float)
    ps,pe=0,14
    cs,ce=15,len(df)-1
    impulse=(c[pe]-c[ps])/max(abs(c[ps]),1e-9)
    if bullish and impulse<.10: return None
    if not bullish and impulse>-.10: return None
    cons_range=(max(h[cs:ce+1])-min(l[cs:ce+1]))/max(abs(np.mean(c[cs:ce+1])),1e-9)
    if cons_range>.12: return None
    return {"pole_start":ps,"pole_end":pe,"consolidation_start":cs,"consolidation_end":ce}

def geometry(df):
    H,L=pivots(df)
    found=[]

    db=_best_double(df,"bottom",H,L)
    if db: found.append(("Double Bottom","Bullish"))
    dt=_best_double(df,"top",H,L)
    if dt: found.append(("Double Top","Bearish"))

    hs=_best_hs(df,False,H,L)
    if hs: found.append(("Head and Shoulders","Bearish"))
    ihs=_best_hs(df,True,H,L)
    if ihs: found.append(("Inverse Head and Shoulders","Bullish"))

    rc=_channel_candidate(df,"Rising Channel",H,L)
    if rc and not rc["recent_break"]: found.append(("Rising Channel","Bullish"))
    fc=_channel_candidate(df,"Falling Channel",H,L)
    if fc and not fc["recent_break"]: found.append(("Falling Channel","Bearish"))

    if _flag_candidate(df,True): found.append(("Bullish Flag","Bullish"))
    if _flag_candidate(df,False): found.append(("Bearish Flag","Bearish"))

    found=list(dict.fromkeys((p,d) for p,d in found if p in ALLOWED_PATTERNS))
    scores=[pattern_confidence(df,p,d,H,L) for p,d in found]
    return found, round(max(scores),1) if scores else 0.0, H, L

def _candidate(df,pattern,H,L):
    if pattern=="Double Bottom": return _best_double(df,"bottom",H,L)
    if pattern=="Double Top": return _best_double(df,"top",H,L)
    if pattern=="Head and Shoulders": return _best_hs(df,False,H,L)
    if pattern=="Inverse Head and Shoulders": return _best_hs(df,True,H,L)
    if pattern=="Rising Channel": return _channel_candidate(df,"Rising Channel",H,L)
    if pattern=="Falling Channel": return _channel_candidate(df,"Falling Channel",H,L)
    if pattern=="Bullish Flag": return _flag_candidate(df,True)
    if pattern=="Bearish Flag": return _flag_candidate(df,False)
    return None

def pattern_confidence(df,pattern,direction,H=None,L=None):
    if pattern not in ALLOWED_PATTERNS: return 0.0
    H,L=pivots(df) if H is None or L is None else (H,L)
    cand=_candidate(df,pattern,H,L)
    if not cand: return 0.0

    atr=max(_atr(df),1e-9)
    bonus=5 if cand.get("patternpy") else 0

    if pattern in ("Double Bottom","Double Top"):
        mid=(cand["first_value"]+cand["second_value"])/2
        sim=1-abs(cand["first_value"]-cand["second_value"])/max(abs(mid),1e-9)/max(cand["tolerance"],1e-9)
        reaction=_clamp(cand["height"]/max(1.5*atr,.02*abs(mid))*100)
        return round(_clamp(.55*_clamp(sim*100)+.35*reaction+.10*_clamp(cand["second"]-cand["first"],5,45)/45*100+bonus),1)

    if pattern in ("Head and Shoulders","Inverse Head and Shoulders"):
        mid=(cand["left_value"]+cand["right_value"])/2
        sim=1-abs(cand["left_value"]-cand["right_value"])/max(abs(mid),1e-9)/max(cand["tolerance"],1e-9)
        prom=_clamp(cand["prominence"]/max(1.5*atr,.03*abs(cand["head_value"]))*100)
        return round(_clamp(.55*_clamp(sim*100)+.40*prom+bonus),1)

    if pattern in ("Rising Channel","Falling Channel"):
        fit=(cand["hi_fit"]+cand["lo_fit"])/2*100
        parallel=cand["slope_ratio"]*100
        containment=cand["containment"]*100
        return round(_clamp(.40*fit+.30*parallel+.30*containment+bonus),1)

    # Flag confidence: impulse strength + compact consolidation.
    pole_strength=_clamp(abs((df["Close"].iloc[14]-df["Close"].iloc[0])/max(abs(df["Close"].iloc[0]),1e-9))*500)
    return round(_clamp(.55*pole_strength+.45*70+bonus),1)

def pattern_status(df,pattern,direction,H=None,L=None):
    details=pattern_target(df,pattern,direction,H,L)
    if not details: return "Forming"
    c=float(df["Close"].iloc[-1]); e=float(details["entry"])
    if pattern in ("Double Bottom","Inverse Head and Shoulders","Bullish Flag"):
        return "Formed/Active" if c>=e else "Forming"
    if pattern in ("Double Top","Head and Shoulders","Bearish Flag"):
        return "Formed/Active" if c<=e else "Forming"
    cand=_candidate(df,pattern,H,L)
    if cand and cand.get("recent_break"): return "Forming"
    return "Formed/Active"

def pattern_target(df,pattern,direction,H=None,L=None):
    if pattern not in ALLOWED_PATTERNS or len(df)==0: return None
    H,L=pivots(df) if H is None or L is None else (H,L)
    cand=_candidate(df,pattern,H,L)
    if not cand: return None
    c=float(df["Close"].iloc[-1])
    entry=c; stop=None; t1=None; t2=None; method="PatternPy candidate + structural projection"

    if pattern=="Double Bottom":
        n=cand["neckline"]; height=cand["height"]; entry=n; stop=min(cand["first_value"],cand["second_value"]); t1=n+height; t2=n+1.618*height
    elif pattern=="Double Top":
        n=cand["neckline"]; height=cand["height"]; entry=n; stop=max(cand["first_value"],cand["second_value"]); t1=n-height; t2=n-1.618*height
    elif pattern in ("Head and Shoulders","Inverse Head and Shoulders"):
        n=(cand["neck1"]+cand["neck2"])/2; height=cand["prominence"]; entry=n
        if pattern=="Inverse Head and Shoulders":
            stop=cand["head_value"]; t1=n+height; t2=n+1.618*height
        else:
            stop=cand["head_value"]; t1=n-height; t2=n-1.618*height
    elif pattern in ("Rising Channel","Falling Channel"):
        sup=float(np.min(df["Low"].iloc[cand["lo"]]))
        res=float(np.max(df["High"].iloc[cand["hi"]]))
        if direction=="Bullish":
            entry=c; stop=sup; t1=res; t2=res+(res-sup)*.618
        else:
            entry=c; stop=res; t1=sup; t2=sup-(res-sup)*.618
    else:
        fc=cand
        pole=float(max(df["High"].iloc[fc["pole_start"]:fc["pole_end"]+1])-min(df["Low"].iloc[fc["pole_start"]:fc["pole_end"]+1]))
        if direction=="Bullish":
            entry=c; stop=float(min(df["Low"].iloc[fc["consolidation_start"]:fc["consolidation_end"]+1])); t1=c+pole; t2=c+1.618*pole
        else:
            entry=c; stop=float(max(df["High"].iloc[fc["consolidation_start"]:fc["consolidation_end"]+1])); t1=c-pole; t2=c-1.618*pole

    return {"current":c,"entry":float(entry),"stop":float(stop),
            "target1":float(t1),"target2":float(t2),"method":method}

def pattern_points(df,pattern,direction,H=None,L=None):
    H,L=pivots(df) if H is None or L is None else (H,L)
    cand=_candidate(df,pattern,H,L)
    if not cand: return None
    if pattern in ("Double Top","Double Bottom"):
        return {"type":"double","indices":[cand["first"],cand["second"]],
                "neck_idx":cand["second"],"neckline":cand["neckline"]}
    if pattern in ("Head and Shoulders","Inverse Head and Shoulders"):
        return {"type":"hs","indices":[cand["left"],cand["head"],cand["right"]],
                "neck_idx":[cand["left"],cand["right"]],
                "neckline":(cand["neck1"]+cand["neck2"])/2}
    if pattern in ("Rising Channel","Falling Channel"):
        return {"type":"channel","hi":cand["hi"],"lo":cand["lo"]}
    return {"type":"flag","pole_start":cand["pole_start"],"pole_end":cand["pole_end"],
            "consolidation_start":cand["consolidation_start"],
            "consolidation_end":cand["consolidation_end"]}
