import numpy as np

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


def _clamp(x,lo=0,hi=100):
    return float(max(lo,min(hi,x)))


def _r2(values, indices):
    if len(indices) < 2:
        return 0.0
    x=np.asarray(indices,float); y=np.asarray(values,float)
    if np.std(x)==0 or np.std(y)==0:
        return 1.0 if np.std(y)==0 else 0.0
    r=np.corrcoef(x,y)[0,1]
    return float(r*r) if np.isfinite(r) else 0.0


def _line_fit_score(values,indices):
    if len(indices)<2:
        return 0.0
    y=np.asarray(values,float); x=np.asarray(indices,float)
    coef=np.polyfit(x,y,1)
    pred=coef[0]*x+coef[1]
    err=np.mean(np.abs(y-pred))/max(np.mean(np.abs(y)),1e-9)
    return _clamp(100*(1-err*12))


def _atr(df, period=14):
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float); c=df["Close"].to_numpy(float)
    if len(df)<2:
        return float(np.mean(h-l)) if len(df) else 0.0
    tr=np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    return float(np.mean(tr[-min(period,len(tr)):]))


def _double_candidate(df, side, piv):
    """Return the strongest structurally valid double-top/bottom candidate."""
    if len(piv) < 2:
        return None
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float)
    c=df["Close"].to_numpy(float); atr=max(_atr(df),1e-9)
    candidates=[]
    # Do not restrict to the last two pivots: a minor pivot can sit between the
    # two meaningful bottoms/tops. Search a realistic recent window.
    recent=piv[-8:]
    for ai in range(len(recent)-2, -1, -1):
        for bi in range(ai+1, len(recent)):
            a,b=recent[ai],recent[bi]
            gap=b-a
            if gap < 5 or gap > min(100, max(20, len(df)-5)):
                continue
            if side=="bottom":
                va,vb=l[a],l[b]
                vals=l[a:b+1]
                neck_idx=a+int(np.argmax(h[a:b+1]))
                neck=float(h[neck_idx])
                extreme=float(min(va,vb))
                rise=neck-extreme
                similarity=abs(va-vb)/max((va+vb)/2,1e-9)
            else:
                va,vb=h[a],h[b]
                neck_idx=a+int(np.argmin(l[a:b+1]))
                neck=float(l[neck_idx])
                extreme=float(max(va,vb))
                rise=extreme-neck
                similarity=abs(va-vb)/max((va+vb)/2,1e-9)
            # Volatility-aware similarity: allow a little more room for volatile
            # stocks, but never let two clearly different levels qualify.
            tol=min(0.045, max(0.012, 0.55*atr/max(abs((va+vb)/2),1e-9)))
            if similarity > tol:
                continue
            # The middle reaction must be meaningful, not a flat V/noise move.
            min_reaction=max(1.6*atr, 0.025*max(abs((va+vb)/2),1e-9))
            if rise < min_reaction:
                continue
            # Neckline must occur between the two extremes and leave a real swing.
            if not (a < neck_idx < b):
                continue
            # Prefer candidates with a clean separation and a recent second leg.
            recency=max(0.0, 1.0-(len(df)-b)/max(len(df),1))
            spacing=min(1.0, gap/25.0)
            sim_score=max(0.0,1.0-similarity/max(tol,1e-9))
            reaction_score=min(1.0,rise/max(4*atr,0.05*abs(neck)))
            score=0.45*sim_score+0.25*reaction_score+0.15*spacing+0.15*recency
            candidates.append((score,{"first":a,"second":b,"neck_idx":neck_idx,
                                      "first_value":float(va),"second_value":float(vb),
                                      "neckline":neck,"height":rise,"tolerance":tol}))
    if not candidates:
        return None
    return max(candidates,key=lambda x:x[0])[1]


def _channel_candidate(df,H,L,kind):
    if len(H)<3 or len(L)<3:
        return None
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float); c=df["Close"].to_numpy(float)
    hi=H[-7:]; lo=L[-7:]
    hs=_slope(h[hi],hi); ls=_slope(l[lo],lo)
    scale=max(float(np.mean(c[-min(50,len(c)): ])),1e-9)
    # Normalize slopes by price so a ₹1 move in a ₹100 stock is not treated
    # the same as a ₹1 move in a ₹10,000 stock.
    nhs=hs/scale; nls=ls/scale
    if kind=="Rising Channel" and not (nhs>0 and nls>0):
        return None
    if kind=="Falling Channel" and not (nhs<0 and nls<0):
        return None
    slope_ratio=min(abs(nhs),abs(nls))/max(abs(nhs),abs(nls),1e-12)
    if slope_ratio < 0.55:
        return None

    hi_fit=_r2(h[hi],hi); lo_fit=_r2(l[lo],lo)
    if hi_fit < 0.55 or lo_fit < 0.55:
        return None

    # Measure the channel width at each pivot and require it to be reasonably stable.
    mh,bh=np.polyfit(np.asarray(hi,float),h[hi],1)
    ml,bl=np.polyfit(np.asarray(lo,float),l[lo],1)
    xs=np.array(sorted(set(hi+lo)),float)
    widths=(mh*xs+bh)-(ml*xs+bl)
    if np.any(widths <= 0):
        return None
    width_cv=float(np.std(widths)/max(np.mean(widths),1e-9))
    if width_cv > 0.65:
        return None

    # Count actual price interactions with each fitted boundary.
    atr=max(_atr(df),1e-9)
    upper=mh*np.arange(len(df))+bh
    lower=ml*np.arange(len(df))+bl
    hi_touch=sum(abs(h[i]-upper[i]) <= max(0.9*atr,0.006*c[i]) for i in hi)
    lo_touch=sum(abs(l[i]-lower[i]) <= max(0.9*atr,0.006*c[i]) for i in lo)
    if hi_touch < 3 or lo_touch < 3:
        return None

    # Containment: most recent candles should lie between the boundaries.
    start=max(0,len(df)-min(50,max(20,len(df)//2)))
    inside=np.logical_and(c[start:] <= upper[start:]+1.1*atr, c[start:] >= lower[start:]-1.1*atr)
    containment=float(np.mean(inside)) if len(inside) else 0.0
    if containment < 0.62:
        return None

    # A decisive recent break invalidates an active channel. We still return the
    # geometry so callers can use the candidate, but mark it as not active later.
    recent_break=False
    for i in range(max(start, len(df)-5),len(df)):
        if kind=="Rising Channel" and c[i] < lower[i]-1.25*atr:
            recent_break=True
        if kind=="Falling Channel" and c[i] > upper[i]+1.25*atr:
            recent_break=True

    return {"hi":hi,"lo":lo,"upper":upper,"lower":lower,"slope_ratio":slope_ratio,
            "hi_fit":hi_fit,"lo_fit":lo_fit,"width_cv":width_cv,
            "hi_touch":hi_touch,"lo_touch":lo_touch,"containment":containment,
            "recent_break":recent_break}


def pattern_confidence(df,pattern,direction,H=None,L=None):
    if pattern not in ALLOWED_PATTERNS or df is None or len(df)==0:
        return 0.0
    if H is None or L is None:
        H,L=pivots(df)
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float)
    quality=50.0

    if pattern in ("Double Top","Double Bottom"):
        cand=_double_candidate(df,"top" if pattern=="Double Top" else "bottom",H if pattern=="Double Top" else L)
        if not cand:
            return 0.0
        mid=(cand["first_value"]+cand["second_value"])/2
        similarity=abs(cand["first_value"]-cand["second_value"])/max(abs(mid),1e-9)
        sim_score=_clamp(100*(1-similarity/max(cand["tolerance"],1e-9)))
        reaction_score=_clamp(cand["height"]/max(4*_atr(df),0.05*abs(cand["neckline"]))*100)
        spacing=_clamp(abs(cand["second"]-cand["first"])/25*100)
        quality=.55*sim_score+.30*reaction_score+.15*spacing

    elif pattern in ("Head and Shoulders","Inverse Head and Shoulders"):
        idx=H[-3:] if pattern=="Head and Shoulders" else L[-3:]
        if len(idx)>=3:
            vals=(h[idx] if pattern=="Head and Shoulders" else l[idx])
            shoulders=_clamp(100-abs(vals[0]-vals[2])/max(abs(np.mean([vals[0],vals[2]])),1e-9)*1000)
            head_gap=((vals[1]-max(vals[0],vals[2]))/max(abs(vals[1]),1e-9)*100
                      if pattern=="Head and Shoulders"
                      else (min(vals[0],vals[2])-vals[1])/max(abs(vals[1]),1e-9)*100)
            quality=.55*shoulders+.45*_clamp(head_gap*3)

    elif pattern in ("Rising Channel","Falling Channel"):
        cand=_channel_candidate(df,H,L,pattern)
        if not cand:
            return 0.0
        quality=(.25*cand["hi_fit"]*100 + .25*cand["lo_fit"]*100 +
                 .20*cand["slope_ratio"]*100 + .15*(1-cand["width_cv"])*100 +
                 .15*cand["containment"]*100)
        if cand["recent_break"]:
            quality*=0.55

    elif pattern in ("Bullish Flag","Bearish Flag") and len(df)>=25:
        c=df["Close"].to_numpy(float)
        p=c[-25:-10]; q=c[-10:]
        impulse=abs(p[-1]/p[0]-1) if p[0] else 0
        consolidation=(q.max()-q.min())/max(abs(q.mean()),1e-9)
        quality=.55*_clamp(impulse*450)+.45*_clamp(100-consolidation*700)

    touch=_clamp(35+min(len(H)+len(L),12)*4)
    last_pivot=max((H[-1] if H else 0),(L[-1] if L else 0))
    recency=_clamp(100-max(0,len(df)-last_pivot)*1.2)
    return round(_clamp(.50*50+.40*quality+.07*touch+.03*recency),1)


def geometry(df):
    H,L=pivots(df)
    if df is None or len(df)==0:
        return [],0.0,H,L
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float)
    c=df["Close"].to_numpy(float)
    found=[]

    for kind,direction in (("Rising Channel","Bullish"),("Falling Channel","Bearish")):
        if _channel_candidate(df,H,L,kind):
            found.append((kind,direction))

    db=_double_candidate(df,"bottom",L)
    if db:
        found.append(("Double Bottom","Bullish"))
    dt=_double_candidate(df,"top",H)
    if dt:
        found.append(("Double Top","Bearish"))

    if len(H)>=3:
        a,b,d=H[-3:]
        if h[b]>h[a] and h[b]>h[d] and _near(h[a],h[d],.08):
            found.append(("Head and Shoulders","Bearish"))
    if len(L)>=3:
        a,b,d=L[-3:]
        if l[b]<l[a] and l[b]<l[d] and _near(l[a],l[d],.08):
            found.append(("Inverse Head and Shoulders","Bullish"))

    if len(df)>=25:
        p=c[-25:-10]; q=c[-10:]
        imp=p[-1]/p[0]-1 if p[0] else 0
        if abs(imp)>.12 and (q.max()-q.min())/max(abs(q.mean()),1e-9)<.08:
            found.append(("Bullish Flag" if imp>0 else "Bearish Flag",
                          "Bullish" if imp>0 else "Bearish"))

    found=list(dict.fromkeys(x for x in found if x[0] in ALLOWED_PATTERNS))
    scores=[pattern_confidence(df,p,d,H,L) for p,d in found]
    return found,round(max(scores),1) if scores else 0.0,H,L


def pattern_status(df,pattern,direction,H=None,L=None):
    if pattern not in ALLOWED_PATTERNS or df is None or len(df)==0:
        return "Forming"
    if H is None or L is None:
        H,L=pivots(df)
    d=pattern_target(df,pattern,direction,H,L)
    if not d or d.get("entry") is None:
        return "Forming"
    c=float(df["Close"].iloc[-1]); e=float(d["entry"])
    # For doubles, status is based on neckline breakout. For channels, a recent
    # decisive boundary break means the active channel is no longer active.
    if pattern in ("Rising Channel","Falling Channel"):
        ch=_channel_candidate(df,H,L,pattern)
        if not ch or ch["recent_break"]:
            return "Forming"
    if direction=="Bullish":
        return "Formed/Active" if c>=e else "Forming"
    if direction=="Bearish":
        return "Formed/Active" if c<=e else "Forming"
    return "Forming"


def pattern_target(df,pattern,direction,H=None,L=None):
    if pattern not in ALLOWED_PATTERNS or df is None or len(df)==0:
        return None
    if H is None or L is None:
        H,L=pivots(df)
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float)
    c=float(df["Close"].iloc[-1])
    entry=c; stop=None; t1=None; t2=None; method="Recent range"

    if pattern=="Double Bottom":
        cand=_double_candidate(df,"bottom",L)
        if cand:
            base=min(cand["first_value"],cand["second_value"])
            neck=cand["neckline"]
            height=max(neck-base,0)
            entry=neck; stop=base; t1=neck+height; t2=neck+1.618*height
            method="Neckline + pattern height"
    elif pattern=="Double Top":
        cand=_double_candidate(df,"top",H)
        if cand:
            top=max(cand["first_value"],cand["second_value"])
            neck=cand["neckline"]
            height=max(top-neck,0)
            entry=neck; stop=top; t1=neck-height; t2=neck-1.618*height
            method="Neckline - pattern height"
    elif pattern=="Head and Shoulders" and len(H)>=3 and len(L)>=2:
        head=h[H[-2]]
        neck=(min(l[L[-2]:H[-2]+1]) + min(l[H[-2]:H[-1]+1]))/2
        height=max(head-neck,0)
        entry=neck; stop=head; t1=neck-height; t2=neck-1.618*height
        method="Head-to-neckline measured move"
    elif pattern=="Inverse Head and Shoulders" and len(L)>=3 and len(H)>=2:
        head=l[L[-2]]
        neck=(max(h[H[-2]:L[-2]+1]) + max(h[L[-2]:L[-1]+1]))/2
        height=max(neck-head,0)
        entry=neck; stop=head; t1=neck+height; t2=neck+1.618*height
        method="Head-to-neckline measured move"
    elif pattern in ("Bullish Flag","Bearish Flag") and len(df)>=25:
        pole=float(max(h[-25:-10])-min(l[-25:-10]))
        if direction=="Bullish":
            entry=c; stop=float(min(l[-10:])); t1=c+pole; t2=c+1.618*pole
        else:
            entry=c; stop=float(max(h[-10:])); t1=c-pole; t2=c-1.618*pole
        method="Measured prior impulse"
    elif pattern in ("Rising Channel","Falling Channel"):
        ch=_channel_candidate(df,H,L,pattern)
        if ch:
            res=float(np.mean(ch["upper"][-3:])); sup=float(np.mean(ch["lower"][-3:]))
            if direction=="Bullish":
                entry=c; stop=sup; t1=res; t2=res+(res-sup)*.618
            else:
                entry=c; stop=res; t1=sup; t2=sup-(res-sup)*.618
            method="Channel boundary projection"

    if t1 is None:
        rng=float(max(h[-min(20,len(h)):])-min(l[-min(20,len(l)):]))
        if direction=="Bullish":
            t1=c+rng; t2=c+1.618*rng; stop=float(min(l[-min(20,len(l)):]))
        elif direction=="Bearish":
            t1=c-rng; t2=c-1.618*rng; stop=float(max(h[-min(20,len(h)):]))
        else:
            t1=c+rng; t2=c-rng
        method="Recent 20-candle range fallback"

    return {"current":c,"entry":entry,"stop":stop,"target1":t1,"target2":t2,"method":method}
