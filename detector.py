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

def _line_fit_score(values,indices):
    if len(indices)<2:
        return 0.0
    y=np.asarray(values,float); x=np.asarray(indices,float)
    coef=np.polyfit(x,y,1)
    pred=coef[0]*x+coef[1]
    err=np.mean(np.abs(y-pred))/max(np.mean(np.abs(y)),1e-9)
    return _clamp(100*(1-err*12))

def pattern_confidence(df,pattern,direction,H=None,L=None):
    if pattern not in ALLOWED_PATTERNS or df is None or len(df)==0:
        return 0.0
    if H is None or L is None:
        H,L=pivots(df)
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float)
    quality=50.0

    if pattern in ("Double Top","Double Bottom"):
        idx=H[-2:] if pattern=="Double Top" else L[-2:]
        if len(idx)>=2:
            vals=(h[idx] if pattern=="Double Top" else l[idx])
            similarity=_clamp(100-(abs(vals[0]-vals[1])/max(abs(np.mean(vals)),1e-9))*1000)
            spacing=_clamp(min(100,abs(idx[1]-idx[0])/20*100))
            quality=.70*similarity+.30*spacing

    elif pattern in ("Head and Shoulders","Inverse Head and Shoulders"):
        idx=H[-3:] if pattern=="Head and Shoulders" else L[-3:]
        if len(idx)>=3:
            vals=(h[idx] if pattern=="Head and Shoulders" else l[idx])
            shoulders=_clamp(100-abs(vals[0]-vals[2])/max(abs(np.mean([vals[0],vals[2]])),1e-9)*1000)
            head_gap=((vals[1]-max(vals[0],vals[2]))/max(abs(vals[1]),1e-9)*100
                      if pattern=="Head and Shoulders"
                      else (min(vals[0],vals[2])-vals[1])/max(abs(vals[1]),1e-9)*100)
            quality=.55*shoulders+.45*_clamp(head_gap*3)

    elif pattern in ("Rising Channel","Falling Channel") and len(H)>=2 and len(L)>=2:
        hi=H[-5:]; lo=L[-5:]
        hs=_slope(h[hi],hi); ls=_slope(l[lo],lo)
        fit=(_line_fit_score(h[hi],hi)+_line_fit_score(l[lo],lo))/2
        parallel=_clamp(100-abs(hs-ls)/max(abs(hs),abs(ls),1e-9)*100)
        quality=.65*fit+.35*parallel

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

    if len(H)>=2 and len(L)>=2:
        hi=H[-5:]; lo=L[-5:]
        hs=_slope(h[hi],hi); ls=_slope(l[lo],lo)
        if hs>0 and ls>0 and _near(hs,ls,.55):
            found.append(("Rising Channel","Bullish"))
        if hs<0 and ls<0 and _near(hs,ls,.55):
            found.append(("Falling Channel","Bearish"))

    if len(H)>=2 and H[-1]-H[-2]>=8 and _near(h[H[-1]],h[H[-2]],.03):
        found.append(("Double Top","Bearish"))
    if len(L)>=2 and L[-1]-L[-2]>=8 and _near(l[L[-1]],l[L[-2]],.03):
        found.append(("Double Bottom","Bullish"))

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

    if pattern=="Double Bottom" and len(L)>=2:
        base=min(l[L[-2]],l[L[-1]])
        neck=float(max(h[L[-2]:L[-1]+1]))
        height=max(neck-base,0)
        entry=neck; stop=base; t1=neck+height; t2=neck+1.618*height
        method="Neckline + pattern height"
    elif pattern=="Double Top" and len(H)>=2:
        top=max(h[H[-2]],h[H[-1]])
        neck=float(min(l[H[-2]:H[-1]+1]))
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
    elif pattern in ("Rising Channel","Falling Channel") and len(H)>=2 and len(L)>=2:
        res=float(max(h[H[-5:]])); sup=float(min(l[L[-5:]]))
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
