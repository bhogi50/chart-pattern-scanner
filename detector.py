
import numpy as np

def pivots(df, left=3, right=3):
    if df is None or len(df) < left + right + 3:
        return [], []
    h = df["High"].to_numpy(float); l = df["Low"].to_numpy(float)
    H, L = [], []
    for i in range(left, len(df)-right):
        if h[i] >= np.max(h[i-left:i]) and h[i] >= np.max(h[i+1:i+right+1]):
            H.append(i)
        if l[i] <= np.min(l[i-left:i]) and l[i] <= np.min(l[i+1:i+right+1]):
            L.append(i)
    return H, L

def _near(a,b,t=.04):
    return abs(a-b)/max(abs(a),abs(b),1e-9) <= t

def _slope(y,x):
    return float(np.polyfit(np.asarray(x,float), np.asarray(y,float), 1)[0]) if len(x) >= 2 else 0.0

def _clamp(x, lo=0, hi=100):
    return float(max(lo, min(hi, x)))

def _line_fit_score(values, indices):
    if len(indices) < 2:
        return 0.0
    y=np.asarray(values,dtype=float)
    x=np.asarray(indices,dtype=float)
    coef=np.polyfit(x,y,1)
    pred=coef[0]*x+coef[1]
    err=np.mean(np.abs(y-pred))/max(np.mean(np.abs(y)),1e-9)
    return _clamp(100*(1-err*12))

def pattern_confidence(df, pattern, direction, H=None, L=None):
    if df is None or len(df)==0:
        return 0.0
    if H is None or L is None:
        _,_,H,L=geometry(df)
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float); c=df["Close"].to_numpy(float)

    touch = _clamp(35 + min(len(H)+len(L),12)*4)
    recency = _clamp(100 - max(0,len(df)-max((H[-1] if H else 0),(L[-1] if L else 0)))*1.2)
    base = 45.0
    quality = 0.0

    if len(H)>=2 and len(L)>=2:
        hi=H[-5:]; lo=L[-5:]
        hs=_slope(h[hi],hi); ls=_slope(l[lo],lo)
        hf=_line_fit_score(h[hi],hi); lf=_line_fit_score(l[lo],lo)
        fit=(hf+lf)/2

        if pattern in ("Symmetrical Triangle","Ascending Triangle","Descending Triangle",
                       "Rising Wedge","Falling Wedge","Rising Channel","Falling Channel"):
            quality = fit
            if pattern=="Symmetrical Triangle":
                convergence = _clamp(100 - abs((h[hi[-1]]-l[lo[-1]])/(max(h[hi[-1]],1e-9))
                                               -(h[hi[0]]-l[lo[0]])/(max(h[hi[0]],1e-9)))*500)
                quality=.65*fit+.35*convergence
            elif pattern=="Ascending Triangle":
                quality=.5*hf+.5*_clamp(100-abs(hs)*100000)
            elif pattern=="Descending Triangle":
                quality=.5*lf+.5*_clamp(100-abs(ls)*100000)
            elif pattern in ("Rising Wedge","Falling Wedge"):
                convergence=_clamp(100-abs(abs(hs)-abs(ls))/max(abs(hs),abs(ls),1e-9)*100)
                quality=.65*fit+.35*convergence
            else:
                parallel=_clamp(100-abs(hs-ls)/max(abs(hs),abs(ls),1e-9)*100)
                quality=.65*fit+.35*parallel

        elif pattern in ("Double Top","Double Bottom"):
            pts=h[H[-2:]] if pattern=="Double Top" else l[L[-2:]]
            similarity=_clamp(100-(abs(pts[0]-pts[1])/max(abs(np.mean(pts)),1e-9))*1000)
            spacing=_clamp(min(100,(abs((H[-1]-H[-2]) if pattern=="Double Top" else (L[-1]-L[-2])))/20*100))
            quality=.65*similarity+.35*spacing

        elif pattern in ("Head and Shoulders","Inverse Head and Shoulders"):
            pts=h[H[-3:]] if pattern=="Head and Shoulders" else l[L[-3:]]
            shoulders=_clamp(100-abs(pts[0]-pts[2])/max(abs(np.mean([pts[0],pts[2]])),1e-9)*1000)
            head_gap = ((pts[1]-max(pts[0],pts[2]))/max(abs(pts[1]),1e-9)*100
                        if pattern=="Head and Shoulders"
                        else (min(pts[0],pts[2])-pts[1])/max(abs(pts[1]),1e-9)*100)
            head_quality=_clamp(head_gap*3)
            quality=.55*shoulders+.45*head_quality

        elif pattern=="Rectangle":
            top=np.median(h[H[-5:]]); bot=np.median(l[L[-5:]])
            top_dev=np.std(h[H[-5:]])/max(abs(top),1e-9)
            bot_dev=np.std(l[L[-5:]])/max(abs(bot),1e-9)
            quality=_clamp(100-(top_dev+bot_dev)*1000)

    if pattern in ("Bullish Flag","Bearish Flag","Bullish Pennant","Bearish Pennant","Cup and Handle"):
        if len(df)>=25:
            p=c[-25:-10]; q=c[-10:]
            impulse=abs(p[-1]/p[0]-1) if p[0] else 0
            consolidation=(q.max()-q.min())/max(abs(q.mean()),1e-9)
            impulse_score=_clamp(impulse*450)
            consolidation_score=_clamp(100-consolidation*700)
            quality=.55*impulse_score+.45*consolidation_score
            if pattern=="Cup and Handle" and len(df)>=45:
                w=c[-45:]; left=np.max(w[:18]); bottom=np.min(w[12:34]); right=np.max(w[30:40])
                cup_depth=(left-bottom)/max(abs(left),1e-9)
                rim_similarity=_clamp(100-abs(left-right)/max(abs(left),1e-9)*1000)
                quality=.45*_clamp(cup_depth*450)+.55*rim_similarity

    if pattern in ("Resistance Breakout","Support Breakdown"):
        recent_range=max(h[-20:])-min(l[-20:])
        distance=abs(c[-1]-(max(h[H[-4:]]) if H else c[-1]) if direction=="Bullish"
                     else c[-1]-(min(l[L[-4:]]) if L else c[-1]))
        quality=_clamp(55+(distance/max(recent_range,1e-9))*450)

    # A recent, well-defined structure gets more weight than a stale structure.
    score=0.50*base + 0.40*quality + 0.07*touch + 0.03*recency
    return round(_clamp(score),1)

def geometry(df):
    H,L=pivots(df)
    if len(df)==0:
        return [], 0.0, H, L
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float); c=df["Close"].to_numpy(float)
    found=[]
    if len(H)>=2 and len(L)>=2:
        hi=H[-5:]; lo=L[-5:]
        sh=_slope(h[hi],hi); sl=_slope(l[lo],lo)
        if sh<0 and sl>0: found.append(("Symmetrical Triangle","Neutral"))
        if abs(sh)<abs(sl)*.35 and sl>0: found.append(("Ascending Triangle","Bullish"))
        if abs(sl)<abs(sh)*.35 and sh<0: found.append(("Descending Triangle","Bearish"))
        if sh>0 and sl>0 and sh>sl*1.25: found.append(("Rising Wedge","Bearish"))
        if sh<0 and sl<0 and abs(sl)>abs(sh)*1.25: found.append(("Falling Wedge","Bullish"))
        if sh>0 and sl>0 and _near(sh,sl,.55): found.append(("Rising Channel","Bullish"))
        if sh<0 and sl<0 and _near(sh,sl,.55): found.append(("Falling Channel","Bearish"))
        if H[-1]-H[-2]>=8 and _near(h[H[-1]],h[H[-2]],.03): found.append(("Double Top","Bearish"))
        if L[-1]-L[-2]>=8 and _near(l[L[-1]],l[L[-2]],.03): found.append(("Double Bottom","Bullish"))

    if len(H)>=3:
        a,b,d=H[-3:]
        if h[b]>h[a] and h[b]>h[d] and _near(h[a],h[d],.08): found.append(("Head and Shoulders","Bearish"))
    if len(L)>=3:
        a,b,d=L[-3:]
        if l[b]<l[a] and l[b]<l[d] and _near(l[a],l[d],.08): found.append(("Inverse Head and Shoulders","Bullish"))

    if len(H)>=2 and len(L)>=2:
        hh=h[H[-5:]]; ll=l[L[-5:]]
        top=float(np.median(hh)); bot=float(np.median(ll))
        if np.max(abs(hh-top))/max(abs(top),1e-9)<.035 and np.max(abs(ll-bot))/max(abs(bot),1e-9)<.035:
            found.append(("Rectangle","Neutral"))

    if len(df)>=25:
        p=c[-25:-10]; q=c[-10:]
        imp=p[-1]/p[0]-1 if p[0] else 0
        if abs(imp)>.12 and (q.max()-q.min())/max(abs(q.mean()),1e-9)<.08:
            found.append(("Bullish Flag" if imp>0 else "Bearish Flag","Bullish" if imp>0 else "Bearish"))

    if len(df)>=45:
        w=c[-45:]; left=int(np.argmax(w[:18])); bottom=int(np.argmin(w[12:34]))+12
        if bottom+4<40:
            right=int(np.argmax(w[bottom+4:40]))+bottom+4
            if left<bottom<right and _near(w[left],w[right],.10):
                if (min(w[left],w[right])-w[bottom])/max(abs(w[bottom]),1e-9)>.08:
                    found.append(("Cup and Handle","Bullish"))

    if len(df)>=25:
        p=c[-25:-10]; q=c[-10:]
        imp=p[-1]/p[0]-1 if p[0] else 0
        if abs(imp)>.10 and (q.max()-q.min()) < .75*(df.High.iloc[-10]-df.Low.iloc[-10]):
            found.append(("Bullish Pennant" if imp>0 else "Bearish Pennant","Bullish" if imp>0 else "Bearish"))

    if len(H)>=2 and len(L)>=2:
        res=float(max(h[H[-4:]])); sup=float(min(l[L[-4:]]))
        rng=float(np.mean(h[-min(20,len(h)):] - l[-min(20,len(l)):]))
        if c[-1] > res + .15*rng: found.append(("Resistance Breakout","Bullish"))
        if c[-1] < sup - .15*rng: found.append(("Support Breakdown","Bearish"))

    found=list(dict.fromkeys(found))
    scores=[pattern_confidence(df,p,d,H,L) for p,d in found]
    overall=round(max(scores),1) if scores else 0.0
    return found, overall, H, L

def pattern_target(df, pattern, direction, H=None, L=None):
    if df is None or len(df)==0:
        return None
    if H is None or L is None:
        _,_,H,L=geometry(df)
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float); c=float(df["Close"].iloc[-1])
    entry=c; stop=None; t1=None; t2=None; method="Recent range"

    if len(H)>=2 and len(L)>=2:
        res=float(max(h[H[-4:]])); sup=float(min(l[L[-4:]])); height=max(res-sup,0)
        if direction=="Bullish": entry=res; stop=sup; t1=res+height; t2=res+1.618*height
        elif direction=="Bearish": entry=sup; stop=res; t1=sup-height; t2=sup-1.618*height
        else: t1=c+height; t2=c-height
        method="Pattern range measured move"

    if pattern=="Double Bottom" and len(L)>=2:
        base=min(l[L[-2]],l[L[-1]]); neck=float(max(h[L[-2]:L[-1]+1])); height=neck-base
        entry=neck; stop=base; t1=neck+height; t2=neck+1.618*height; method="Neckline + pattern height"
    elif pattern=="Double Top" and len(H)>=2:
        top=max(h[H[-2]],h[H[-1]]); neck=float(min(l[H[-2]:H[-1]+1])); height=top-neck
        entry=neck; stop=top; t1=neck-height; t2=neck-1.618*height; method="Neckline - pattern height"
    elif pattern in ("Head and Shoulders","Inverse Head and Shoulders") and len(H)>=3 and len(L)>=3:
        if pattern=="Head and Shoulders":
            head=h[H[-2]]
            neck=(min(l[H[-3]:H[-2]+1])+min(l[H[-2]:H[-1]+1]))/2
            height=head-neck; entry=neck; stop=head; t1=neck-height; t2=neck-1.618*height
        else:
            head=l[L[-2]]
            neck=(max(h[L[-3]:L[-2]+1])+max(h[L[-2]:L[-1]+1]))/2
            height=neck-head; entry=neck; stop=head; t1=neck+height; t2=neck+1.618*height
        method="Head-to-neckline measured move"
    elif pattern in ("Bullish Flag","Bearish Flag","Rising Wedge","Falling Wedge","Cup and Handle") and len(df)>=25:
        pole=float(max(h[-25:-10])-min(l[-25:-10]))
        if direction=="Bullish": stop=float(min(l[-10:])); t1=c+pole; t2=c+1.618*pole
        elif direction=="Bearish": stop=float(max(h[-10:])); t1=c-pole; t2=c-1.618*pole
        method="Measured prior impulse"
    elif pattern in ("Rising Channel","Falling Channel") and len(H)>=2 and len(L)>=2:
        res=float(max(h[H[-5:]])); sup=float(min(l[L[-5:]]))
        if direction=="Bullish": stop=sup; t1=res; t2=res+(res-sup)*.618
        else: stop=res; t1=sup; t2=sup-(res-sup)*.618
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
