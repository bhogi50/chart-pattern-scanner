
import numpy as np

def pivots(df,left=3,right=3):
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float)
    H=[]; L=[]
    for i in range(left,len(df)-right):
        if h[i]>=max(h[i-left:i]) and h[i]>=max(h[i+1:i+right+1]): H.append(i)
        if l[i]<=min(l[i-left:i]) and l[i]<=min(l[i+1:i+right+1]): L.append(i)
    return H,L

def _near(a,b,t=.04): return abs(a-b)/max(abs(a),abs(b),1e-9)<=t
def _slope(y,x):
    return float(np.polyfit(np.asarray(x,float),np.asarray(y,float),1)[0]) if len(x)>=2 else 0.

def geometry(df):
    H,L=pivots(df); h=df.High.to_numpy(float); l=df.Low.to_numpy(float); c=df.Close.to_numpy(float)
    f=[]
    if len(H)>=2 and len(L)>=2:
        hi=H[-5:]; lo=L[-5:]; sh=_slope(h[hi],hi); sl=_slope(l[lo],lo)
        if sh<0 and sl>0:f.append(("Symmetrical Triangle","Neutral"))
        if abs(sh)<abs(sl)*.35 and sl>0:f.append(("Ascending Triangle","Bullish"))
        if abs(sl)<abs(sh)*.35 and sh<0:f.append(("Descending Triangle","Bearish"))
        if sh>0 and sl>0 and sh>sl*1.25:f.append(("Rising Wedge","Bearish"))
        if sh<0 and sl<0 and abs(sl)>abs(sh)*1.25:f.append(("Falling Wedge","Bullish"))
        if sh>0 and sl>0 and _near(sh,sl,.55):f.append(("Rising Channel","Bullish"))
        if sh<0 and sl<0 and _near(sh,sl,.55):f.append(("Falling Channel","Bearish"))
        if _near(h[H[-1]],h[H[-2]],.03):f.append(("Double Top","Bearish"))
        if _near(l[L[-1]],l[L[-2]],.03):f.append(("Double Bottom","Bullish"))
    if len(H)>=3:
        a,b,d=H[-3:]; 
        if h[b]>h[a] and h[b]>h[d] and _near(h[a],h[d],.08):f.append(("Head and Shoulders","Bearish"))
    if len(L)>=3:
        a,b,d=L[-3:]
        if l[b]<l[a] and l[b]<l[d] and _near(l[a],l[d],.08):f.append(("Inverse Head and Shoulders","Bullish"))
    if len(H)>=2 and len(L)>=2:
        hh=h[H[-5:]]; ll=l[L[-5:]]; top=np.median(hh); bot=np.median(ll)
        if np.max(abs(hh-top))/max(top,1e-9)<.035 and np.max(abs(ll-bot))/max(bot,1e-9)<.035:
            f.append(("Rectangle","Neutral"))
    if len(df)>=25:
        p=c[-25:-10]; q=c[-10:]; imp=p[-1]/p[0]-1 if p[0] else 0
        if abs(imp)>.12 and (q.max()-q.min())/max(abs(q.mean()),1e-9)<.08:
            f.append(("Bullish Flag" if imp>0 else "Bearish Flag","Bullish" if imp>0 else "Bearish"))
    if len(df)>=45:
        w=c[-45:]; left=np.argmax(w[:18]); bottom=np.argmin(w[12:34])+12
        if bottom+4<40:
            right=np.argmax(w[bottom+4:40])+bottom+4
            if left<bottom<right and _near(w[left],w[right],.10) and (min(w[left],w[right])-w[bottom])/max(w[bottom],1e-9)>.08:
                f.append(("Cup and Handle","Bullish"))
    if len(df)>=25:
        p=c[-25:-10]; q=c[-10:]; imp=p[-1]/p[0]-1 if p[0] else 0
        if abs(imp)>.10 and (q.max()-q.min()) < .75*(df.High.iloc[-10]-df.Low.iloc[-10]):
            f.append(("Bullish Pennant" if imp>0 else "Bearish Pennant","Bullish" if imp>0 else "Bearish"))
    if len(H)>=2 and len(L)>=2:
        res=max(h[H[-4:]]); sup=min(l[L[-4:]]); rng=np.mean(h[-20:]-l[-20:])
        if c[-1]>res+.15*rng:f.append(("Resistance Breakout","Bullish"))
        if c[-1]<sup-.15*rng:f.append(("Support Breakdown","Bearish"))
    score=round(min(100,max(0,35+min(len(H)+len(L),10)*5)),1)
    return list(dict.fromkeys(f)),score,H,L

def pattern_target(df,pattern,direction,H=None,L=None):
    if H is None or L is None: _,_,H,L=geometry(df)
    h=df.High.to_numpy(float); l=df.Low.to_numpy(float); c=float(df.Close.iloc[-1])
    entry=c; stop=None; t1=None; t2=None; method="Recent range"
    if len(H)>=2 and len(L)>=2:
        res=float(max(h[H[-4:]])); sup=float(min(l[L[-4:]])); height=res-sup
        if direction=="Bullish": entry=res; stop=sup; t1=res+height; t2=res+1.618*height
        elif direction=="Bearish": entry=sup; stop=res; t1=sup-height; t2=sup-1.618*height
        else: t1=c+height; t2=c-height
        method="Pattern range measured move"
    if pattern=="Double Bottom" and len(L)>=2:
        base=min(l[L[-2]],l[L[-1]]); neck=max(h[L[-2]:L[-1]+1]); height=neck-base
        entry=neck; stop=base; t1=neck+height; t2=neck+1.618*height; method="Neckline + pattern height"
    elif pattern=="Double Top" and len(H)>=2:
        top=max(h[H[-2]],h[H[-1]]); neck=min(l[H[-2]:H[-1]+1]); height=top-neck
        entry=neck; stop=top; t1=neck-height; t2=neck-1.618*height; method="Neckline - pattern height"
    elif pattern in ("Bullish Flag","Bearish Flag","Rising Wedge","Falling Wedge","Cup and Handle") and len(df)>=25:
        pole=max(h[-25:-10])-min(l[-25:-10])
        if direction=="Bullish": stop=min(l[-10:]); t1=c+pole; t2=c+1.618*pole
        else: stop=max(h[-10:]); t1=c-pole; t2=c-1.618*pole
        method="Measured prior impulse"
    if t1 is None:
        rng=max(h[-20:])-min(l[-20:])
        t1=c+rng if direction=="Bullish" else c-rng
        t2=c+1.618*rng if direction=="Bullish" else c-1.618*rng
        stop=min(l[-20:]) if direction=="Bullish" else max(h[-20:])
    return {"current":c,"entry":entry,"stop":stop,"target1":t1,"target2":t2,"method":method}
