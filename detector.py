
import numpy as np
import pandas as pd

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(c,n=14):
    d=c.diff(); u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    v=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+u/v.replace(0,np.nan))
def stoch(df,n=14,d=3):
    lo=df.Low.rolling(n).min(); hi=df.High.rolling(n).max()
    k=100*(df.Close-lo)/(hi-lo).replace(0,np.nan)
    return k,k.rolling(d).mean()
def bb(df,n=20,m=2):
    mid=df.Close.rolling(n).mean(); sd=df.Close.rolling(n).std()
    return mid,mid+m*sd,mid-m*sd
def dmi(df,n=14):
    h,l,c=df.High,df.Low,df.Close; up=h.diff(); dn=-l.diff()
    p=up.where((up>dn)&(up>0),0.); q=dn.where((dn>up)&(dn>0),0.)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    a=tr.ewm(alpha=1/n,adjust=False).mean()
    pi=100*p.ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan)
    mi=100*q.ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan)
    dx=100*(pi-mi).abs()/(pi+mi).replace(0,np.nan)
    return pi,mi,dx.ewm(alpha=1/n,adjust=False).mean()
def piv(df,l=3,r=3):
    h=df.High.to_numpy(float); lo=df.Low.to_numpy(float); H=[]; L=[]
    for i in range(l,len(df)-r):
        if h[i]>=max(h[i-l:i+r+1]): H.append(i)
        if lo[i]<=min(lo[i-l:i+r+1]): L.append(i)
    return H,L
def reg(x,y):
    if len(x)<2:return 0.,0.,0.
    x=np.asarray(x,float); y=np.asarray(y,float); m,b=np.polyfit(x,y,1); p=m*x+b
    s=((y-p)**2).sum(); t=((y-y.mean())**2).sum()
    return float(m),float(b),float(np.clip(1-s/t if t else 0,0,1))
def trend(df):
    if len(df)<30:return 50.
    c=df.Close; return float(np.clip(50+(c.iloc[-10:].mean()/c.iloc[-30:].mean()-1)*500+(c.iloc[-1]/c.iloc[-11]-1)*200,0,100))
def vr(df):
    if "Volume" not in df or len(df)<22:return np.nan
    return float(df.Volume.iloc[-1]/max(df.Volume.iloc[-21:-1].mean(),1e-9))
def candles(df):
    o,h,l,c=df.Open,df.High,df.Low,df.Close
    po,pc=o.shift(1),c.shift(1)
    bull=bool(c.iloc[-1]>o.iloc[-1] and pc.iloc[-1]<po.iloc[-1] and c.iloc[-1]>=po.iloc[-1] and o.iloc[-1]<=pc.iloc[-1])
    bear=bool(c.iloc[-1]<o.iloc[-1] and pc.iloc[-1]>po.iloc[-1] and c.iloc[-1]<=po.iloc[-1] and o.iloc[-1]>=pc.iloc[-1])
    pierce=bool(len(df)>1 and c.iloc[-1]>o.iloc[-1] and pc.iloc[-1]<po.iloc[-1] and c.iloc[-1]>(po.iloc[-1]+pc.iloc[-1])/2)
    return bull or pierce,bear
def geometry(df):
    H,L=piv(df)
    out=[] 
    if len(H)>=2 and len(L)>=2:
        xh=np.array(H[-5:]); yh=df.High.iloc[H[-5:]].to_numpy(float)
        xl=np.array(L[-5:]); yl=df.Low.iloc[L[-5:]].to_numpy(float)
        mh,bh,rh=reg(xh,yh); ml,bl,rl=reg(xl,yl); sc=max(df.Close.mean(),1e-9)
        sh,sl=mh/sc,ml/sc; s=max(0,len(df)-40); e=len(df)-1
        w0=(mh*s+bh)-(ml*s+bl); w1=(mh*e+bh)-(ml*e+bl)
        cv=(w0-w1)/abs(w0) if abs(w0)>1e-9 else 0
        g=float(np.clip(50*(rh+rl)/2+50*np.clip(cv,0,1),0,100))
        if sh<-.0007 and sl>.0007 and cv>.12: out.append(("Symmetrical Triangle","Neutral"))
        if abs(sh)<.0007 and sl>.0007 and cv>.10: out.append(("Ascending Triangle","Bullish"))
        if sh<-.0007 and abs(sl)<.0007 and cv>.10: out.append(("Descending Triangle","Bearish"))
        if sh>.0004 and sl>.0004 and sh<sl and cv>.08: out.append(("Rising Wedge","Bearish"))
        if sh<-.0004 and sl<-.0004 and sh>sl and cv>.08: out.append(("Falling Wedge","Bullish"))
        if abs(sh-sl)<.0007 and abs(sh)>.00035: out.append(("Rising Channel" if sh>0 else "Falling Channel","Bullish" if sh>0 else "Bearish"))
    if len(H)>=2:
        a,b=H[-2],H[-1]
        if b-a>=8 and abs(df.High.iloc[a]-df.High.iloc[b])/max(df.High.iloc[a],df.High.iloc[b])<.03: out.append(("Double Top","Bearish"))
    if len(L)>=2:
        a,b=L[-2],L[-1]
        if b-a>=8 and abs(df.Low.iloc[a]-df.Low.iloc[b])/max(df.Low.iloc[a],df.Low.iloc[b])<.03: out.append(("Double Bottom","Bullish"))
    if len(df)>32:
        p=df.Close.iloc[-30:-20]; c=df.Close.iloc[-20:]; ret=p.iloc[-1]/p.iloc[0]-1; rng=(c.max()-c.min())/max(c.mean(),1e-9)
        if ret>.08 and rng<.10: out.append(("Bullish Flag","Bullish"))
        if ret<-.08 and rng<.10: out.append(("Bearish Flag","Bearish"))
    return out,g if 'g' in locals() else 0.,H,L
def indicators(df,direction):
    R=rsi(df.Close); K,D=stoch(df); PI,MI,A=dmi(df); e5,e13,e26=ema(df.Close,5),ema(df.Close,13),ema(df.Close,26); mid,up,lo=bb(df)
    bull=direction=="Bullish"; cb,cs=candles(df); V=vr(df)
    return dict(
      rsi=float(R.iloc[-1]), k=float(K.iloc[-1]), d=float(D.iloc[-1]), pi=float(PI.iloc[-1]), mi=float(MI.iloc[-1]), adx=float(A.iloc[-1]),
      bb="Lower" if df.Close.iloc[-1]<=lo.iloc[-1]*1.01 else ("Upper" if df.Close.iloc[-1]>=up.iloc[-1]*.99 else "Inside"),
      vol=V, ema=(e5.iloc[-1]>e13.iloc[-1] and e5.iloc[-1]>e26.iloc[-1]) if bull else (e5.iloc[-1]<e13.iloc[-1] and e5.iloc[-1]<e26.iloc[-1]),
      candle=cb if bull else cs,
      pco=bool(K.iloc[-1]>D.iloc[-1] and K.iloc[-2]<=D.iloc[-2]),
      nco=bool(K.iloc[-1]<D.iloc[-1] and K.iloc[-2]>=D.iloc[-2]),
      dipco=bool(PI.iloc[-1]>MI.iloc[-1] and PI.iloc[-2]<=MI.iloc[-2]),
      dinco=bool(PI.iloc[-1]<MI.iloc[-1] and PI.iloc[-2]>=MI.iloc[-2]))
def score(m,s,w=.35): return round(100*(w*m/len(m)+(1-w)*s/len(s)),1)
def row(setup,d,m,s):
    return dict(Setup=setup,Direction=d,Eligible=all(x[1] for x in m),Mandatory=f"{sum(x[1] for x in m)}/{len(m)}",Supporting=f"{sum(x[1] for x in s)}/{len(s)}",Confidence=score(m,s),Checklist="; ".join(n+(" ✓" if ok else " ✗") for n,ok in m+s))

def _safe_slope(vals, idxs):
    if len(idxs)<2:return 0.0
    return float(__import__('numpy').polyfit(__import__('numpy').asarray(idxs,float),__import__('numpy').asarray(vals,float),1)[0])
def _near(a,b,tol): return abs(a-b)/max(abs(a),abs(b),1e-9)<=tol
def _advanced_patterns(df,H,L):
    out=[]; n=len(df); c=df.Close.to_numpy(float); h=df.High.to_numpy(float); lo=df.Low.to_numpy(float)
    if len(H)>=3:
        a,b,d=H[-3:]
        if a<b<d and h[b]>h[a] and h[b]>h[d] and _near(h[a],h[d],.08):
            v1=lo[a:b+1].min(); v2=lo[b:d+1].min(); nl=(v1+v2)/2
            if h[b]>nl*1.03: out.append(('Head and Shoulders','Bearish'))
    if len(L)>=3:
        a,b,d=L[-3:]
        if a<b<d and lo[b]<lo[a] and lo[b]<lo[d] and _near(lo[a],lo[d],.08):
            p1=h[a:b+1].max(); p2=h[b:d+1].max(); nl=(p1+p2)/2
            if lo[b]<nl*.97: out.append(('Inverse Head and Shoulders','Bullish'))
    if len(H)>=2 and len(L)>=2:
        hh=h[H[-5:]]; ll=lo[L[-5:]]; top=float(__import__('numpy').median(hh)); bot=float(__import__('numpy').median(ll))
        if max(abs(hh-top))/top<.035 and max(abs(ll-bot))/bot<.035 and .03<(top-bot)/((top+bot)/2)<.30: out.append(('Rectangle','Neutral'))
    if n>=45:
        w=c[-45:]; left=int(__import__('numpy').argmax(w[:18])); bottom=int(__import__('numpy').argmin(w[12:34]))+12; right=int(__import__('numpy').argmax(w[bottom+4:40]))+bottom+4
        if 5<=left<bottom<right<=40:
            rim=(w[left]+w[right])/2
            if (rim-w[bottom])/rim>.08 and _near(w[left],w[right],.10) and len(w[right+1:])>=3 and (rim-w[right+1:].min())/rim<.10: out.append(('Cup and Handle','Bullish'))
    if n>=25:
        pre=c[-25:-10]; ch=h[-10:]; cl=lo[-10:]; impulse=pre[-1]/pre[0]-1; mh=_safe_slope(ch,range(10)); ml=_safe_slope(cl,range(10)); span0=ch[0]-cl[0]; span1=ch[-1]-cl[-1]
        if abs(impulse)>.10 and span1<span0*.75 and mh<0 and ml>0: out.append(('Bullish Pennant' if impulse>0 else 'Bearish Pennant', 'Bullish' if impulse>0 else 'Bearish'))
    if len(H)>=2 and len(L)>=2:
        res=max(h[H[-4:]]); sup=min(lo[L[-4:]]); ar=(h[-20:]-lo[-20:]).mean()
        if c[-1]>res+.15*ar: out.append(('Resistance Breakout','Bullish'))
        elif c[-1]<sup-.15*ar: out.append(('Support Breakdown','Bearish'))
    return out
_old_geometry=geometry
def geometry(df):
    found,g,H,L=_old_geometry(df); seen=set(found)
    for p in _advanced_patterns(df,H,L):
        if p not in seen: found.append(p); seen.add(p)
    return found,g,H,L


def pattern_target(df, pattern, direction, H=None, L=None):
    import numpy as np
    c=float(df.Close.iloc[-1]); h=df.High.to_numpy(float); lo=df.Low.to_numpy(float)
    if H is None or L is None: _,_,H,L=geometry(df)
    entry=c; stop=None; t1=None; t2=None; method='Measured move'
    if pattern in ('Double Bottom','Double Top') and len(H)>=2 and len(L)>=2:
        if pattern=='Double Bottom':
            neck=float(max(h[H[-2]],h[H[-1]])); base=float(min(lo[L[-2]],lo[L[-1]])); height=neck-base
            entry=neck; t1=neck+height; t2=neck+1.618*height; stop=base; method='Neckline + pattern height'
        else:
            neck=float(min(lo[L[-2]],lo[L[-1]])); top=float(max(h[H[-2]],h[H[-1]])); height=top-neck
            entry=neck; t1=neck-height; t2=neck-1.618*height; stop=top; method='Neckline - pattern height'
    elif pattern in ('Head and Shoulders','Inverse Head and Shoulders') and len(H)>=3 and len(L)>=3:
        if pattern=='Head and Shoulders':
            head=float(h[H[-2]]); neck=(float(min(lo[H[-3]:H[-2]+1]))+float(min(lo[H[-2]:H[-1]+1])))/2; height=head-neck
            entry=neck; t1=neck-height; t2=neck-1.618*height; stop=head; method='Head-to-neckline projection'
        else:
            head=float(lo[L[-2]]); neck=(float(max(h[L[-3]:L[-2]+1]))+float(max(h[L[-2]:L[-1]+1])))/2; height=neck-head
            entry=neck; t1=neck+height; t2=neck+1.618*height; stop=head; method='Head-to-neckline projection'
    elif len(H)>=2 and len(L)>=2:
        resistance=float(max(h[H[-4:]])); support=float(min(lo[L[-4:]])); height=max(resistance-support,0)
        if direction=='Bullish': entry=resistance; t1=resistance+height; t2=resistance+1.618*height; stop=support
        elif direction=='Bearish': entry=support; t1=support-height; t2=support-1.618*height; stop=resistance
        method='Pattern range projection'
    if t1 is None:
        rng=float(np.max(h[-20:])-np.min(lo[-20:]))
        if direction=='Bullish': t1=c+rng; t2=c+1.618*rng; stop=float(np.min(lo[-10:]))
        elif direction=='Bearish': t1=c-rng; t2=c-1.618*rng; stop=float(np.max(h[-10:]))
        method='Recent-range fallback'
    return dict(current=c,entry=entry,stop=stop,target1=t1,target2=t2,method=method)
