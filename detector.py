
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
def scan_setups(df,wave):
    pats,g,H,L=geometry(df); out=[]; td=trend(df); wt=trend(wave)
    for d in ("Bullish","Bearish"):
        I=indicators(df,d); bp=any(x[1]==d for x in pats)
        if d=="Bullish":
            m=[("Tide bullish",td>=55),("Wave retracement",wt<td),("Bullish pattern",bp),("Bullish candle",I["candle"])]
            s=[("Lower BB",I["bb"]=="Lower"),("EMA positive",I["ema"]),("RSI >40",I["rsi"]>40),("Stochastic PCO",I["pco"]),("Volume >avg",not np.isnan(I["vol"]) and I["vol"]>1)]
        else:
            m=[("Tide bearish",td<=45),("Wave retracement",wt>td),("Bearish pattern",bp),("Bearish candle",I["candle"])]
            s=[("Upper BB",I["bb"]=="Upper"),("EMA negative",I["ema"]),("RSI <60",I["rsi"]<60),("Stochastic NCO",I["nco"]),("Volume >avg",not np.isnan(I["vol"]) and I["vol"]>1)]
        out.append(row("ASTA Triple Screen",d,m,s))
    for d in ("Bullish","Bearish"):
        I=indicators(df,d)
        ispat=any(x[1]==d and ("Double" in x[0]) for x in pats)
        if d=="Bullish":
            m=[("Double Bottom/Fake Breakdown",ispat),("Bullish reversal candle",I["candle"])]
            s=[("Lower BB",I["bb"]=="Lower"),("High confirmation volume",not np.isnan(I["vol"]) and I["vol"]>1.5),("TI uptick proxy",td>=50),("RSI >40",I["rsi"]>40),("Stochastic PCO",I["pco"]),("DI PCO/converging",I["dipco"] or abs(I["pi"]-I["mi"])<5)]
        else:
            m=[("Double Top/Fake Breakout",ispat),("Bearish reversal candle",I["candle"])]
            s=[("Upper BB",I["bb"]=="Upper"),("High confirmation volume",not np.isnan(I["vol"]) and I["vol"]>1.5),("TI downtick proxy",td<=50),("RSI <60",I["rsi"]<60),("Stochastic NCO",I["nco"]),("DI NCO/converging",I["dinco"] or abs(I["pi"]-I["mi"])<5)]
        out.append(row("ASTA Swing Trader",d,m,s))
    for d in ("Bullish","Bearish"):
        I=indicators(df,d); bull=d=="Bullish"; td=trend(df)
        # TLBO/TLBD is represented by a transparent pivot-break proxy.
        if bull:
            H,_=piv(df); br=bool(H and df.Close.iloc[-1]>df.High.iloc[H[-1]])
            m=[("BB upper-half challenge",I["bb"] in ("Upper","Inside")),("Tide bullish",td>=55),("RSI >40 (momentum)",I["rsi"]>40),("TLBO proxy",br),("Above-average volume",not np.isnan(I["vol"]) and I["vol"]>1),("EMA positive",I["ema"]),("DI PCO",I["dipco"]),("ADX >15",I["adx"]>15)]
            s=[("P >50 EMA",df.Close.iloc[-1]>ema(df.Close,50).iloc[-1]),("No immediate resistance proxy",True)]
        else:
            _,L=piv(df); br=bool(L and df.Close.iloc[-1]<df.Low.iloc[L[-1]])
            m=[("BB lower-half challenge",I["bb"] in ("Lower","Inside")),("Tide bearish",td<=45),("RSI <60 (momentum)",I["rsi"]<60),("TLBD proxy",br),("Above-average volume",not np.isnan(I["vol"]) and I["vol"]>1),("EMA negative",I["ema"]),("DI NCO",I["dinco"]),("ADX >15",I["adx"]>15)]
            s=[("P <50 EMA",df.Close.iloc[-1]<ema(df.Close,50).iloc[-1]),("No immediate support proxy",True)]
        out.append(row("ASTA Momentum Trader",d,m,s))
    return out
