import numpy as np

def pivots(df,left=3,right=3):
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float)
    hi=[]; lo=[]
    for i in range(left,len(df)-right):
        if h[i]>=max(h[i-left:i+right+1]): hi.append(i)
        if l[i]<=min(l[i-left:i+right+1]): lo.append(i)
    return hi,lo

def reg(x,y):
    if len(x)<2:return 0.,0.,0.
    m,b=np.polyfit(np.asarray(x,float),np.asarray(y,float),1)
    p=m*np.asarray(x)+b
    ss=np.sum((np.asarray(y)-p)**2); st=np.sum((np.asarray(y)-np.mean(y))**2)
    return float(m),float(b),float(np.clip(1-ss/st if st else 0,0,1))

def trend_score(df):
    c=df["Close"].to_numpy(float)
    if len(c)<30:return 50.
    return float(np.clip(50+(np.mean(c[-10:])/max(np.mean(c[-30:]),1e-9)-1)*500+
                         (c[-1]/max(c[-11],1e-9)-1)*200,0,100))

def volume_score(df):
    if "Volume" not in df or len(df)<25:return 50.
    v=df["Volume"].dropna()
    if len(v)<25:return 50.
    return float(np.clip(50+(v.iloc[-1]/max(v.iloc[-21:-1].mean(),1e-9)-1)*50,0,100))

def detect_patterns(df,tide_df=None,wave_df=None,lookback=120):
    req={"Open","High","Low","Close"}
    if not req.issubset(df.columns): return []
    df=df.dropna(subset=list(req)).tail(lookback)
    if len(df)<45:return []
    tide_df=tide_df if tide_df is not None else df
    wave_df=wave_df if wave_df is not None else df
    hi,lo=pivots(df)
    if len(hi)<2 or len(lo)<2:return []
    xh=np.array(hi[-5:],float); yh=df.High.iloc[hi[-5:]].to_numpy(float)
    xl=np.array(lo[-5:],float); yl=df.Low.iloc[lo[-5:]].to_numpy(float)
    mh,bh,rh=reg(xh,yh); ml,bl,rl=reg(xl,yl)
    scale=max(float(df.Close.mean()),1e-9); sh,sl=mh/scale,ml/scale
    s=max(0,len(df)-40); e=len(df)-1
    w0=(mh*s+bh)-(ml*s+bl); w1=(mh*e+bh)-(ml*e+bl)
    conv=(w0-w1)/abs(w0) if abs(w0)>1e-9 else 0
    geometry=float(np.clip(50*(rh+rl)/2+50*np.clip(conv,0,1),0,100))
    cleanliness=float(np.clip(50+(len(xh)+len(xl))*6,0,100))
    volume=volume_score(df)
    ts=trend_score(tide_df); ws=trend_score(wave_df)
    out=[]
    def add(name,direction):
        if direction=="Bullish": a,b=ts,ws
        elif direction=="Bearish": a,b=100-ts,100-ws
        else:a=b=50.
        score=.30*geometry+.25*a+.20*b+.10*volume+.10*50+.05*cleanliness
        out.append({"Pattern":name,"Direction":direction,"Confidence":round(float(np.clip(score,0,100)),1),
                    "Tide Score":round(a,1),"Wave Score":round(b,1),
                    "Geometry":round(geometry,1),"Volume":round(volume,1),"Status":"FORMING"})
    if sh<-.0007 and sl>.0007 and conv>.12:add("Symmetrical Triangle","Neutral")
    if abs(sh)<.0007 and sl>.0007 and conv>.10:add("Ascending Triangle","Bullish")
    if sh<-.0007 and abs(sl)<.0007 and conv>.10:add("Descending Triangle","Bearish")
    if sh>.0004 and sl>.0004 and sh<sl and conv>.08:add("Rising Wedge","Bearish")
    if sh<-.0004 and sl<-.0004 and sh>sl and conv>.08:add("Falling Wedge","Bullish")
    if abs(sh-sl)<.0007 and abs(sh)>.00035:add("Rising Channel" if sh>0 else "Falling Channel","Bullish" if sh>0 else "Bearish")
    if len(hi)>=2:
        a,b=hi[-2],hi[-1]; va,vb=float(df.High.iloc[a]),float(df.High.iloc[b])
        if b-a>=5 and abs(va-vb)/max(va,vb)<.025:add("Double Top","Bearish")
    if len(lo)>=2:
        a,b=lo[-2],lo[-1]; va,vb=float(df.Low.iloc[a]),float(df.Low.iloc[b])
        if b-a>=5 and abs(va-vb)/max(va,vb)<.025:add("Double Bottom","Bullish")
    if len(df)>32:
        p=df.Close.iloc[-30:-20]; c=df.Close.iloc[-20:]; r=float(p.iloc[-1]/p.iloc[0]-1)
        cr=float((c.max()-c.min())/max(c.mean(),1e-9))
        if r>.08 and cr<.10:add("Bullish Flag","Bullish")
        if r<-.08 and cr<.10:add("Bearish Flag","Bearish")
    best={}
    for x in out:
        if x["Pattern"] not in best or x["Confidence"]>best[x["Pattern"]]["Confidence"]:best[x["Pattern"]]=x
    return sorted(best.values(),key=lambda x:x["Confidence"],reverse=True)
