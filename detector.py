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

# ---------------------------------------------------------------------------
# Tunable thresholds. Pulled to the top so they can be re-calibrated against
# a real backtest (see backtest.py) instead of buried as magic numbers.
# ---------------------------------------------------------------------------
RECENT_PIVOT_WINDOW = 12          # how many recent swing points are searched for a candidate
DOUBLE_TOL_FLOOR = 0.018          # min relative similarity tolerance between the two peaks/troughs
DOUBLE_TOL_CAP = 0.06             # max relative similarity tolerance
DOUBLE_TOL_ATR_MULT = 1.1         # tolerance scales with ATR/price
DOUBLE_MIN_REACTION_ATR_MULT = 1.3
DOUBLE_MIN_REACTION_PCT = 0.02
HS_TOL_FLOOR = 0.02               # shoulder symmetry tolerance (relative)
HS_TOL_CAP = 0.07
HS_TOL_ATR_MULT = 1.2
HS_MIN_PROMINENCE_ATR_MULT = 1.2  # head must clear shoulders by at least this much ATR
HS_MIN_PROMINENCE_PCT = 0.02
VOLUME_LOOKBACK = 20
VOLUME_BREAKOUT_MULT = 1.2        # breakout candle volume vs its trailing average


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


def _volume_ratio(df, idx, lookback=VOLUME_LOOKBACK):
    """Volume on candle `idx` relative to its trailing average. None if no
    Volume column, insufficient history, or non-positive average (illiquid
    data)."""
    if df is None or "Volume" not in df.columns or idx is None:
        return None
    v = df["Volume"].to_numpy(float)
    idx = int(idx)
    if idx < 0 or idx >= len(v):
        return None
    start = max(0, idx - lookback)
    if idx <= start:
        return None
    avg = np.nanmean(v[start:idx])
    if not np.isfinite(avg) or avg <= 0 or not np.isfinite(v[idx]):
        return None
    return float(v[idx] / avg)


def _breakout_volume_confirmed(df, entry_idx=None):
    """Checks volume on/near the most recent candle (proxy for the breakout
    candle) against its trailing average. Returns (confirmed: bool|None,
    ratio: float|None). None means "no volume data available", not "failed"."""
    idx = len(df) - 1 if entry_idx is None else entry_idx
    ratio = _volume_ratio(df, idx)
    if ratio is None:
        return None, None
    return ratio >= VOLUME_BREAKOUT_MULT, ratio


def _double_candidate(df, side, piv):
    """Return the strongest structurally valid double-top/bottom candidate."""
    if len(piv) < 2:
        return None
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float)
    atr=max(_atr(df),1e-9)
    candidates=[]
    # Do not restrict to the last two pivots: a minor pivot can sit between the
    # two meaningful bottoms/tops. Search a realistic recent window.
    recent=piv[-RECENT_PIVOT_WINDOW:]
    for ai in range(len(recent)-2, -1, -1):
        for bi in range(ai+1, len(recent)):
            a,b=recent[ai],recent[bi]
            gap=b-a
            if gap < 5 or gap > min(100, max(20, len(df)-5)):
                continue
            if side=="bottom":
                va,vb=l[a],l[b]
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
            # Volatility-aware similarity: real double tops/bottoms rarely land
            # on the exact same print. Scale tolerance with ATR but keep a
            # sane floor/ceiling so we don't accept clearly different levels
            # on very volatile names or reject clean patterns on very calm ones.
            tol=min(DOUBLE_TOL_CAP, max(DOUBLE_TOL_FLOOR, DOUBLE_TOL_ATR_MULT*atr/max(abs((va+vb)/2),1e-9)))
            if similarity > tol:
                continue
            # The middle reaction must be meaningful, not a flat V/noise move.
            min_reaction=max(DOUBLE_MIN_REACTION_ATR_MULT*atr, DOUBLE_MIN_REACTION_PCT*max(abs((va+vb)/2),1e-9))
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


def _hs_candidate(df, side, piv, piv_opp):
    """Structurally-searched Head & Shoulders / Inverse H&S candidate.

    side: "head" (H&S, uses highs for shoulders/head, lows for neckline) or
          "inverse" (uses lows for shoulders/head, highs for neckline).
    piv: pivot indices of the extreme type (H for "head", L for "inverse").
    piv_opp: pivot indices of the opposite type, used to locate the neckline.
    Searches triples instead of blindly taking the last 3 pivots, so a minor
    wiggle between the true shoulders and head no longer hides the pattern.
    """
    if len(piv) < 3:
        return None
    h=df["High"].to_numpy(float); l=df["Low"].to_numpy(float)
    atr=max(_atr(df),1e-9)
    recent=piv[-RECENT_PIVOT_WINDOW:]
    is_head = side=="head"
    vals = h if is_head else l
    candidates=[]
    for li in range(len(recent)-2):
        for hi in range(li+1, len(recent)-1):
            for ri in range(hi+1, len(recent)):
                lft, head, rgt = recent[li], recent[hi], recent[ri]
                gap = rgt - lft
                if gap < 8 or gap > min(140, max(25, len(df)-5)):
                    continue
                v_l, v_h, v_r = vals[lft], vals[head], vals[rgt]
                if is_head:
                    if not (v_h > v_l and v_h > v_r):
                        continue
                    prominence = v_h - max(v_l, v_r)
                else:
                    if not (v_h < v_l and v_h < v_r):
                        continue
                    prominence = min(v_l, v_r) - v_h
                mid = (v_l+v_r)/2
                similarity = abs(v_l-v_r)/max(abs(mid),1e-9)
                tol = min(HS_TOL_CAP, max(HS_TOL_FLOOR, HS_TOL_ATR_MULT*atr/max(abs(mid),1e-9)))
                if similarity > tol:
                    continue
                min_prom = max(HS_MIN_PROMINENCE_ATR_MULT*atr, HS_MIN_PROMINENCE_PCT*max(abs(v_h),1e-9))
                if prominence < min_prom:
                    continue
                # Neckline: the two opposite-type pivots bracketing the head
                # (the reaction lows for H&S, reaction highs for inverse).
                opp_left = [p for p in piv_opp if lft < p < head]
                opp_right = [p for p in piv_opp if head < p < rgt]
                if not opp_left or not opp_right:
                    continue
                n1_idx = opp_left[-1]
                n2_idx = opp_right[0]
                n1 = (l[n1_idx] if is_head else h[n1_idx])
                n2 = (l[n2_idx] if is_head else h[n2_idx])
                neckline = (n1+n2)/2
                recency = max(0.0, 1.0-(len(df)-rgt)/max(len(df),1))
                spacing = min(1.0, gap/35.0)
                sim_score = max(0.0, 1.0-similarity/max(tol,1e-9))
                prom_score = min(1.0, prominence/max(4*atr, 0.06*abs(v_h)))
                score = 0.40*sim_score + 0.35*prom_score + 0.13*spacing + 0.12*recency
                candidates.append((score, {
                    "left":lft, "head":head, "right":rgt,
                    "left_value":float(v_l), "head_value":float(v_h), "right_value":float(v_r),
                    "neck1_idx":int(n1_idx), "neck2_idx":int(n2_idx),
                    "neckline":float(neckline), "prominence":float(prominence), "tolerance":float(tol),
                }))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


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
    quality=50.0
    breakout_idx=None

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
        breakout_idx=cand["second"]

    elif pattern in ("Head and Shoulders","Inverse Head and Shoulders"):
        cand=_hs_candidate(df,"head" if pattern=="Head and Shoulders" else "inverse",
                            H if pattern=="Head and Shoulders" else L,
                            L if pattern=="Head and Shoulders" else H)
        if not cand:
            return 0.0
        mid=(cand["left_value"]+cand["right_value"])/2
        similarity=abs(cand["left_value"]-cand["right_value"])/max(abs(mid),1e-9)
        sim_score=_clamp(100*(1-similarity/max(cand["tolerance"],1e-9)))
        prom_score=_clamp(cand["prominence"]/max(4*_atr(df),0.06*abs(cand["head_value"]))*100)
        spacing=_clamp((cand["right"]-cand["left"])/35*100)
        quality=.45*sim_score+.40*prom_score+.15*spacing
        breakout_idx=cand["right"]

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
        breakout_idx=len(df)-1

    touch=_clamp(35+min(len(H)+len(L),12)*4)
    last_pivot=max((H[-1] if H else 0),(L[-1] if L else 0))
    recency=_clamp(100-max(0,len(df)-last_pivot)*1.2)

    # Volume is a soft confirmation signal, not a hard gate: absence of
    # Volume data (None) contributes nothing rather than penalizing.
    vol_bonus=0.0
    confirmed,_ratio=_breakout_volume_confirmed(df, breakout_idx)
    if confirmed is True:
        vol_bonus=5.0
    elif confirmed is False:
        vol_bonus=-5.0

    # NOTE: this used to be `.50*50` (a hardcoded 25 regardless of pattern
    # quality), which capped every confidence score at 75% no matter how
    # clean the pattern was. Quality now actually drives the bulk of the
    # score, with touch/recency/volume as secondary adjustments.
    base=.70*quality+.18*touch+.07*recency+vol_bonus
    return round(_clamp(base),1)


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

    if _double_candidate(df,"bottom",L):
        found.append(("Double Bottom","Bullish"))
    if _double_candidate(df,"top",H):
        found.append(("Double Top","Bearish"))

    if _hs_candidate(df,"head",H,L):
        found.append(("Head and Shoulders","Bearish"))
    if _hs_candidate(df,"inverse",L,H):
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


def pattern_points(df, pattern, direction, H=None, L=None):
    """Returns the *exact* swing indices/values a given pattern's confidence
    and target were computed from, so the chart can draw the same geometry
    that was actually scored instead of a naive 'last N pivots' guess."""
    if pattern not in ALLOWED_PATTERNS or df is None or len(df)==0:
        return None
    if H is None or L is None:
        H,L=pivots(df)

    if pattern in ("Double Top","Double Bottom"):
        cand=_double_candidate(df,"top" if pattern=="Double Top" else "bottom",H if pattern=="Double Top" else L)
        if not cand:
            return None
        return {"type":"double","indices":[cand["first"],cand["second"]],
                "values":[cand["first_value"],cand["second_value"]],
                "neck_idx":cand["neck_idx"],"neckline":cand["neckline"]}

    if pattern in ("Head and Shoulders","Inverse Head and Shoulders"):
        cand=_hs_candidate(df,"head" if pattern=="Head and Shoulders" else "inverse",
                            H if pattern=="Head and Shoulders" else L,
                            L if pattern=="Head and Shoulders" else H)
        if not cand:
            return None
        return {"type":"hs","indices":[cand["left"],cand["head"],cand["right"]],
                "values":[cand["left_value"],cand["head_value"],cand["right_value"]],
                "neck_idx":[cand["neck1_idx"],cand["neck2_idx"]],"neckline":cand["neckline"]}

    if pattern in ("Rising Channel","Falling Channel"):
        cand=_channel_candidate(df,H,L,pattern)
        if not cand:
            return None
        return {"type":"channel","hi":cand["hi"],"lo":cand["lo"]}

    if pattern in ("Bullish Flag","Bearish Flag") and len(df)>=25:
        return {"type":"flag","pole_start":len(df)-25,"pole_end":len(df)-10,
                "consolidation_start":len(df)-10,"consolidation_end":len(df)-1}

    return None


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
    elif pattern=="Head and Shoulders":
        cand=_hs_candidate(df,"head",H,L)
        if cand:
            head=cand["head_value"]; neck=cand["neckline"]
            height=max(head-neck,0)
            entry=neck; stop=head; t1=neck-height; t2=neck-1.618*height
            method="Head-to-neckline measured move"
    elif pattern=="Inverse Head and Shoulders":
        cand=_hs_candidate(df,"inverse",L,H)
        if cand:
            head=cand["head_value"]; neck=cand["neckline"]
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

    volume_confirmed, volume_ratio = _breakout_volume_confirmed(df)
    return {"current":c,"entry":entry,"stop":stop,"target1":t1,"target2":t2,"method":method,
            "volume_confirmed":volume_confirmed,"volume_ratio":volume_ratio}
