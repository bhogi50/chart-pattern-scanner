
import numpy as np
import pandas as pd


def _linreg(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    m, b = np.polyfit(x, y, 1)
    pred = m * x + b
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(m), float(b), float(r2)


def pivots(df, left=3, right=3):
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    hi, lo = [], []
    for i in range(left, len(df) - right):
        if h[i] == np.max(h[i-left:i+right+1]):
            hi.append(i)
        if l[i] == np.min(l[i-left:i+right+1]):
            lo.append(i)
    return hi, lo


def _quality(r2a, r2b, convergence, touches, volume_ok=True):
    q = 100 * (0.35*np.clip(r2a,0,1) + 0.35*np.clip(r2b,0,1)
               + 0.20*np.clip(convergence,0,1) + 0.10*np.clip(touches/6,0,1))
    if not volume_ok:
        q *= 0.95
    return round(float(np.clip(q, 0, 99)), 1)


def detect_patterns(df, lookback=100):
    df = df.copy().dropna()
    if len(df) < 40:
        return []

    df = df.tail(lookback)
    hi, lo = pivots(df)
    if len(hi) < 2 or len(lo) < 2:
        return []

    results = []
    xh = np.array(hi[-5:], float)
    yh = df["High"].iloc[hi[-5:]].to_numpy(float)
    xl = np.array(lo[-5:], float)
    yl = df["Low"].iloc[lo[-5:]].to_numpy(float)

    mh, bh, r2h = _linreg(xh, yh)
    ml, bl, r2l = _linreg(xl, yl)

    span_start = max(0, len(df)-35)
    x = np.arange(span_start, len(df))
    close = df["Close"].iloc[span_start:].to_numpy(float)
    if len(close) < 20:
        return []

    # Relative slope normalized by price.
    p = float(close.mean())
    ns_h = mh / p if np.isfinite(mh) else 0
    ns_l = ml / p if np.isfinite(ml) else 0

    width_start = (mh*span_start+bh) - (ml*span_start+bl)
    width_end = (mh*(len(df)-1)+bh) - (ml*(len(df)-1)+bl)
    convergence = (width_start-width_end)/abs(width_start) if abs(width_start)>1e-9 else 0
    touches = min(len(xh)+len(xl), 10)

    # Triangle family
    if ns_h < -0.0008 and ns_l > 0.0008 and convergence > 0.15:
        results.append(("Symmetrical Triangle", "Neutral",
                        _quality(r2h,r2l,convergence,touches)))
    if abs(ns_h) < 0.0007 and ns_l > 0.0008 and convergence > 0.10:
        results.append(("Ascending Triangle", "Bullish",
                        _quality(r2h,r2l,convergence,touches)))
    if ns_h < -0.0008 and abs(ns_l) < 0.0007 and convergence > 0.10:
        results.append(("Descending Triangle", "Bearish",
                        _quality(r2h,r2l,convergence,touches)))

    # Wedges
    if ns_h > 0.0005 and ns_l > 0.0005 and ns_h < ns_l and convergence > 0.10:
        results.append(("Rising Wedge", "Bearish",
                        _quality(r2h,r2l,convergence,touches)))
    if ns_h < -0.0005 and ns_l < -0.0005 and ns_h > ns_l and convergence > 0.10:
        results.append(("Falling Wedge", "Bullish",
                        _quality(r2h,r2l,convergence,touches)))

    # Channels: parallel-ish boundaries.
    if abs(ns_h-ns_l) < 0.0008 and abs(ns_h) > 0.0004:
        direction = "Bullish" if ns_h > 0 else "Bearish"
        name = "Rising Channel" if ns_h > 0 else "Falling Channel"
        results.append((name, direction,
                        _quality(r2h,r2l,0.5,touches)))

    # Double top / bottom from recent pivots.
    if len(hi) >= 2:
        a,b = hi[-2], hi[-1]
        va,vb = df["High"].iloc[a], df["High"].iloc[b]
        if abs(va-vb)/max(va,vb) < 0.025 and b-a >= 5:
            results.append(("Double Top", "Bearish", round(float(75 + 20*(1-abs(va-vb)/max(va,vb)/0.025)),1)))
    if len(lo) >= 2:
        a,b = lo[-2], lo[-1]
        va,vb = df["Low"].iloc[a], df["Low"].iloc[b]
        if abs(va-vb)/max(va,vb) < 0.025 and b-a >= 5:
            results.append(("Double Bottom", "Bullish", round(float(75 + 20*(1-abs(va-vb)/max(va,vb)/0.025)),1)))

    # Bull/Bear flag approximation: strong prior move followed by tight consolidation.
    n = min(25, len(df)-10)
    prior = df["Close"].iloc[-n-10:-n].to_numpy(float)
    cons = df["Close"].iloc[-n:].to_numpy(float)
    if len(prior) >= 8 and len(cons) >= 12:
        prior_ret = prior[-1]/prior[0]-1
        cons_range = (cons.max()-cons.min())/max(cons.mean(),1e-9)
        if prior_ret > 0.08 and cons_range < 0.10:
            results.append(("Bullish Flag", "Bullish", 78.0))
        if prior_ret < -0.08 and cons_range < 0.10:
            results.append(("Bearish Flag", "Bearish", 78.0))

    # Remove duplicate pattern names, keep best confidence.
    best = {}
    for name, direction, score in results:
        if name not in best or score > best[name][2]:
            best[name] = (name, direction, score)

    return sorted(best.values(), key=lambda z: z[2], reverse=True)


def scan_dataframe(df, symbol="UNKNOWN"):
    patterns = detect_patterns(df)
    return [
        {"Stock": symbol, "Pattern": p, "Direction": d, "Confidence": s}
        for p,d,s in patterns
    ]
