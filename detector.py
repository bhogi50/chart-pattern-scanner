import numpy as np
import pandas as pd


def _regression(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0, 0.0, 0.0
    m, b = np.polyfit(x, y, 1)
    pred = m * x + b
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(m), float(b), float(np.clip(r2, 0, 1))


def find_pivots(df, left=3, right=3):
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    highs, lows = [], []

    for i in range(left, len(df) - right):
        window_h = high[i-left:i+right+1]
        window_l = low[i-left:i+right+1]
        if high[i] >= window_h.max():
            highs.append(i)
        if low[i] <= window_l.min():
            lows.append(i)

    return highs, lows


def _confidence(r2_high, r2_low, convergence, touches):
    value = (
        0.35 * r2_high
        + 0.35 * r2_low
        + 0.20 * np.clip(convergence, 0, 1)
        + 0.10 * np.clip(touches / 8, 0, 1)
    )
    return round(float(np.clip(value * 100, 0, 99)), 1)


def detect_patterns(df, lookback=120):
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        return []

    df = df.copy().dropna(subset=list(required)).tail(lookback)
    if len(df) < 45:
        return []

    highs, lows = find_pivots(df)
    if len(highs) < 2 or len(lows) < 2:
        return []

    xh = np.asarray(highs[-5:], dtype=float)
    yh = df["High"].iloc[highs[-5:]].to_numpy(float)
    xl = np.asarray(lows[-5:], dtype=float)
    yl = df["Low"].iloc[lows[-5:]].to_numpy(float)

    mh, bh, r2h = _regression(xh, yh)
    ml, bl, r2l = _regression(xl, yl)

    price_scale = max(float(df["Close"].mean()), 1e-9)
    high_slope = mh / price_scale
    low_slope = ml / price_scale

    start = max(0, len(df) - 40)
    end = len(df) - 1

    width_start = (mh * start + bh) - (ml * start + bl)
    width_end = (mh * end + bh) - (ml * end + bl)
    convergence = (
        (width_start - width_end) / abs(width_start)
        if abs(width_start) > 1e-9 else 0.0
    )

    touches = len(xh) + len(xl)
    results = []

    def add(name, direction, confidence):
        results.append({
            "Pattern": name,
            "Direction": direction,
            "Confidence": round(float(confidence), 1),
        })

    if high_slope < -0.0007 and low_slope > 0.0007 and convergence > 0.12:
        add("Symmetrical Triangle", "Neutral",
            _confidence(r2h, r2l, convergence, touches))

    if abs(high_slope) < 0.0007 and low_slope > 0.0007 and convergence > 0.10:
        add("Ascending Triangle", "Bullish",
            _confidence(r2h, r2l, convergence, touches))

    if high_slope < -0.0007 and abs(low_slope) < 0.0007 and convergence > 0.10:
        add("Descending Triangle", "Bearish",
            _confidence(r2h, r2l, convergence, touches))

    if high_slope > 0.0004 and low_slope > 0.0004:
        if high_slope < low_slope and convergence > 0.08:
            add("Rising Wedge", "Bearish",
                _confidence(r2h, r2l, convergence, touches))

    if high_slope < -0.0004 and low_slope < -0.0004:
        if high_slope > low_slope and convergence > 0.08:
            add("Falling Wedge", "Bullish",
                _confidence(r2h, r2l, convergence, touches))

    if abs(high_slope - low_slope) < 0.0007 and abs(high_slope) > 0.00035:
        add(
            "Rising Channel" if high_slope > 0 else "Falling Channel",
            "Bullish" if high_slope > 0 else "Bearish",
            _confidence(r2h, r2l, 0.5, touches),
        )

    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        va = float(df["High"].iloc[a])
        vb = float(df["High"].iloc[b])
        similarity = abs(va - vb) / max(va, vb)
        if b - a >= 5 and similarity < 0.025:
            add("Double Top", "Bearish", 78 + 15 * (1 - similarity / 0.025))

    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        va = float(df["Low"].iloc[a])
        vb = float(df["Low"].iloc[b])
        similarity = abs(va - vb) / max(va, vb)
        if b - a >= 5 and similarity < 0.025:
            add("Double Bottom", "Bullish", 78 + 15 * (1 - similarity / 0.025))

    if len(df) > 32:
        prior = df["Close"].iloc[-30:-20]
        consolidation = df["Close"].iloc[-20:]
        prior_return = float(prior.iloc[-1] / prior.iloc[0] - 1)
        consolidation_range = float(
            (consolidation.max() - consolidation.min())
            / max(consolidation.mean(), 1e-9)
        )

        if prior_return > 0.08 and consolidation_range < 0.10:
            add("Bullish Flag", "Bullish", 78)

        if prior_return < -0.08 and consolidation_range < 0.10:
            add("Bearish Flag", "Bearish", 78)

    best = {}
    for result in results:
        key = result["Pattern"]
        if key not in best or result["Confidence"] > best[key]["Confidence"]:
            best[key] = result

    return sorted(best.values(), key=lambda x: x["Confidence"], reverse=True)
