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

# Pattern-detection parameters. These are deliberately conservative: a stock
# should not be labelled with a structure merely because two regression lines
# happen to slope in the same direction.
RECENT_PIVOT_WINDOW = 12
DOUBLE_TOL_FLOOR = 0.018
DOUBLE_TOL_CAP = 0.055
DOUBLE_TOL_ATR_MULT = 1.10
DOUBLE_MIN_REACTION_ATR_MULT = 1.50
DOUBLE_MIN_REACTION_PCT = 0.025
DOUBLE_MIN_GAP = 6
DOUBLE_MAX_GAP = 90
HS_TOL_FLOOR = 0.020
HS_TOL_CAP = 0.065
HS_TOL_ATR_MULT = 1.20
HS_MIN_PROMINENCE_ATR_MULT = 1.50
HS_MIN_PROMINENCE_PCT = 0.025
CHANNEL_MIN_SLOPE_RATIO = 0.72
CHANNEL_MIN_R2 = 0.62
CHANNEL_MAX_WIDTH_CV = 0.45
CHANNEL_MIN_CONTAINMENT = 0.78
CHANNEL_TOUCH_TOL_ATR = 0.75
VOLUME_LOOKBACK = 20
VOLUME_BREAKOUT_MULT = 1.20


def pivots(df, left=3, right=3):
    if df is None or len(df) < left + right + 3:
        return [], []
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    H, L = [], []
    for i in range(left, len(df) - right):
        if h[i] >= np.max(h[i-left:i]) and h[i] >= np.max(h[i+1:i+right+1]):
            H.append(i)
        if l[i] <= np.min(l[i-left:i]) and l[i] <= np.min(l[i+1:i+right+1]):
            L.append(i)
    return H, L


def _clamp(x, lo=0.0, hi=100.0):
    return float(max(lo, min(hi, x)))


def _atr(df, period=14):
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    if len(df) < 2:
        return float(np.mean(h-l)) if len(df) else 0.0
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-min(period, len(tr)):]))


def _volume_ratio(df, idx, lookback=VOLUME_LOOKBACK):
    if df is None or "Volume" not in df.columns or idx is None:
        return None
    v = df["Volume"].to_numpy(float)
    idx = int(idx)
    if idx < 1 or idx >= len(v):
        return None
    start = max(0, idx - lookback)
    avg = np.nanmean(v[start:idx])
    if not np.isfinite(avg) or avg <= 0 or not np.isfinite(v[idx]):
        return None
    return float(v[idx] / avg)


def _breakout_volume_confirmed(df, entry_idx=None):
    idx = len(df) - 1 if entry_idx is None else int(entry_idx)
    ratio = _volume_ratio(df, idx)
    if ratio is None:
        return None, None
    return ratio >= VOLUME_BREAKOUT_MULT, ratio


def _double_candidate(df, side, piv):
    """Find the strongest genuine double top/bottom among recent swing pairs.

    A valid candidate needs similar extremes, a substantial reaction between
    them, and a real intervening neckline swing. Minor pivots are allowed, so a
    small wiggle does not hide the larger structure.
    """
    if len(piv) < 2:
        return None
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    atr = max(_atr(df), 1e-9)
    recent = piv[-RECENT_PIVOT_WINDOW:]
    candidates = []

    for ai in range(len(recent) - 2, -1, -1):
        for bi in range(ai + 1, len(recent)):
            a, b = recent[ai], recent[bi]
            gap = b - a
            if gap < DOUBLE_MIN_GAP or gap > min(DOUBLE_MAX_GAP, len(df) - 4):
                continue

            if side == "bottom":
                va, vb = float(l[a]), float(l[b])
                between = [i for i in range(a + 1, b) if i in set(piv)]
                if between:
                    neck_idx = max(between, key=lambda i: h[i])
                    neck = float(h[neck_idx])
                else:
                    neck_idx = a + 1 + int(np.argmax(h[a+1:b]))
                    neck = float(h[neck_idx])
                base = min(va, vb)
                reaction = neck - base
            else:
                va, vb = float(h[a]), float(h[b])
                between = [i for i in range(a + 1, b) if i in set(piv)]
                if between:
                    neck_idx = min(between, key=lambda i: l[i])
                    neck = float(l[neck_idx])
                else:
                    neck_idx = a + 1 + int(np.argmin(l[a+1:b]))
                    neck = float(l[neck_idx])
                base = max(va, vb)
                reaction = base - neck

            if not (a < neck_idx < b):
                continue

            mid = max(abs((va + vb) / 2.0), 1e-9)
            similarity = abs(va - vb) / mid
            tol = min(DOUBLE_TOL_CAP, max(DOUBLE_TOL_FLOOR, DOUBLE_TOL_ATR_MULT * atr / mid))
            if similarity > tol:
                continue

            min_reaction = max(DOUBLE_MIN_REACTION_ATR_MULT * atr,
                               DOUBLE_MIN_REACTION_PCT * mid)
            if reaction < min_reaction:
                continue

            # Prefer a second extreme that is recent, but not at the expense of
            # a materially cleaner/stronger older structure.
            recency = 1.0 - (len(df) - b) / max(len(df), 1)
            spacing = min(1.0, gap / 30.0)
            similarity_score = max(0.0, 1.0 - similarity / max(tol, 1e-9))
            reaction_score = min(1.0, reaction / max(4.0 * atr, 0.05 * abs(neck)))
            score = (0.42 * similarity_score + 0.30 * reaction_score +
                     0.13 * spacing + 0.15 * recency)
            candidates.append((score, {
                "first": int(a), "second": int(b), "neck_idx": int(neck_idx),
                "first_value": va, "second_value": vb, "neckline": neck,
                "height": reaction, "tolerance": tol, "score": score,
            }))

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def _hs_candidate(df, side, piv, piv_opp):
    if len(piv) < 3:
        return None
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    atr = max(_atr(df), 1e-9)
    recent = piv[-RECENT_PIVOT_WINDOW:]
    is_head = side == "head"
    vals = h if is_head else l
    candidates = []
    for li in range(len(recent) - 2):
        for hi in range(li + 1, len(recent) - 1):
            for ri in range(hi + 1, len(recent)):
                left, head, right = recent[li], recent[hi], recent[ri]
                gap = right - left
                if gap < 10 or gap > min(140, max(25, len(df) - 5)):
                    continue
                vl, vh, vr = float(vals[left]), float(vals[head]), float(vals[right])
                if is_head:
                    if not (vh > vl and vh > vr):
                        continue
                    prominence = vh - max(vl, vr)
                else:
                    if not (vh < vl and vh < vr):
                        continue
                    prominence = min(vl, vr) - vh
                mid = max(abs((vl + vr) / 2.0), 1e-9)
                similarity = abs(vl - vr) / mid
                tol = min(HS_TOL_CAP, max(HS_TOL_FLOOR, HS_TOL_ATR_MULT * atr / mid))
                if similarity > tol:
                    continue
                min_prom = max(HS_MIN_PROMINENCE_ATR_MULT * atr, HS_MIN_PROMINENCE_PCT * max(abs(vh), 1e-9))
                if prominence < min_prom:
                    continue
                opp_left = [p for p in piv_opp if left < p < head]
                opp_right = [p for p in piv_opp if head < p < right]
                if not opp_left or not opp_right:
                    continue
                n1_idx, n2_idx = opp_left[-1], opp_right[0]
                n1 = float(l[n1_idx] if is_head else h[n1_idx])
                n2 = float(l[n2_idx] if is_head else h[n2_idx])
                neckline = (n1 + n2) / 2.0
                recency = 1.0 - (len(df) - right) / max(len(df), 1)
                spacing = min(1.0, gap / 35.0)
                sim_score = max(0.0, 1.0 - similarity / max(tol, 1e-9))
                prom_score = min(1.0, prominence / max(4 * atr, 0.06 * abs(vh)))
                score = 0.40 * sim_score + 0.35 * prom_score + 0.13 * spacing + 0.12 * recency
                candidates.append((score, {
                    "left": int(left), "head": int(head), "right": int(right),
                    "left_value": vl, "head_value": vh, "right_value": vr,
                    "neck1_idx": int(n1_idx), "neck2_idx": int(n2_idx),
                    "neckline": neckline, "prominence": prominence,
                    "tolerance": tol, "score": score,
                }))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def _fit_line(values, indices):
    x = np.asarray(indices, float)
    y = np.asarray(values, float)
    if len(x) < 2:
        return None
    m, b = np.polyfit(x, y, 1)
    pred = m * x + b
    r = np.corrcoef(x, y)[0, 1] if np.std(x) and np.std(y) else 0.0
    r2 = float(r * r) if np.isfinite(r) else 0.0
    return float(m), float(b), r2


def _channel_candidate(df, H, L, kind):
    if len(H) < 3 or len(L) < 3:
        return None
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    hi, lo = H[-7:], L[-7:]
    uf = _fit_line(h[hi], hi)
    lf = _fit_line(l[lo], lo)
    if not uf or not lf:
        return None
    mu, bu, hi_r2 = uf
    ml, bl, lo_r2 = lf
    scale = max(float(np.mean(c[-min(50, len(c)): ])), 1e-9)
    nmu, nml = mu / scale, ml / scale
    if kind == "Rising Channel" and not (nmu > 0 and nml > 0):
        return None
    if kind == "Falling Channel" and not (nmu < 0 and nml < 0):
        return None
    slope_ratio = min(abs(nmu), abs(nml)) / max(abs(nmu), abs(nml), 1e-12)
    if slope_ratio < CHANNEL_MIN_SLOPE_RATIO:
        return None
    if hi_r2 < CHANNEL_MIN_R2 or lo_r2 < CHANNEL_MIN_R2:
        return None

    xs = np.arange(len(df), dtype=float)
    upper = mu * xs + bu
    lower = ml * xs + bl
    widths = upper[np.array(sorted(set(hi + lo)))] - lower[np.array(sorted(set(hi + lo)))]
    if np.any(widths <= 0):
        return None
    width_cv = float(np.std(widths) / max(np.mean(widths), 1e-9))
    if width_cv > CHANNEL_MAX_WIDTH_CV:
        return None

    atr = max(_atr(df), 1e-9)
    touch_tol = max(CHANNEL_TOUCH_TOL_ATR * atr, 0.0045 * c)
    hi_touch = sum(abs(h[i] - upper[i]) <= touch_tol[i] if hasattr(touch_tol, '__len__') else abs(h[i] - upper[i]) <= touch_tol for i in hi)
    lo_touch = sum(abs(l[i] - lower[i]) <= touch_tol[i] if hasattr(touch_tol, '__len__') else abs(l[i] - lower[i]) <= touch_tol for i in lo)
    if hi_touch < 3 or lo_touch < 3:
        return None

    start = max(0, len(df) - min(40, max(20, len(df) // 2)))
    band = max(1.0 * atr, 0.0045 * c[start:])
    inside = (c[start:] <= upper[start:] + band) & (c[start:] >= lower[start:] - band)
    containment = float(np.mean(inside)) if len(inside) else 0.0
    if containment < CHANNEL_MIN_CONTAINMENT:
        return None

    # A recent close beyond a boundary by >1 ATR means this is no longer an
    # active channel. It should disappear from scan results rather than being
    # relabelled as a fresh "forming" channel.
    recent_break = False
    for i in range(max(0, len(df) - 6), len(df)):
        if kind == "Rising Channel" and c[i] < lower[i] - 1.25 * atr:
            recent_break = True
        if kind == "Falling Channel" and c[i] > upper[i] + 1.25 * atr:
            recent_break = True
    if recent_break:
        return None

    return {
        "hi": hi, "lo": lo, "upper": upper, "lower": lower,
        "slope_ratio": slope_ratio, "hi_fit": hi_r2, "lo_fit": lo_r2,
        "width_cv": width_cv, "hi_touch": hi_touch, "lo_touch": lo_touch,
        "containment": containment, "recent_break": False,
    }


def pattern_confidence(df, pattern, direction, H=None, L=None):
    if pattern not in ALLOWED_PATTERNS or df is None or len(df) == 0:
        return 0.0
    if H is None or L is None:
        H, L = pivots(df)
    quality = 0.0
    event_idx = None

    if pattern in ("Double Top", "Double Bottom"):
        cand = _double_candidate(df, "top" if pattern == "Double Top" else "bottom", H if pattern == "Double Top" else L)
        if not cand:
            return 0.0
        mid = max(abs((cand["first_value"] + cand["second_value"]) / 2), 1e-9)
        similarity = abs(cand["first_value"] - cand["second_value"]) / mid
        sim_score = _clamp(100 * (1 - similarity / max(cand["tolerance"], 1e-9)))
        reaction_score = _clamp(cand["height"] / max(4 * _atr(df), 0.05 * abs(cand["neckline"])) * 100)
        spacing_score = _clamp(abs(cand["second"] - cand["first"]) / 30 * 100)
        recency_score = _clamp(100 * (1 - (len(df) - cand["second"]) / max(len(df), 1)))
        quality = .45 * sim_score + .30 * reaction_score + .10 * spacing_score + .15 * recency_score
        event_idx = cand["second"]

    elif pattern in ("Head and Shoulders", "Inverse Head and Shoulders"):
        cand = _hs_candidate(df, "head" if pattern == "Head and Shoulders" else "inverse",
                             H if pattern == "Head and Shoulders" else L,
                             L if pattern == "Head and Shoulders" else H)
        if not cand:
            return 0.0
        mid = max(abs((cand["left_value"] + cand["right_value"]) / 2), 1e-9)
        similarity = abs(cand["left_value"] - cand["right_value"]) / mid
        sim_score = _clamp(100 * (1 - similarity / max(cand["tolerance"], 1e-9)))
        prom_score = _clamp(cand["prominence"] / max(4 * _atr(df), 0.06 * abs(cand["head_value"])) * 100)
        spacing_score = _clamp((cand["right"] - cand["left"]) / 35 * 100)
        recency_score = _clamp(100 * (1 - (len(df) - cand["right"]) / max(len(df), 1)))
        quality = .43 * sim_score + .38 * prom_score + .08 * spacing_score + .11 * recency_score
        event_idx = cand["right"]

    elif pattern in ("Rising Channel", "Falling Channel"):
        cand = _channel_candidate(df, H, L, pattern)
        if not cand:
            return 0.0
        quality = (
            .25 * cand["hi_fit"] * 100 + .25 * cand["lo_fit"] * 100 +
            .20 * cand["slope_ratio"] * 100 +
            .15 * max(0, 1 - cand["width_cv"]) * 100 +
            .15 * cand["containment"] * 100
        )

    elif pattern in ("Bullish Flag", "Bearish Flag") and len(df) >= 25:
        c = df["Close"].to_numpy(float)
        p = c[-25:-10]
        q = c[-10:]
        impulse = abs(p[-1] / p[0] - 1) if p[0] else 0
        consolidation = (q.max() - q.min()) / max(abs(q.mean()), 1e-9)
        if impulse < 0.12 or consolidation >= 0.08:
            return 0.0
        quality = .55 * _clamp(impulse * 450) + .45 * _clamp(100 - consolidation * 700)
        event_idx = len(df) - 1
    else:
        return 0.0

    # Secondary adjustments are intentionally small; pattern geometry drives
    # the score rather than a fixed confidence value.
    vol_bonus = 0.0
    confirmed, _ = _breakout_volume_confirmed(df, event_idx)
    if confirmed is True:
        vol_bonus = 4.0
    elif confirmed is False:
        vol_bonus = -4.0

    return round(_clamp(.86 * quality + vol_bonus), 1)


def geometry(df):
    if df is None or len(df) == 0:
        return [], 0.0, [], []
    H, L = pivots(df)
    found = []
    checks = [
        ("Double Bottom", "Bullish", _double_candidate(df, "bottom", L)),
        ("Double Top", "Bearish", _double_candidate(df, "top", H)),
        ("Head and Shoulders", "Bearish", _hs_candidate(df, "head", H, L)),
        ("Inverse Head and Shoulders", "Bullish", _hs_candidate(df, "inverse", L, H)),
        ("Rising Channel", "Bullish", _channel_candidate(df, H, L, "Rising Channel")),
        ("Falling Channel", "Bearish", _channel_candidate(df, H, L, "Falling Channel")),
    ]
    for pattern, direction, candidate in checks:
        if candidate is not None:
            found.append((pattern, direction))

    if len(df) >= 25:
        c = df["Close"].to_numpy(float)
        p, q = c[-25:-10], c[-10:]
        imp = p[-1] / p[0] - 1 if p[0] else 0
        consolidation = (q.max() - q.min()) / max(abs(q.mean()), 1e-9)
        if abs(imp) >= .12 and consolidation < .08:
            found.append(("Bullish Flag" if imp > 0 else "Bearish Flag",
                          "Bullish" if imp > 0 else "Bearish"))

    found = list(dict.fromkeys(x for x in found if x[0] in ALLOWED_PATTERNS))
    scores = [pattern_confidence(df, p, d, H, L) for p, d in found]
    return found, round(max(scores), 1) if scores else 0.0, H, L


def pattern_points(df, pattern, direction, H=None, L=None):
    if pattern not in ALLOWED_PATTERNS or df is None or len(df) == 0:
        return None
    if H is None or L is None:
        H, L = pivots(df)
    if pattern in ("Double Top", "Double Bottom"):
        cand = _double_candidate(df, "top" if pattern == "Double Top" else "bottom", H if pattern == "Double Top" else L)
        if not cand:
            return None
        return {"type": "double", "indices": [cand["first"], cand["second"]],
                "values": [cand["first_value"], cand["second_value"]],
                "neck_idx": cand["neck_idx"], "neckline": cand["neckline"]}
    if pattern in ("Head and Shoulders", "Inverse Head and Shoulders"):
        cand = _hs_candidate(df, "head" if pattern == "Head and Shoulders" else "inverse",
                             H if pattern == "Head and Shoulders" else L,
                             L if pattern == "Head and Shoulders" else H)
        if not cand:
            return None
        return {"type": "hs", "indices": [cand["left"], cand["head"], cand["right"]],
                "values": [cand["left_value"], cand["head_value"], cand["right_value"]],
                "neck_idx": [cand["neck1_idx"], cand["neck2_idx"]], "neckline": cand["neckline"]}
    if pattern in ("Rising Channel", "Falling Channel"):
        cand = _channel_candidate(df, H, L, pattern)
        if not cand:
            return None
        return {"type": "channel", "hi": cand["hi"], "lo": cand["lo"]}
    if pattern in ("Bullish Flag", "Bearish Flag") and len(df) >= 25:
        return {"type": "flag", "pole_start": len(df)-25, "pole_end": len(df)-10,
                "consolidation_start": len(df)-10, "consolidation_end": len(df)-1}
    return None


def pattern_target(df, pattern, direction, H=None, L=None):
    if pattern not in ALLOWED_PATTERNS or df is None or len(df) == 0:
        return None
    if H is None or L is None:
        H, L = pivots(df)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = float(df["Close"].iloc[-1])
    entry = c
    stop = None
    t1 = None
    t2 = None
    method = "Recent range"

    if pattern == "Double Bottom":
        cand = _double_candidate(df, "bottom", L)
        if cand:
            base = min(cand["first_value"], cand["second_value"])
            height = max(cand["neckline"] - base, 0)
            entry, stop = cand["neckline"], base
            t1, t2 = cand["neckline"] + height, cand["neckline"] + 1.618 * height
            method = "Neckline + pattern height"
    elif pattern == "Double Top":
        cand = _double_candidate(df, "top", H)
        if cand:
            top = max(cand["first_value"], cand["second_value"])
            height = max(top - cand["neckline"], 0)
            entry, stop = cand["neckline"], top
            t1, t2 = cand["neckline"] - height, cand["neckline"] - 1.618 * height
            method = "Neckline - pattern height"
    elif pattern == "Head and Shoulders":
        cand = _hs_candidate(df, "head", H, L)
        if cand:
            height = max(cand["head_value"] - cand["neckline"], 0)
            entry, stop = cand["neckline"], cand["head_value"]
            t1, t2 = cand["neckline"] - height, cand["neckline"] - 1.618 * height
            method = "Head-to-neckline measured move"
    elif pattern == "Inverse Head and Shoulders":
        cand = _hs_candidate(df, "inverse", L, H)
        if cand:
            height = max(cand["neckline"] - cand["head_value"], 0)
            entry, stop = cand["neckline"], cand["head_value"]
            t1, t2 = cand["neckline"] + height, cand["neckline"] + 1.618 * height
            method = "Head-to-neckline measured move"
    elif pattern in ("Bullish Flag", "Bearish Flag") and len(df) >= 25:
        pole = float(max(h[-25:-10]) - min(l[-25:-10]))
        if direction == "Bullish":
            entry, stop = c, float(min(l[-10:]))
            t1, t2 = c + pole, c + 1.618 * pole
        else:
            entry, stop = c, float(max(h[-10:]))
            t1, t2 = c - pole, c - 1.618 * pole
        method = "Measured prior impulse"
    elif pattern in ("Rising Channel", "Falling Channel"):
        ch = _channel_candidate(df, H, L, pattern)
        if ch:
            res = float(np.mean(ch["upper"][-3:]))
            sup = float(np.mean(ch["lower"][-3:]))
            if direction == "Bullish":
                entry, stop, t1 = c, sup, res
                t2 = res + (res - sup) * .618
            else:
                entry, stop, t1 = c, res, sup
                t2 = sup - (res - sup) * .618
            method = "Channel boundary projection"

    if t1 is None:
        rng = float(max(h[-min(20, len(h)):]) - min(l[-min(20, len(l)):]))
        if direction == "Bullish":
            stop, t1, t2 = float(min(l[-min(20, len(l)): ])), c + rng, c + 1.618 * rng
        else:
            stop, t1, t2 = float(max(h[-min(20, len(h)): ])), c - rng, c - 1.618 * rng
        method = "Recent 20-candle range fallback"

    volume_confirmed, volume_ratio = _breakout_volume_confirmed(df)
    return {"current": c, "entry": float(entry), "stop": None if stop is None else float(stop),
            "target1": None if t1 is None else float(t1), "target2": None if t2 is None else float(t2),
            "method": method, "volume_confirmed": volume_confirmed, "volume_ratio": volume_ratio}


def pattern_status(df, pattern, direction, H=None, L=None):
    if pattern not in ALLOWED_PATTERNS or df is None or len(df) == 0:
        return "Forming"
    if H is None or L is None:
        H, L = pivots(df)

    if pattern in ("Rising Channel", "Falling Channel"):
        if _channel_candidate(df, H, L, pattern) is None:
            return "Forming"
        return "Formed/Active"

    details = pattern_target(df, pattern, direction, H, L)
    if not details or details.get("entry") is None:
        return "Forming"
    c = float(df["Close"].iloc[-1])
    e = float(details["entry"])

    # For doubles and H&S, the neckline is the activation level. A close on the
    # correct side means the structure is formed/active; otherwise it is forming.
    if direction == "Bullish":
        return "Formed/Active" if c >= e else "Forming"
    if direction == "Bearish":
        return "Formed/Active" if c <= e else "Forming"
    return "Forming"
