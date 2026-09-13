"""Human-like chart structure and pattern detection engine.

Pipeline
--------
1. Extract meaningful swing highs/lows from OHLC.
2. Build price-frequency support/resistance zones (R1-R3 / S1-S3).
3. Classify the current geometry: Channel Up, Channel Down,
   Ascending Triangle, Descending Triangle, Flat/Range, or Unclear.
4. Locate the current candle relative to the zones and structure boundaries.
5. Only then evaluate relevant Double Top / Double Bottom candidates.

H&S / inverse H&S are intentionally not part of this version.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Swing:
    idx: int
    date: object
    price: float
    kind: str  # H / L


@dataclass
class Zone:
    price: float
    low: float
    high: float
    touches: int
    strength: float
    side: str
    indices: list = field(default_factory=list)

    def as_dict(self):
        return {
            "price": round(float(self.price), 2),
            "low": round(float(self.low), 2),
            "high": round(float(self.high), 2),
            "touches": int(self.touches),
            "strength": round(float(self.strength), 1),
            "side": self.side,
            "indices": list(self.indices),
        }


@dataclass
class Structure:
    name: str
    confidence: float
    upper_line: tuple = None
    lower_line: tuple = None
    upper_slope_pct_bar: float = 0.0
    lower_slope_pct_bar: float = 0.0
    high_trend: str = "mixed"
    low_trend: str = "mixed"
    note: str = ""


@dataclass
class Location:
    zone: str
    zone_price: float = None
    structure_position: str = "Unknown"
    candle_behavior: str = "Neutral"
    distance_pct: float = None
    note: str = ""


@dataclass
class PatternMatch:
    name: str
    direction: str
    confidence: float
    points: list
    neckline: tuple = None
    note: str = ""
    status: str = ""
    bars_since_breakout: int = None
    structure: str = "Unclear"
    location: str = "Unknown"
    structure_confidence: float = 0.0
    resistance: list = field(default_factory=list)
    support: list = field(default_factory=list)
    higher_tf: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    upper_line: tuple = None
    lower_line: tuple = None


# ---------------------------------------------------------------------------
# 1. Swings
# ---------------------------------------------------------------------------

def _auto_threshold(df: pd.DataFrame) -> float:
    if len(df) < 15:
        return 3.0
    h, l, c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    atr_pct = (tr.rolling(14).mean().iloc[-1] / c.iloc[-1]) * 100
    if pd.isna(atr_pct) or atr_pct <= 0:
        return 3.0
    return float(np.clip(atr_pct * 1.6, 1.5, 6.0))


def find_swings(df: pd.DataFrame, pct_threshold: float = None):
    """Extract alternating pivots using close reversal, then use candle H/L at pivots."""
    if len(df) < 5:
        return []
    if pct_threshold is None:
        pct_threshold = _auto_threshold(df)

    close = df["Close"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    dates = df.index
    swings = []
    trend = None
    anchor_idx = 0
    anchor_price = close[0]

    for i in range(1, len(df)):
        price = close[i]
        change = (price-anchor_price) / anchor_price * 100
        if trend is None:
            if change >= pct_threshold:
                swings.append(Swing(anchor_idx, dates[anchor_idx], lows[anchor_idx], "L"))
                trend, anchor_idx, anchor_price = "up", i, price
            elif change <= -pct_threshold:
                swings.append(Swing(anchor_idx, dates[anchor_idx], highs[anchor_idx], "H"))
                trend, anchor_idx, anchor_price = "down", i, price
        elif trend == "up":
            if price >= anchor_price:
                anchor_idx, anchor_price = i, price
            elif (anchor_price-price)/anchor_price*100 >= pct_threshold:
                swings.append(Swing(anchor_idx, dates[anchor_idx], highs[anchor_idx], "H"))
                trend, anchor_idx, anchor_price = "down", i, price
        else:
            if price <= anchor_price:
                anchor_idx, anchor_price = i, price
            elif (price-anchor_price)/anchor_price*100 >= pct_threshold:
                swings.append(Swing(anchor_idx, dates[anchor_idx], lows[anchor_idx], "L"))
                trend, anchor_idx, anchor_price = "up", i, price

    if trend == "up":
        swings.append(Swing(anchor_idx, dates[anchor_idx], highs[anchor_idx], "H"))
    elif trend == "down":
        swings.append(Swing(anchor_idx, dates[anchor_idx], lows[anchor_idx], "L"))
    return swings


def _pct_diff(a, b):
    den = (abs(a)+abs(b))/2
    return abs(a-b)/den*100 if den else 0.0


def _score(diff, tolerance):
    return float(np.clip(100*(1-diff/max(tolerance, 1e-9)), 0, 100))


# ---------------------------------------------------------------------------
# 2. Price-frequency support/resistance
# ---------------------------------------------------------------------------

def _cluster_values(values, indices, tol_pct):
    pairs = sorted(zip(values, indices), key=lambda x: x[0])
    clusters = []
    for value, idx in pairs:
        target = None
        for c in clusters:
            if _pct_diff(value, c["center"]) <= tol_pct:
                target = c
                break
        if target is None:
            target = {"values": [], "indices": [], "center": float(value)}
            clusters.append(target)
        target["values"].append(float(value))
        target["indices"].append(int(idx))
        target["center"] = float(np.mean(target["values"]))
    return clusters


def build_zones(df, swings=None, side="resistance", max_zones=3, cluster_tol_pct=None):
    """Build zones from ALL candle highs/lows, weighted by frequency.

    The user-facing R/S levels are selected from price clusters rather than
    simply taking the absolute highest/lowest candle.
    """
    if df.empty:
        return []
    current = float(df["Close"].iloc[-1])
    series = df["High"] if side == "resistance" else df["Low"]
    values = series.to_numpy(float)
    n = len(values)
    if cluster_tol_pct is None:
        atr = float((df["High"]-df["Low"]).rolling(14).mean().iloc[-1]) if n >= 14 else float((df["High"]-df["Low"]).mean())
        cluster_tol_pct = float(np.clip(max(current*0.003, atr*0.55) / max(current, 1e-9) * 100, 0.35, 1.5))

    clusters = _cluster_values(values, range(n), cluster_tol_pct)
    zones = []
    for c in clusters:
        center = c["center"]
        if side == "resistance" and center <= current:
            continue
        if side == "support" and center >= current:
            continue
        touches = len(c["values"])
        recency = max(c["indices"])/max(n-1,1)
        # Frequency dominates. Distinct candles and recent activity add modestly.
        frequency_score = min(70.0, touches / max(n,1) * 100 * 1.8)
        recurrence_score = min(20.0, touches * 1.8)
        recency_score = recency * 10.0
        strength = float(np.clip(frequency_score + recurrence_score + recency_score, 0, 100))
        half_width = max(center*cluster_tol_pct/100, (max(c["values"])-min(c["values"]))/2)
        zones.append(Zone(center, center-half_width, center+half_width, touches, strength, side, c["indices"]))

    # First nearest level, then progressively farther levels. Within a
    # comparable distance, frequency/strength wins.
    if side == "resistance":
        zones.sort(key=lambda z: (abs(z.price-current)/current > 0.12, abs(z.price-current)/current, -z.strength))
    else:
        zones.sort(key=lambda z: (abs(z.price-current)/current > 0.12, abs(z.price-current)/current, -z.strength))
    return [z.as_dict() for z in zones[:max_zones]]


def find_support_resistance(df, swings=None, pct_threshold=None, cluster_tol_pct=None, top_n=3):
    return (
        build_zones(df, swings, "support", top_n, cluster_tol_pct),
        build_zones(df, swings, "resistance", top_n, cluster_tol_pct),
    )


# ---------------------------------------------------------------------------
# 3. Structure geometry
# ---------------------------------------------------------------------------

def _linear_fit(points):
    if len(points) < 2:
        return None
    x = np.array([p.idx for p in points], dtype=float)
    y = np.array([p.price for p in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    mean = max(float(np.mean(y)), 1e-9)
    slope_pct = slope / mean * 100
    pred = slope*x + intercept
    residual = float(np.mean(np.abs(y-pred)) / mean * 100)
    r2 = 1 - float(np.sum((y-pred)**2) / max(np.sum((y-y.mean())**2), 1e-9))
    return slope, intercept, slope_pct, residual, float(np.clip(r2, 0, 1))


def _trend_label(points, flat_tol_pct_bar):
    fit = _linear_fit(points)
    if not fit:
        return "mixed", 0.0
    slope_pct = fit[2]
    if abs(slope_pct) <= flat_tol_pct_bar:
        return "flat", slope_pct
    return ("rising" if slope_pct > 0 else "falling"), slope_pct


def _line_segment(df, fit, points):
    if not fit or not points:
        return None
    slope, intercept = fit[0], fit[1]
    a, b = min(p.idx for p in points), len(df)-1
    return ((df.index[a], slope*a+intercept), (df.index[b], slope*b+intercept))


def identify_structure(df, swings=None):
    """Classify geometry using recent meaningful swing highs/lows.

    Flat + rising/falling = triangle. Similar signed slopes = channel.
    Flat + flat = range. Anything else = unclear.
    """
    if swings is None:
        swings = find_swings(df)
    # Recent structure is more representative, but retain enough swings to see shape.
    recent = [s for s in swings if s.idx >= max(0, len(df)-100)]
    highs = [s for s in recent if s.kind == "H"]
    lows = [s for s in recent if s.kind == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return Structure("Unclear", 0, note="Not enough meaningful swing points")

    all_prices = np.array([s.price for s in recent])
    avg = max(float(np.mean(all_prices)), 1e-9)
    # Flat threshold is deliberately conservative; it is normalized by price and bars.
    flat_tol = max(0.015, min(0.12, 0.75 * (float(np.std(np.diff(all_prices))) / avg * 100 if len(all_prices)>3 else 0.03)))
    # Keep a minimum practical threshold so ordinary small drift can be flat.
    flat_tol = max(flat_tol, 0.025)

    hf = _linear_fit(highs)
    lf = _linear_fit(lows)
    ht, hs = _trend_label(highs, flat_tol)
    lt, ls = _trend_label(lows, flat_tol)

    # Similar slopes = channel. Require same direction and reasonable slope ratio.
    name = "Unclear"
    base = 0.0
    if ht == "rising" and lt == "rising":
        ratio = min(abs(hs), abs(ls)) / max(abs(hs), abs(ls), 1e-9)
        if ratio >= 0.35:
            name, base = "Channel Up", 65 + 25*ratio
    elif ht == "falling" and lt == "falling":
        ratio = min(abs(hs), abs(ls)) / max(abs(hs), abs(ls), 1e-9)
        if ratio >= 0.35:
            name, base = "Channel Down", 65 + 25*ratio
    elif ht == "flat" and lt == "rising":
        name, base = "Ascending Triangle", 72
    elif ht == "falling" and lt == "flat":
        name, base = "Descending Triangle", 72
    elif ht == "flat" and lt == "flat":
        name, base = "Flat / Range", 70

    if name == "Unclear":
        return Structure(name, 0, high_trend=ht, low_trend=lt,
                         upper_slope_pct_bar=hs, lower_slope_pct_bar=ls,
                         note=f"Highs {ht}, lows {lt}")

    fit_quality = ((hf[4] if hf else 0) + (lf[4] if lf else 0))/2
    touch_score = min(1.0, (len(highs)+len(lows))/10)
    confidence = float(np.clip(base*0.7 + fit_quality*20 + touch_score*10, 0, 100))
    return Structure(
        name=name, confidence=round(confidence,1),
        upper_line=_line_segment(df, hf, highs), lower_line=_line_segment(df, lf, lows),
        upper_slope_pct_bar=hs, lower_slope_pct_bar=ls,
        high_trend=ht, low_trend=lt,
        note=f"Highs: {ht} ({hs:.3f}%/bar); Lows: {lt} ({ls:.3f}%/bar); touches H{len(highs)}/L{len(lows)}",
    )


# ---------------------------------------------------------------------------
# 4. Current location and candle behavior
# ---------------------------------------------------------------------------

def _line_value(line, idx):
    if not line:
        return None
    (d1,p1),(d2,p2) = line
    x1 = pd.Timestamp(d1).value
    x2 = pd.Timestamp(d2).value
    x = pd.Timestamp(idx).value
    if x2 == x1:
        return float(p1)
    return float(p1 + (p2-p1)*(x-x1)/(x2-x1))


def locate_current(df, structure, supports, resistances):
    row = df.iloc[-1]
    close = float(row.Close); high = float(row.High); low = float(row.Low); op = float(row.Open)
    candidates = []
    for side, zones in (("R", resistances), ("S", supports)):
        for z in zones:
            # Distance to zone, using zero when candle overlaps the zone.
            if low <= z["high"] and high >= z["low"]:
                dist = 0.0
            else:
                dist = min(abs(close-z["low"]), abs(close-z["high"])) / close * 100
            candidates.append((dist, side, z))
    nearest = min(candidates, key=lambda x: x[0]) if candidates else None

    upper = _line_value(structure.upper_line, df.index[-1])
    lower = _line_value(structure.lower_line, df.index[-1])
    structure_pos = "Middle"
    if upper is not None and lower is not None and upper > lower:
        span = upper-lower
        p = (close-lower)/span
        structure_pos = "Near lower boundary" if p <= .25 else "Near upper boundary" if p >= .75 else "Middle"

    if op != 0:
        body = close-op
        if high > op and high > close and high-close >= abs(body)*1.2:
            candle_behavior = "Upper rejection"
        elif low < op and low < close and close-low >= abs(body)*1.2:
            candle_behavior = "Lower rejection"
        elif close > op:
            candle_behavior = "Bullish close"
        elif close < op:
            candle_behavior = "Bearish close"
        else:
            candle_behavior = "Neutral"
    else:
        candle_behavior = "Neutral"

    if nearest:
        dist, side, z = nearest
        zone = f"Near {side}{zones_rank(z, supports, resistances)}"
        return Location(zone, z["price"], structure_pos, candle_behavior, round(dist,2),
                        f"Current close {close:.2f}; nearest {side} zone {z['price']:.2f} ({dist:.2f}%)")
    return Location("No nearby R/S", None, structure_pos, candle_behavior, None, "No relevant R/S zone")


def zones_rank(z, supports, resistances):
    for i, x in enumerate(resistances, 1):
        if x is z or x["price"] == z["price"]:
            return i
    for i, x in enumerate(supports, 1):
        if x is z or x["price"] == z["price"]:
            return i
    return "?"


# ---------------------------------------------------------------------------
# 5. Pattern candidates — activated by structure + location
# ---------------------------------------------------------------------------

def _find_breakout_idx(df, start_idx, level, direction):
    close = df["Close"].to_numpy(float)
    for i in range(start_idx, len(close)):
        if direction == "below" and close[i] < level:
            return i
        if direction == "above" and close[i] > level:
            return i
    return None


def _status(df, breakout_idx):
    if breakout_idx is None:
        return "Developing", None
    bars = len(df)-1-breakout_idx
    return ("Confirmed" if bars <= max(8, len(df)//10) else "Completed (historical)"), bars


def _candidate_location_ok(structure_name, location, pattern):
    z = location.zone
    pos = location.structure_position
    if pattern == "Double Bottom":
        return ("S" in z) or pos == "Near lower boundary" or structure_name in {"Ascending Triangle", "Flat / Range", "Channel Down"}
    return ("R" in z) or pos == "Near upper boundary" or structure_name in {"Descending Triangle", "Flat / Range", "Channel Up"}


def detect_double_top(df, swings, location, structure):
    if not _candidate_location_ok(structure.name, location, "Double Top"):
        return None
    for i in range(max(0, len(swings)-8), len(swings)-2):
        seq = swings[i:i+3]
        if [s.kind for s in seq] != ["H","L","H"]:
            continue
        h1, trough, h2 = seq
        peak_diff = _pct_diff(h1.price,h2.price)
        if peak_diff > 2.5:
            continue
        depth = (min(h1.price,h2.price)-trough.price)/((h1.price+h2.price)/2)*100
        if depth < 1.5:
            continue
        score = 55 + _score(peak_diff,2.5)*0.25 + min(depth/5,1)*20
        neckline = trough.price
        br = _find_breakout_idx(df,h2.idx,neckline,"below")
        status,bars = _status(df,br)
        if status == "Completed (historical)":
            continue
        if br is not None: score += 15
        note = f"Two highs near {((h1.price+h2.price)/2):.2f}; neckline {neckline:.2f}"
        if br is None: note += "; waiting for neckline break"
        return PatternMatch("Double Top","Bearish",round(float(np.clip(score,0,100)),1),
            [(h1.date,h1.price),(trough.date,trough.price),(h2.date,h2.price)],
            neckline=((trough.date,neckline),(df.index[-1],neckline)),note=note,status=status,bars_since_breakout=bars,
            structure=structure.name,location=location.zone,structure_confidence=structure.confidence)
    return None


def detect_double_bottom(df, swings, location, structure):
    if not _candidate_location_ok(structure.name, location, "Double Bottom"):
        return None
    for i in range(max(0, len(swings)-8), len(swings)-2):
        seq = swings[i:i+3]
        if [s.kind for s in seq] != ["L","H","L"]:
            continue
        l1, peak, l2 = seq
        low_diff = _pct_diff(l1.price,l2.price)
        if low_diff > 2.5:
            continue
        height = (peak.price-max(l1.price,l2.price))/((l1.price+l2.price)/2)*100
        if height < 1.5:
            continue
        score = 55 + _score(low_diff,2.5)*0.25 + min(height/5,1)*20
        neckline = peak.price
        br = _find_breakout_idx(df,l2.idx,neckline,"above")
        status,bars = _status(df,br)
        if status == "Completed (historical)":
            continue
        if br is not None: score += 15
        note = f"Two lows near {((l1.price+l2.price)/2):.2f}; neckline {neckline:.2f}"
        if br is None: note += "; waiting for neckline break"
        return PatternMatch("Double Bottom","Bullish",round(float(np.clip(score,0,100)),1),
            [(l1.date,l1.price),(peak.date,peak.price),(l2.date,l2.price)],
            neckline=((peak.date,neckline),(df.index[-1],neckline)),note=note,status=status,bars_since_breakout=bars,
            structure=structure.name,location=location.zone,structure_confidence=structure.confidence)
    return None


def _structure_match(structure):
    if structure.name == "Channel Up":
        return PatternMatch("Channel Up","Bullish",round(structure.confidence,1),[],note=structure.note,status="Active",
                            structure=structure.name,structure_confidence=structure.confidence,upper_line=structure.upper_line,lower_line=structure.lower_line)
    if structure.name == "Channel Down":
        return PatternMatch("Channel Down","Bearish",round(structure.confidence,1),[],note=structure.note,status="Active",
                            structure=structure.name,structure_confidence=structure.confidence,upper_line=structure.upper_line,lower_line=structure.lower_line)
    if structure.name in {"Ascending Triangle","Descending Triangle","Flat / Range"}:
        return PatternMatch(structure.name,"Neutral",round(structure.confidence,1),[],note=structure.note,status="Active",
                            structure=structure.name,structure_confidence=structure.confidence,upper_line=structure.upper_line,lower_line=structure.lower_line)
    return None


def analyze_chart(df: pd.DataFrame, higher_tf: dict = None, pct_threshold=None):
    """Run the complete human-like pipeline and return a structured analysis."""
    if df.empty or len(df) < 20:
        return {"swings": [], "supports": [], "resistances": [], "structure": Structure("Unclear",0),
                "location": Location("Unknown"), "patterns": [], "higher_tf": higher_tf or {}}
    swings = find_swings(df, pct_threshold)
    supports, resistances = find_support_resistance(df, swings, pct_threshold, top_n=3)
    structure = identify_structure(df, swings)
    location = locate_current(df, structure, supports, resistances)

    # Structure is always a result; reversal patterns are selectively activated.
    patterns = []
    structural = _structure_match(structure)
    if structural:
        structural.location = location.zone
        structural.resistance = resistances
        structural.support = supports
        structural.higher_tf = higher_tf or {}
        patterns.append(structural)

    dt = detect_double_top(df, swings, location, structure)
    db = detect_double_bottom(df, swings, location, structure)
    for m in (dt, db):
        if m:
            m.resistance, m.support, m.higher_tf = resistances, supports, higher_tf or {}
            # Location/structure are contextual evidence, not a replacement for geometry.
            m.confidence = round(float(np.clip(m.confidence + structure.confidence*0.12, 0, 100)),1)
            patterns.append(m)

    # Higher-timeframe directional context modestly adjusts reversal confidence.
    if higher_tf:
        hdir = higher_tf.get("direction")
        for m in patterns:
            if m.direction in {"Bullish","Bearish"} and hdir == m.direction:
                m.confidence = round(min(100, m.confidence + 7),1)
                m.note += "; higher timeframe agrees"

    patterns.sort(key=lambda x: x.confidence, reverse=True)
    return {"swings": swings, "supports": supports, "resistances": resistances,
            "structure": structure, "location": location, "patterns": patterns,
            "higher_tf": higher_tf or {}}


def detect_all_patterns(df: pd.DataFrame, pct_threshold=None, active_only=True, higher_tf=None):
    analysis = analyze_chart(df, higher_tf=higher_tf, pct_threshold=pct_threshold)
    results = analysis["patterns"]
    if active_only:
        results = [m for m in results if not m.status.startswith("Completed")]
    return results
