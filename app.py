import io
import json
import html
import requests
import pandas as pd
import streamlit as st
from tradingpatterns.tradingpatterns import (
    detect_head_shoulder,
    detect_double_top_bottom,
    detect_channel,
)

st.set_page_config(page_title="Chart Pattern Scanner", page_icon="📈", layout="wide")

UNIVERSES = {
    "NIFTY 50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "NIFTY Next 50": "https://www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
    "NIFTY Midcap 150": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    "NIFTY Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

CANDLE_OPTIONS = {
    "Daily": [50, 100, 150, 200, 250, 300],
    "Weekly": [26, 52, 78, 104, 156],
    "Monthly": [12, 24, 36, 60, 120],
}

BIAS = {
    "Double Bottom": "Bullish",
    "Double Top": "Bearish",
    "Head and Shoulder": "Bearish",
    "Inverse Head and Shoulder": "Bullish",
    "Channel Up": "Bullish",
    "Channel Down": "Bearish",
}

PATTERN_COLUMNS = (
    "head_shoulder_pattern",
    "double_pattern",
    "channel_pattern",
)


def period_for(tf, n):
    # Request enough source history for PatternPy, then pass exactly n candles to it.
    if tf == "Daily":
        return f"{max(2, int(n * 1.7))}d"
    if tf == "Weekly":
        return f"{max(2, int(n * 1.7 / 52) + 1)}y"
    return f"{max(2, int(n * 1.7 / 12) + 1)}y"


@st.cache_data(ttl=86400)
def symbols(name):
    r = requests.get(
        UNIVERSES[name], headers={"User-Agent": "Mozilla/5.0"}, timeout=20
    )
    r.raise_for_status()
    x = pd.read_csv(io.BytesIO(r.content))
    c = next(c for c in x.columns if c.upper() == "SYMBOL")
    return x[c].dropna().astype(str).str.strip().tolist()


@st.cache_data(ttl=900)
def prices(symbol, tf, n):
    import yfinance as yf

    x = yf.download(
        symbol,
        period=period_for(tf, n),
        interval={"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}[tf],
        auto_adjust=False,
        progress=False,
    )
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    need = {"Open", "High", "Low", "Close"}
    if x.empty or not need.issubset(x.columns):
        return pd.DataFrame()
    return x.dropna(subset=list(need)).tail(n)


def patternpy(df):
    """Run only the PatternPy detectors; no custom pattern recognition."""
    x = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].copy()
    x = detect_head_shoulder(x, window=3)
    x = detect_double_top_bottom(x, window=3, threshold=0.05)
    x = detect_channel(x, window=3)
    return x


def latest_patternpy_results(x, direction="All"):
    """Return only PatternPy labels from the most recent detection bar."""
    detections = []
    for col in PATTERN_COLUMNS:
        if col not in x.columns:
            continue
        series = x[col].dropna()
        for idx, value in series.items():
            value = str(value).strip()
            if value in BIAS and (direction == "All" or BIAS[value] == direction):
                detections.append((idx, value))
    if not detections:
        return []
    latest_idx = max(idx for idx, _ in detections)
    return list(dict.fromkeys(value for idx, value in detections if idx == latest_idx))


def pattern_detection_dates(x, patterns):
    """Return the PatternPy detection timestamp for each displayed label."""
    result = {}
    for col in PATTERN_COLUMNS:
        if col not in x.columns:
            continue
        for idx, value in x[col].dropna().items():
            value = str(value).strip()
            if value in patterns:
                result[value] = pd.Timestamp(idx)
    return result


def lightweight_chart(df, patterns, detection_dates):
    """Render TradingView Lightweight Charts using the same OHLC data fed to PatternPy."""
    candles = []
    for idx, row in df.iterrows():
        ts = pd.Timestamp(idx)
        candles.append({
            "time": int(ts.timestamp()),
            "open": float(row["Open"]), "high": float(row["High"]),
            "low": float(row["Low"]), "close": float(row["Close"]),
        })
    markers = []
    by_time = {int(pd.Timestamp(idx).timestamp()): (idx, row) for idx, row in df.iterrows()}
    for pattern, dt in detection_dates.items():
        t = int(pd.Timestamp(dt).timestamp())
        if t in by_time:
            row = by_time[t][1]
            bullish = BIAS.get(pattern) == "Bullish"
            markers.append({
                "time": t,
                "position": "belowBar" if bullish else "aboveBar",
                "shape": "arrowUp" if bullish else "arrowDown",
                "text": pattern.replace(" and ", " & "),
            })
    payload = json.dumps({"candles": candles, "markers": markers}, separators=(",", ":"))
    html_doc = f"""
<!doctype html><html><head><meta charset="utf-8">
<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
<style>html,body{{margin:0;background:#0f131a}}#chart{{height:680px;width:100%}}</style></head>
<body><div id="chart"></div><script>
const D={payload};
const chart=LightweightCharts.createChart(document.getElementById('chart'),{{
  autoSize:true, layout:{{background:{{type:'solid',color:'#0f131a'}},textColor:'#c9d1d9'}},
  grid:{{vertLines:{{color:'#202631'}},horzLines:{{color:'#202631'}}}},
  rightPriceScale:{{borderColor:'#303744'}}, timeScale:{{borderColor:'#303744',timeVisible:true,secondsVisible:false}},
  crosshair:{{mode:LightweightCharts.CrosshairMode.Normal}}
}});
const series=chart.addSeries(LightweightCharts.CandlestickSeries,{{
  upColor:'#26a69a',downColor:'#ef5350',borderUpColor:'#26a69a',borderDownColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350'
}});
series.setData(D.candles);
if(D.markers.length && LightweightCharts.createSeriesMarkers) LightweightCharts.createSeriesMarkers(series,D.markers);
chart.timeScale().fitContent();
</script></body></html>
"""
    st.html(html_doc, height=700)


st.title("📈 Chart Pattern Scanner")
st.caption("Pattern detection powered directly by PatternPy")

with st.sidebar:
    universe = st.selectbox("Universe", list(UNIVERSES))
    tf = st.selectbox("Pattern timeframe", list(CANDLE_OPTIONS))
    n = st.selectbox(
        "Number of candles",
        CANDLE_OPTIONS[tf],
        index=CANDLE_OPTIONS[tf].index({"Daily": 100, "Weekly": 52, "Monthly": 36}[tf]),
    )
    direction = st.radio("Direction filter", ["All", "Bullish", "Bearish"])
    scan = st.button("🔎 SCAN PATTERNS", type="primary", width="stretch")
    st.caption(f"PatternPy receives the latest {n} {tf.lower()} candles.")

if scan:
    try:
        syms = symbols(universe)
    except Exception as e:
        st.error(f"Unable to load {universe}: {e}")
        st.stop()

    rows = []
    bar = st.progress(0)
    for i, s in enumerate(syms, 1):
        try:
            d = prices(s + ".NS", tf, n)
            if d.empty:
                continue
            x = patternpy(d)
            ps = latest_patternpy_results(x, direction)
            if ps:
                rows.append({"Stock": s, "Pattern": ", ".join(ps)})
        except Exception:
            pass
        bar.progress(i / len(syms))

    st.session_state.results = rows
    st.session_state.selected = None

if "results" in st.session_state:
    rows = st.session_state.results
    st.subheader(f"Patterns detected ({len(rows)} stocks)")
    if not rows:
        st.info("No PatternPy patterns were detected for this scan.")
    else:
        h = st.columns([2, 7, 1])
        h[0].markdown("**Stock**")
        h[1].markdown("**Pattern**")
        h[2].markdown("**View**")
        for r in rows:
            c = st.columns([2, 7, 1])
            c[0].write(r["Stock"])
            c[1].write(r["Pattern"])
            if c[2].button("View", key="view_" + r["Stock"]):
                st.session_state.selected = r["Stock"]

if st.session_state.get("selected"):
    s = st.session_state.selected
    st.divider()
    st.subheader(f"{s} — Chart")
    displayed = next((r["Pattern"] for r in st.session_state.results if r["Stock"] == s), "None")
    st.caption(f"PatternPy detected: {displayed}")
    d = prices(s + ".NS", tf, n)
    x = patternpy(d)
    pats = latest_patternpy_results(x, direction="All")
    dates = pattern_detection_dates(x, pats)
    lightweight_chart(d, pats, dates)
