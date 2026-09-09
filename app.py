import io
import json
import html
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
    """Return only the most recent PatternPy detection(s) on the latest detection bar."""
    detections = []
    for col in PATTERN_COLUMNS:
        if col not in x.columns:
            continue
        s = x[col].dropna()
        for idx, value in s.items():
            value = str(value)
            if value in BIAS and (direction == "All" or BIAS[value] == direction):
                detections.append((idx, value))

    if not detections:
        return []

    latest_idx = max(idx for idx, _ in detections)
    latest = []
    for idx, value in detections:
        if idx == latest_idx and value not in latest:
            latest.append(value)
    return latest


def tradingview_chart(symbol, tf):
    """Render TradingView's free hosted Advanced Chart widget."""
    interval = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[tf]
    tv_symbol = f"NSE:{symbol}"
    config = {
        "autosize": True,
        "symbol": tv_symbol,
        "interval": interval,
        "timezone": "exchange",
        "theme": "dark",
        "style": "1",
        "withdateranges": True,
        "hide_side_toolbar": False,
        "allow_symbol_change": True,
        "save_image": False,
        "locale": "en",
        "calendar": False,
        "support_host": "https://www.tradingview.com",
        "details": False,
    }
    config_json = json.dumps(config, separators=(",", ":"))
    copyright_symbol = html.escape(tv_symbol)
    widget = f"""
    <div class="tradingview-widget-container" style="height:100%;width:100%">
      <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
      <div class="tradingview-widget-copyright" style="font-size:11px;text-align:center">
        <a href="https://www.tradingview.com/symbols/{copyright_symbol}/" rel="noopener nofollow" target="_blank">
          <span>Chart by TradingView</span>
        </a>
      </div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {config_json}
      </script>
    </div>
    """
    components.html(widget, height=720, scrolling=False)


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
    scan = st.button("🔎 SCAN PATTERNS", type="primary", use_container_width=True)
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
    st.subheader(f"{s} — TradingView chart")
    st.caption(f"PatternPy detected: {next((r['Pattern'] for r in st.session_state.results if r['Stock'] == s), 'None')}")
    tradingview_chart(s, tf)
