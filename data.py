"""
OHLCV data fetching for NSE-listed stocks via yfinance, plus resampling
to Weekly / Monthly timeframes.
"""

import pandas as pd
import streamlit as st
import yfinance as yf

TIMEFRAME_RESAMPLE = {
    "Daily": None,     # no resampling needed
    "Weekly": "W-FRI",
    "Monthly": "ME",
}

# Extra bars fetched on top of what the user asked for, so that after
# resampling to weekly/monthly we still end up with enough candles, and so
# the zigzag/pattern detectors have context before the visible window.
BUFFER_MULTIPLIER = {"Daily": 1.6, "Weekly": 6.0, "Monthly": 24.0}


def _period_for(timeframe: str, num_candles: int) -> str:
    """Choose a yfinance `period` string generous enough for the request."""
    needed_days = int(num_candles * BUFFER_MULTIPLIER.get(timeframe, 1.6)) + 30
    if needed_days <= 60:
        return "3mo"
    if needed_days <= 180:
        return "1y"
    if needed_days <= 400:
        return "2y"
    if needed_days <= 900:
        return "5y"
    if needed_days <= 1800:
        return "10y"
    return "max"


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)  # 4 hour cache
def fetch_ohlc(ticker: str, timeframe: str, num_candles: int) -> pd.DataFrame:
    """
    Fetch daily OHLC for `ticker` (NSE symbol, e.g. 'RELIANCE.NS'),
    resample to the requested timeframe, and return the last `num_candles`
    rows. Returns an empty DataFrame on failure.
    """
    period = _period_for(timeframe, num_candles)
    try:
        raw = yf.download(
            ticker, period=period, interval="1d",
            auto_adjust=True, progress=False, threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance sometimes returns MultiIndex columns for single tickers
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)

    rule = TIMEFRAME_RESAMPLE[timeframe]
    if rule:
        df = df.resample(rule).agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()

    df = df.tail(num_candles)
    return df


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def fetch_many(tickers: tuple, timeframe: str, num_candles: int) -> dict:
    """Fetch OHLC for many tickers. Returns {ticker: DataFrame}."""
    out = {}
    for t in tickers:
        df = fetch_ohlc(t, timeframe, num_candles)
        if not df.empty and len(df) >= 20:
            out[t] = df
    return out
