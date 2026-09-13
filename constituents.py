"""
Index constituent lists for Nifty 50 / Next 50 / Midcap 150 / Smallcap 250.

Primary source: NSE's public index CSV archives (fetched live, so the list
stays current without needing code changes). If that fetch fails (NSE
occasionally rate-limits or blocks non-browser requests, and this can also
happen if the host has no internet access), we fall back to a bundled
static CSV snapshot in constituents/*.csv so the app still works.
"""

import io
import os
import pandas as pd
import requests
import streamlit as st

NSE_URLS = {
    "Nifty 50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "Nifty Next 50": "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "Nifty Midcap 150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "Nifty Smallcap 250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
}

FALLBACK_FILES = {
    "Nifty 50": "constituents/nifty50.csv",
    "Nifty Next 50": "constituents/niftynext50.csv",
    "Nifty Midcap 150": "constituents/niftymidcap150.csv",
    "Nifty Smallcap 250": "constituents/niftysmallcap250.csv",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
}


def _base_dir():
    return os.path.dirname(os.path.abspath(__file__))


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)  # refresh once a day
def get_index_constituents(index_name: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: Symbol, CompanyName, Ticker (yfinance-ready).
    """
    df = None

    url = NSE_URLS.get(index_name)
    if url:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            if resp.status_code == 200 and len(resp.content) > 100:
                df = pd.read_csv(io.BytesIO(resp.content))
        except Exception:
            df = None

    if df is None or df.empty:
        fallback_path = os.path.join(_base_dir(), FALLBACK_FILES[index_name])
        if os.path.exists(fallback_path):
            df = pd.read_csv(fallback_path)
        else:
            return pd.DataFrame(columns=["Symbol", "CompanyName", "Ticker"])

    # NSE CSVs use columns: "Company Name", "Industry", "Symbol", "Series", "ISIN Code"
    symbol_col = next((c for c in df.columns if c.strip().lower() == "symbol"), df.columns[-1])
    name_col = next((c for c in df.columns if "company" in c.strip().lower()), df.columns[0])

    out = pd.DataFrame({
        "Symbol": df[symbol_col].astype(str).str.strip(),
        "CompanyName": df[name_col].astype(str).str.strip(),
    })
    out["Ticker"] = out["Symbol"] + ".NS"
    out = out.drop_duplicates(subset="Symbol").reset_index(drop=True)
    return out
