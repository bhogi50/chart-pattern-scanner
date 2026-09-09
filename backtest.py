"""
Walk-forward backtest for detector.py.

This answers the question the app itself never answers: when the detector
flags a pattern as "Formed/Active", how often does price actually reach
Target 1 before hitting the Stop? Without this, "confidence" is just a
formula, not a validated probability.

Usage (run locally, where yfinance has network access):

    python backtest.py --universe NIFTY500.csv --timeframe Daily --candles 150 --horizon 40
    python backtest.py --symbols RELIANCE,TCS,INFY --timeframe Daily --candles 150

If run with no arguments, it backtests against a small synthetic dataset so
you can sanity-check the harness itself without network access.

Output: a per-pattern summary (n trades, win rate, avg R-multiple, avg
confidence of winners vs losers) plus a CSV of every individual trade so you
can slice by confidence bucket, direction, timeframe, etc. in a spreadsheet.

IMPORTANT: this measures the SAME thing the app shows the user (does price
reach target1 before stop, within `horizon` bars of the breakout). It does
not know about slippage, brokerage, gaps, or circuit limits — treat the
win-rate as a relative signal for comparing/tuning pattern thresholds, not
as a promise of real trading returns.
"""
import argparse
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from detector import geometry, pattern_confidence, pattern_target, pattern_status, pivots


@dataclass
class Trade:
    symbol: str
    pattern: str
    direction: str
    detected_index: int
    detected_date: str
    confidence: float
    entry: float
    stop: float
    target1: float
    outcome: str          # "win", "loss", "open" (neither hit within horizon)
    bars_to_outcome: int
    r_multiple: float      # realized (or open, marked-to-last-close) R


def _outcome_after_breakout(df, direction, entry, stop, target1, start_idx, horizon):
    """Walk forward from start_idx and see whether target1 or stop is hit
    first, using subsequent High/Low so intrabar touches count."""
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    end = min(len(df), start_idx + 1 + horizon)
    risk = abs(entry - stop) if stop is not None else None
    for i in range(start_idx + 1, end):
        if direction == "Bullish":
            hit_target = h[i] >= target1
            hit_stop = stop is not None and l[i] <= stop
        else:
            hit_target = l[i] <= target1
            hit_stop = stop is not None and h[i] >= stop
        if hit_target and hit_stop:
            # Ambiguous same-bar hit; conservatively count as a loss (stop
            # first) unless you have intraday data to disambiguate.
            r = -1.0 if risk else 0.0
            return "loss", i - start_idx, r
        if hit_target:
            reward = abs(target1 - entry)
            r = reward / risk if risk else None
            return "win", i - start_idx, (r if r is not None else float("nan"))
        if hit_stop:
            return "loss", i - start_idx, -1.0
    # Neither hit within horizon: mark-to-market against last available close.
    last = c[end - 1] if end > 0 else c[start_idx]
    reward = (last - entry) if direction == "Bullish" else (entry - last)
    r = reward / risk if risk else float("nan")
    return "open", end - 1 - start_idx, r


def walk_forward_backtest(df, symbol, min_window=60, step=2, horizon=40, min_confidence=0.0):
    """Slides a `min_window`-bar lookback across `df`, re-running the
    detector at each step, and records a trade the first time a pattern
    transitions into 'Formed/Active' (so an already-active pattern isn't
    counted again every single day it stays active)."""
    trades = []
    active_since = {}  # (pattern, direction) -> True while still active, to avoid duplicate entries
    n = len(df)
    for end in range(min_window, n, step):
        window = df.iloc[:end]
        H, L = pivots(window)
        found, _, H, L = geometry(window)
        seen_this_step = set()
        for pattern, direction in found:
            key = (pattern, direction)
            seen_this_step.add(key)
            status = pattern_status(window, pattern, direction, H, L)
            if status != "Formed/Active":
                active_since[key] = False
                continue
            if active_since.get(key):
                continue  # already counted this activation
            conf = pattern_confidence(window, pattern, direction, H, L)
            if conf < min_confidence:
                active_since[key] = True
                continue
            details = pattern_target(window, pattern, direction, H, L)
            if not details or details.get("target1") is None:
                continue
            outcome, bars, r = _outcome_after_breakout(
                df, direction, details["entry"], details["stop"], details["target1"],
                start_idx=end - 1, horizon=horizon,
            )
            trades.append(Trade(
                symbol=symbol, pattern=pattern, direction=direction,
                detected_index=end - 1,
                detected_date=str(df.index[end - 1]) if hasattr(df.index, "__getitem__") else str(end - 1),
                confidence=conf, entry=details["entry"], stop=details["stop"],
                target1=details["target1"], outcome=outcome, bars_to_outcome=bars,
                r_multiple=r,
            ))
            active_since[key] = True
        for key in list(active_since):
            if key not in seen_this_step:
                active_since[key] = False
    return trades


def summarize(trades):
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(t) for t in trades])
    df["conf_bucket"] = pd.cut(df["confidence"], [0, 50, 65, 80, 100],
                                labels=["<50", "50-65", "65-80", "80+"])
    closed = df[df["outcome"] != "open"]
    rows = []
    for (pattern, bucket), g in closed.groupby(["pattern", "conf_bucket"], observed=True):
        wins = (g["outcome"] == "win").sum()
        rows.append({
            "pattern": pattern, "confidence_bucket": bucket, "n_trades": len(g),
            "win_rate": round(wins / len(g), 3) if len(g) else None,
            "avg_r_multiple": round(g["r_multiple"].mean(), 2),
        })
    return pd.DataFrame(rows).sort_values(["pattern", "confidence_bucket"])


def _synthetic_demo():
    np.random.seed(11)
    n = 400
    x = np.arange(n)
    close = 100 + 0.02 * x + np.cumsum(np.random.randn(n)) * 0.8
    high = close + np.random.rand(n) * 1.5 + 1
    low = close - np.random.rand(n) * 1.5 - 1
    vol = np.random.randint(100000, 500000, n).astype(float)
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol})
    trades = walk_forward_backtest(df, "SYNTH", min_window=60, step=2, horizon=30)
    print(f"Synthetic demo: {len(trades)} trades generated.")
    print(summarize(trades).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", help="Comma-separated NSE symbols, e.g. RELIANCE,TCS,INFY")
    ap.add_argument("--universe", help="CSV file with a SYMBOL column")
    ap.add_argument("--timeframe", default="Daily", choices=["Daily", "Weekly", "Monthly"])
    ap.add_argument("--candles", type=int, default=150, help="rolling lookback window size used by the detector")
    ap.add_argument("--horizon", type=int, default=40, help="max bars to wait for target/stop after breakout")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--out", default="backtest_trades.csv")
    args = ap.parse_args()

    if not args.symbols and not args.universe:
        _synthetic_demo()
        return

    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is required for real backtests: pip install yfinance", file=sys.stderr)
        sys.exit(1)

    if args.universe:
        syms = pd.read_csv(args.universe)
        col = next(c for c in syms.columns if c.upper() == "SYMBOL")
        symbols = syms[col].dropna().astype(str).str.strip().tolist()
    else:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    interval = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}[args.timeframe]
    all_trades = []
    for sym in symbols:
        try:
            hist = yf.download(sym + ".NS", period="5y", interval=interval, auto_adjust=False, progress=False)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            needed = {"Open", "High", "Low", "Close"}
            if hist.empty or not needed.issubset(hist.columns):
                continue
            hist = hist.dropna(subset=list(needed))
            trades = walk_forward_backtest(
                hist, sym, min_window=args.candles, step=2, horizon=args.horizon,
                min_confidence=args.min_confidence,
            )
            all_trades.extend(trades)
            print(f"{sym}: {len(trades)} trades")
        except Exception as e:
            print(f"{sym}: skipped ({e})", file=sys.stderr)

    if not all_trades:
        print("No trades generated.")
        return

    trades_df = pd.DataFrame([asdict(t) for t in all_trades])
    trades_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(trades_df)} trades to {args.out}\n")
    print(summarize(all_trades).to_string(index=False))


if __name__ == "__main__":
    main()
