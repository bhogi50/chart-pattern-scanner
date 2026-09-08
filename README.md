# NSE Chart Pattern Scanner V2

Personal Streamlit application for scanning NIFTY 50 daily OHLCV data and detecting
heuristic chart patterns.

## Files

- `app.py` — Streamlit UI and NIFTY 50 scanner
- `detector.py` — pattern detection engine
- `requirements.txt` — Python dependencies

## Detected patterns

- Symmetrical Triangle
- Ascending Triangle
- Descending Triangle
- Rising Wedge
- Falling Wedge
- Rising Channel
- Falling Channel
- Double Top
- Double Bottom
- Bullish Flag
- Bearish Flag

## Deploy

The repository is intended for Streamlit Community Cloud.

After replacing these files in GitHub, Streamlit Community Cloud should rebuild
the application from the repository.

## Important

The confidence value is a heuristic pattern-quality score. It is not a statistical
probability that a pattern will produce a particular future price movement.

## Next upgrades

1. All NSE equities rather than only NIFTY 50.
2. Pattern overlays on candlestick charts.
3. Breakout/breakdown confirmation.
4. Volume confirmation.
5. Forming / confirmed / failed pattern lifecycle.
6. Head & Shoulders and Inverse Head & Shoulders.
7. Cup & Handle and Rounding Top/Bottom.
8. Multi-timeframe scanning.
9. Historical validation and backtesting.
