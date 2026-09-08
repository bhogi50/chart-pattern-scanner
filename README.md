# Chart Pattern Scanner — MVP

A small Streamlit application for rule-based chart-pattern detection.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Supported patterns

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

The detector uses swing highs/lows, linear regression trendlines, normalized slopes,
convergence, touch counts and simple consolidation tests.

This is an MVP. Pattern recognition is heuristic, so confidence is a ranking score,
not a probability of a future price move.

## CSV format

Date,Open,High,Low,Close,Volume
2026-01-01,100,105,98,103,100000

## Planned production upgrades

1. Scan the complete NSE equity universe.
2. Multi-timeframe detection: 15m, 1h, daily, weekly.
3. Pattern overlays and breakout/breakdown levels.
4. Volume confirmation.
5. Pattern lifecycle: forming / confirmed / failed.
6. Backtesting and precision/recall metrics for each pattern.
7. Persistent scan results and a web dashboard.
