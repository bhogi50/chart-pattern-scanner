# Chart Pattern Scanner

Pattern-only Streamlit scanner.

## Scan
- NIFTY 50
- NIFTY Next 50
- NIFTY Midcap 150
- NIFTY Smallcap 250
- Daily / Weekly / Monthly
- Configurable candle count
- Bullish / Bearish / Neutral / All
- One row per stock
- Multiple patterns are comma-separated
- No charts on the scan page

## Candle options
Daily: 50, 100, 150, 200, 250, 300
Weekly: 26, 52, 78, 104, 156
Monthly: 12, 24, 36, 60, 120

Defaults: Daily 100, Weekly 52, Monthly 36.

## Stock details
Selecting a stock shows its detected patterns. Selecting a pattern shows:
- Current price
- Entry / breakout
- Stop / invalidation
- Target 1
- Target 2
- Target methodology
- Risk/reward
- Candlestick chart

Yahoo Finance data failures and empty datasets are handled without crashing.
