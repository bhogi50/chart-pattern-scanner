# Chart Pattern Scanner — PatternPy Only + Lightweight Charts

Pattern detection is performed only by PatternPy.

The scanner uses PatternPy's:
- `detect_head_shoulder`
- `detect_double_top_bottom`
- `detect_channel`

The result table contains only Stock and Pattern. For each stock, only the most recent PatternPy detection bar is displayed; multiple PatternPy labels on that same bar are comma-separated.

Clicking View opens a candlestick chart rendered with TradingView Lightweight Charts. The chart uses the same OHLC data supplied to PatternPy and marks the PatternPy detection bar.

There is no custom pattern detector, confidence score, target/stop calculation, R:R, or flag detector.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

Market data is fetched with yfinance. The selected candle count is passed to PatternPy after downloading enough history to provide that number of completed candles.
