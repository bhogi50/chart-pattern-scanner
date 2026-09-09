# Chart Pattern Scanner

Free TradingView-style charting is implemented with TradingView Lightweight Charts.
The library is loaded client-side; no paid TradingView charting product is used.

## Chart rendering
The stock detail chart renders:
- Candlesticks
- Crosshair
- Zoom and pan
- Swing high/low markers
- Pattern-specific trendlines/boundaries
- Entry / breakout level
- Stop / invalidation level
- Target 1
- Target 2
- Pattern name, direction and confidence

Pattern geometry is drawn from the actual swing points used by the detector rather than as a generic decorative overlay.

Lightweight Charts requires TradingView attribution on a public page.
