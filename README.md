# ASTA Tide-Wave Scanner V4

Implements the three checklists shown in GEO PAN - SETUPS.pdf:
- ASTA Triple Screen Setup
- ASTA Swing Trader Setup
- ASTA Momentum Trader Setup

Sidebar:
NIFTY 50 / NIFTY Next 50 / NIFTY Midcap 150 / NIFTY Smallcap 250
Daily / Weekly / Monthly Tide
All / Bullish / Bearish
Setup filter
Minimum confidence
SCAN

Timeframes:
Daily Tide -> 4H Wave
Weekly Tide -> Daily Wave
Monthly Tide -> Weekly Wave

The scanner includes RSI, Stochastic PCO/NCO, EMA 5/13/26, Bollinger Bands,
volume, DI PCO/NCO, ADX, pivot-based chart patterns, Double Top/Bottom with
minimum 8-candle separation, Fib-style structure check, and candlestick checks.

Confidence is a setup-quality score:
Triple Screen 35/65 mandatory/supporting
Swing 45/55
Momentum 70/30

The PDF does not define exact formulas for TI, TLBO/TLBD, ATM PE/CE-TMJ, TMJ,
or some discretionary terms. Those are labelled as proxies or left for a later
data-provider integration rather than silently invented.

Data: Nifty Indices constituent CSV endpoints + Yahoo Finance OHLCV.


## Annotated chart

The stock chart now overlays:
- candlesticks
- swing-high / swing-low pivot markers
- detected pattern support/resistance lines
- current-price / entry reference
- stop reference
- target reference
- setup name, direction and confidence in the chart title
- detected pattern name and geometry score below the chart

The overlay is deliberately labelled as a **reference**, not a broker order recommendation.


### RSI interpretation

For ASTA Momentum Trader:
- Bullish momentum condition: RSI > 40
- Bearish momentum condition: RSI < 60

These are deliberately not treated as RSI >50 / RSI <50. The 40–60 range can
therefore qualify for either directional setup when the remaining conditions
provide the directional confirmation.
