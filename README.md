# Chart Pattern Scanner

This version intentionally removes the complete ASTA setup engine.

It is a **pure chart-pattern detector**.

## Universes

- NIFTY 50
- NIFTY Next 50
- NIFTY Midcap 150
- NIFTY Smallcap 250

## Pattern timeframes

The user selects the timeframe on which the pattern itself is detected:

- **Daily** → daily candles
- **Weekly** → weekly candles
- **Monthly** → monthly candles

There is no Tide/Wave/ASTA confirmation in this version.

### Why this timeframe design?

A chart pattern is defined by the candles that form its geometry. Mixing a
Daily pattern with a separate Wave/Tide engine would turn this into a
multi-timeframe strategy rather than a pattern detector.

The scan therefore answers one clean question:

> "Does this stock currently contain this chart pattern on the selected
> timeframe?"

For example:
- Daily scan → detects a Double Bottom from daily candles.
- Weekly scan → detects a Double Bottom from weekly candles.
- Monthly scan → detects a Double Bottom from monthly candles.

The application fetches a longer history for higher timeframes so the detector
has enough candles to establish swing structure.

## Patterns

- Symmetrical Triangle
- Ascending Triangle
- Descending Triangle
- Rising Wedge
- Falling Wedge
- Rising Channel
- Falling Channel
- Double Bottom
- Double Top
- Bullish Flag
- Bearish Flag

## Detection approach

1. Identify swing highs/lows.
2. Build recent structural trendlines.
3. Measure slopes and convergence.
4. Classify triangle/wedge/channel geometry.
5. Detect repeated swing highs/lows for Double Top/Bottom.
6. Detect strong move + consolidation for flags.
7. Draw the detected structure directly on the candlestick chart.

The displayed **Pattern Strength** is a geometry/structure score, not a
probability of profit.

## Important data note

Yahoo Finance is used for OHLCV in this prototype. Before using the scanner
for serious historical research, the data source should be replaced or
validated against a reliable NSE market-data provider.


## Expanded detector
Added Head & Shoulders, Inverse Head & Shoulders, Rectangle, Cup & Handle, Bullish/Bearish Pennants, Resistance Breakout and Support Breakdown. Pattern detection remains single-timeframe and pattern-only.
