# Nifty Pattern Scanner — Human-like Structure v3

This version changes the detector from **pattern-first** to **structure-first**.

## Pipeline
1. Cluster **all candle highs** into resistance zones and rank the nearest relevant zones R1/R2/R3.
2. Cluster **all candle lows** into support zones S1/S2/S3.
3. Analyze the next higher timeframe for context.
4. Extract meaningful swing highs/lows.
5. Classify the current geometry as Channel Up, Channel Down, Ascending Triangle, Descending Triangle, Flat/Range, or Unclear.
6. Determine today's location relative to R/S and the structure boundaries, plus current candle behavior.
7. Only then activate relevant Double Top / Double Bottom checks.
8. Return a human-readable interpretation and confirmation/development status.

H&S and inverse H&S are intentionally removed.

## Triangle rule
A triangle is deliberately simplified to:
- Ascending Triangle = flat upper boundary + rising lower boundary.
- Descending Triangle = falling upper boundary + flat lower boundary.

Symmetrical convergence is not forced into a triangle; it remains Unclear.
