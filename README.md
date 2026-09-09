# Chart Pattern Scanner — PatternPy Edition

This version uses **PatternPy as the pattern-candidate engine** and a separate validation/scoring layer for this scanner.

Pattern families shown by the application:
- Double Top
- Double Bottom
- Head and Shoulders
- Inverse Head and Shoulders
- Bullish Flag
- Bearish Flag
- Rising Channel
- Falling Channel

PatternPy is used for candidate recognition of tops/bottoms, H&S and channels. Flags remain scanner-specific because PatternPy does not provide a flag detector.

The scanner then validates the candidate using price structure, ATR tolerance, spacing, neckline/reaction, channel fit/containment and recent-break checks before displaying it.

## Run
`pip install -r requirements.txt`
`streamlit run app.py`

## Attribution / license
PatternPy: https://github.com/keithorange/PatternPy
PatternPy is licensed CC BY-NC-SA 4.0. This project is intended for personal, non-commercial use.
