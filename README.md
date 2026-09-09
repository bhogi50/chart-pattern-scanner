# PatternPy-only Chart Pattern Scanner

OHLCV data is passed directly to PatternPy's native `detect_head_shoulder`, `detect_double_top_bottom`, and `detect_channel` functions. The application only reads PatternPy's output, filters the displayed rows, and renders the chart.

There is no custom pattern detector, confidence score, target/stop/R:R calculation, flag detector, or custom validation layer.

PatternPy: https://github.com/keithorange/PatternPy
License: CC BY-NC-SA 4.0 (personal/non-commercial use).
