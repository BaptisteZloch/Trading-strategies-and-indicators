import pandas as pd


def momentum_indicator(close: pd.Series, lag: int = 14) -> pd.Series:
    return 100 * (close.pct_change(periods=lag))
