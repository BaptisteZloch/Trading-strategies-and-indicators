import pandas as pd
import numpy as np


def momentum_indicator(close: pd.Series, lag: int = 14) -> pd.Series:
    """Calculate the momentum indicator.

    Args:
        close (pd.Series): The series to compute the momentum on.
        lag (int, optional): The lag for the momentum. Defaults to 14.

    Returns:
        pd.Series: The momentum indicator computed.
    """
    return 100 * (close.pct_change(periods=lag))


def fischer_transformation(close: pd.Series, window: int = 5) -> pd.Series:
    """Compute the fischer transformation on a series.

    Args:
        close (pd.Series): The series to transform.
        window (int): The window to compute the fischer transformation on. Default to 5.

    Returns:
        pd.Series: The transformed series.
    """
    close_normalized: pd.Series = (close - close.rolling(window=window).min()) / (
        close.rolling(window=window).max() - close.rolling(window=window).min()
    )
    close_new: pd.Series = (2 * close_normalized) - 1
    smooth: pd.Series = close_new.ewm(span=5, adjust=True).mean()

    return pd.Series(
        (np.log((1 + smooth) / (1 - smooth))).ewm(span=3, adjust=True).mean(),
        name=f"Fischer{window}",
    )
