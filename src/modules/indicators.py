import pandas as pd
import numpy as np
from ta.momentum import ppo_hist


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

    return (
        pd.Series((np.log((1 + smooth) / (1 - smooth)))).ewm(span=3, adjust=True).mean()
    )


def z_score_indicator(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute the z-score of a time series over a window.

    Args:
        close (pd.Series): The time series to compute the z-score on.
        window (int, optional): The window length of the data used to compute the z-score. Defaults to 14.

    Returns:
        pd.Series: The z-score of the time series.
    """
    return (close - close.rolling(window=window).mean()) / close.rolling(
        window=window
    ).std()


def mean_break_out_indicator(close: pd.Series, window: int = 20) -> pd.Series:
    """Compute the MBO, mean breakout indicator on a time series. It compares the difference between the closing price of a candle and a moving average over N periods to the difference between the min and max value of the closing price over the same N periods.

    Args:
        close (pd.Series): The time series to process.
        window (int, optional): The window length of the data used to compute the MBO. Defaults to 20.

    Returns:
        pd.Series: The MBO indicator.
    """
    return (close - close.ewm(window).mean()) / (
        close.rolling(window).max() - close.rolling(window).min()
    )


def trend_intensity_indicator(close: pd.Series, window: int = 20) -> pd.Series:
    """The Trend Intensity indicator is a measure of the accumulation of the number of periods the price is above the EMA against the price is under the EMA. Given a number of periods equal to the number of periods of the EMA it gives an information on the trend strength and if the trend will tend to reverse.

    Args:
        close (pd.Series): The time series to process.
        window (int, optional): The window length of the data used to compute the trend intensity. Defaults to 20.

    Returns:
        pd.Series: The trend intensity indicator.
    """
    return (
        100
        * pd.DataFrame({"EMA": close.ewm(window).mean().values, "Close": close.values})
        .apply(lambda row: 1 if row["Close"] > row["EMA"] else 0, axis=1)
        .rolling(window)
        .sum()
        / window
    )


def ppo_slope_indicator(close: pd.Series, slope_window: int = 4) -> pd.Series:
    ppo_histo = ppo_hist(close=close)

    def calculate_slope(ppo_values: pd.Series) -> float:
        x = np.arange(1, len(ppo_values) + 1, 1)
        y = np.array(ppo_values)
        m, c = np.polyfit(x, y, 1)
        return m

    return ppo_histo.rolling(slope_window).apply(calculate_slope)


#  @classmethod
#     def PIVOT(cls, ohlc: DataFrame) -> DataFrame:
#         """
#         Pivot Points are significant support and resistance levels that can be used to determine potential trades.
#         The pivot points come as a technical analysis indicator calculated using a financial instrument’s high, low, and close value.
#         The pivot point’s parameters are usually taken from the previous day’s trading range.
#         This means you’ll have to use the previous day’s range for today’s pivot points.
#         Or, last week’s range if you want to calculate weekly pivot points or, last month’s range for monthly pivot points and so on.
#         """

#         df = ohlc.shift()  # pivot is calculated of the previous trading session

#         pivot = pd.Series(pd.Series((ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3), name="pivot")  # pivot is basically a lagging TP

#         s1 = (pivot * 2) - df["high"]
#         s2 = pivot - (df["high"] - df["low"])
#         s3 = df["low"] - (2 * (df["high"] - pivot))
#         s4 = df["low"] - (3 * (df["high"] - pivot))

#         r1 = (pivot * 2) - df["low"]
#         r2 = pivot + (df["high"] - df["low"])
#         r3 = df["high"] + (2 * (pivot - df["low"]))
#         r4 = df["high"] + (3 * (pivot - df["low"]))

#         return pd.concat(
#             [
#                 pivot,
#                 pd.Series(s1, name="s1"),
#                 pd.Series(s2, name="s2"),
#                 pd.Series(s3, name="s3"),
#                 pd.Series(s4, name="s4"),
#                 pd.Series(r1, name="r1"),
#                 pd.Series(r2, name="r2"),
#                 pd.Series(r3, name="r3"),
#                 pd.Series(r4, name="r4"),
#             ],
#             axis=1,
#         )

# @classmethod
# def MI(cls, ohlc: DataFrame, period: int = 9, adjust: bool = True) -> Series:
#     """Developed by Donald Dorsey, the Mass Index uses the high-low range to identify trend reversals based on range expansions.
#     In this sense, the Mass Index is a volatility indicator that does not have a directional bias.
#     Instead, the Mass Index identifies range bulges that can foreshadow a reversal of the current trend."""

#     _range = pd.Series(ohlc["high"] - ohlc["low"], name="range")
#     EMA9 = _range.ewm(span=period, ignore_na=False, adjust=adjust).mean()
#     DEMA9 = EMA9.ewm(span=period, ignore_na=False, adjust=adjust).mean()
#     mass = EMA9 / DEMA9

#     return pd.Series(mass.rolling(window=25).sum(), name="Mass Index")
