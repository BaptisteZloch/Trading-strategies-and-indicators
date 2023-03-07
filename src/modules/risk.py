import numpy as np
import pandas as pd
import scipy.stats as stat

N = 365

def sharpe_ratio_one_asset(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
) -> float:
    """The economist William F. Sharpe proposed the Sharpe ratio in 1966 as an extension of his work on the Capital Asset Pricing Model (CAPM).

    Args:
        returns (pd.Series): The strategy or portfolio returns.
        risk_free_rate (float, optional): The risk free rate usually 10-year bond, buy-and-hold or 0. Defaults to 0.0.

    Returns:
        float: The annualized sharpe ratio.
    """
    return (returns.mean() * N - risk_free_rate) / (
        returns.std() * np.sqrt(N)
    )  # (N**0.5)


def sortino_ratio_one_asset(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
) -> float:
    """The Sortino ratio is very similar to the Sharpe ratio, the only difference being that where the Sharpe ratio uses all the observations for calculating the standard deviation the Sortino ratio only considers the harmful variance.

    Args:
        returns (pd.Series): The strategy or portfolio returns.
        risk_free_rate (float, optional): The risk free rate usually 10-year bond, buy-and-hold or 0. Defaults to 0.0.

    Returns:
        float: The annualized sortino ratio.
    """
    return (returns.mean() * N - risk_free_rate) / (
        semi_deviation(returns) * np.sqrt(N)
    )


def max_drawdown(
    returns: pd.Series,
) -> float:
    """Max drawdown quantifies the steepest decline from peak to trough observed for an investment. This is useful for a number of reasons, mainly the fact that it doesn't rely on the underlying returns being normally distributed. It also gives us an indication of conditionality amongst the returns increments.

    Args:
        returns (pd.Series): The strategy or portfolio returns.

    Returns:
        float: The max drawdown.
    """
    return (
        (
            (returns + 1).cumprod()
            / (returns + 1).cumprod().expanding(min_periods=1).max()
        )
        - 1
    ).min()


def calmar_ratio_one_asset(
    returns: pd.Series,
) -> float:
    """The final risk/reward ratio we will consider is the Calmar ratio. This is similar to the other ratios, with the key difference being that the Calmar ratio uses max drawdown in the denominator as opposed to standard deviation.

    Args:
        returns (pd.Series): The strategy or portfolio returns.

    Returns:
        float: The annualized calmar ratio.
    """
    return (returns.mean() * N) / max_drawdown(returns)


def semi_deviation(returns: pd.Series) -> float:
    """Semi-Deviation is a method of measuring the fluctuations below the mean, unlike variance or standard deviation it only looks at the negative price fluctuations and it's used to evaluate the downside risk (The risk of loss in an investment) of an investment.

    Args:
        returns (pd.Series): The strategy or portfolio returns.

    Returns:
        float: The semi-deviation of returns.
    """
    return returns.loc[returns < 0].std()


def historic_VaR(returns: pd.Series, level: int = 5) -> float:
    """Returns the historical VaR of a Series or DataFrame

    Args:
        returns (pd.Series): The strategy or portfolio returns.
        level (int, optional): Percentile to compute, which must be between 0 and 100 inclusive. Defaults to 5.

    Returns:
        float: The historical VaR.
    """
    return float(-np.percentile(returns, level))


def gaussian_VaR(returns: pd.Series, level: int = 5) -> float:
    """Returns the Parametric Gaussian VaR of a Series or DataFrame

    Args:
        returns (pd.Series): The strategy or portfolio returns.
        level (int, optional): Percentile to compute, which must be between 0 and 100 inclusive. Defaults to 5.

    Returns:
        float: The Parametric Gaussian VaR.
    """
    return float(-(returns.mean() + stat.norm.ppf(level / 100) * returns.std()))


def cornish_fischer_VaR(returns: pd.Series, level: int = 5) -> float:
    """Returns the VaR using the Cornish-Fisher modification of a Series or DataFrame.

    Args:
        returns (pd.Series): The strategy or portfolio returns.
        level (int, optional): Percentile to compute, which must be between 0 and 100 inclusive. Defaults to 5.

    Returns:
        float: The Cornish-Fisher VaR.
    """
    z = stat.norm.ppf(level / 100)
    s = stat.skew(returns.values)
    k = stat.kurtosis(returns.values)

    return float(
        -(
            returns.mean()
            + (
                z
                + (z**2 - 1) * s / 6
                + (z**3 - 3 * z) * (k - 3) / 24
                - (2 * z**3 - 5 * z) * (s**2) / 36
            )
            * returns.std()
        )
    )
