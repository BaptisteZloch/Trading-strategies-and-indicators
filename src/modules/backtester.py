from typing import Optional
import pandas as pd
import numpy as np


def long_only_backtester(
    df: pd.DataFrame,
    long_entry_function: function[[pd.Series, pd.Series], bool],
    long_exit_function: function[[pd.Series, pd.Series, int], bool],
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
    starting_equity: int = 1000,
    maker_fees: float = 0.001,
    taker_fees: float = 0.001,
) -> None | float:
    return None
