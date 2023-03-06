from datetime import datetime
from typing import Callable
import pandas as pd
import numpy as np


def long_only_backtester(
    df: pd.DataFrame,
    long_entry_function: Callable[[pd.Series, pd.Series], bool],
    long_exit_function: Callable[[pd.Series, pd.Series, int], bool],
    take_profit: float = np.inf,
    stop_loss: float = np.inf,
    initial_equity: int = 1000,
    maker_fees: float = 0.001,
    taker_fees: float = 0.001,
) -> None | float | pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        long_entry_function (Callable[[pd.Series, pd.Series], bool]): The long entry function, it should take 2 arguments, the current row and the previous row and return True or False depending on your strategy.
        long_exit_function (Callable[[pd.Series, pd.Series, int], bool]): The long exit function, it should take 2 arguments, the current row and the previous row and return True or False depending on your strategy.
        take_profit (float, optional): _description_. Defaults to np.inf.
        stop_loss (float, optional): _description_. Defaults to np.inf.
        initial_equity (int, optional): _description_. Defaults to 1000.
        maker_fees (float, optional): _description_. Defaults to 0.001.
        taker_fees (float, optional): _description_. Defaults to 0.001.

    Returns:
        None | float | pd.DataFrame: _description_
    """
    ohlcv_df = df.copy()
    previous_row = ohlcv_df.iloc[0]
    position_opened = False
    trading_days = 0
    equity = initial_equity
    net_equity = initial_equity
    current_trade: dict[str, float | datetime | int | str | pd.Timestamp] = {}

    trades_df = pd.DataFrame(
        columns=[
            "buy_date",
            "buy_price",
            "buy_reason",
            "sell_date",
            "sell_price",
            "sell_reason",
            "return",
            "equity",
            "net_equity",
            "fees",
        ]
    )

    for index, row in ohlcv_df[1:].iterrows():
        if position_opened is False and long_entry_function(row, previous_row) is True:
            position_opened = True
            current_trade["buy_date"] = pd.Timestamp(index)
            current_trade["buy_price"] = row.Close
            current_trade["buy_reason"] = "Entry position triggered"
            trading_days = 1
            buy_price = row.Close
        elif (
            position_opened is True
            and long_exit_function(row, previous_row, trading_days) is True
        ):
            position_opened = False
            current_trade["sell_date"] = pd.Timestamp(index)
            current_trade["sell_price"] = row.Close
            current_trade["sell_reason"] = "Exit position triggered"

            ret = (current_trade["sell_price"] / current_trade["buy_price"]) - 1
            current_trade["return"] = ret
            current_trade["equity"] = equity * (1 + ret)
            current_trade["fees"] = abs(net_equity * -2 * taker_fees)
            current_trade["net_equity"] = net_equity * (1 + ret) - current_trade["fees"]

            trades_df = pd.concat([trades_df, pd.DataFrame([current_trade])])
            trading_days = 0
            equity = current_trade["equity"]
            net_equity = current_trade["net_equity"]
            current_trade = {}
        # elif position_opened is True and buy_price * (1 + take_profit) <= row.High:
        #     position_opened = False
        #     current_trade["sell_date"] = index
        #     current_trade["sell_price"] = buy_price * (1 + take_profit)
        #     current_trade["sell_reason"] = "Take profit triggered"

        #     ret = (current_trade["sell_price"] / current_trade["buy_price"]) - 1
        #     current_trade["return"] = ret
        #     current_trade["equity"] = equity * (1 + ret)
        #     current_trade["fees"] = abs(net_equity * -2 * taker_fees)
        #     current_trade["net_equity"] = (
        #         current_trade["equity"] - current_trade["fees"]
        #     )

        #     trades_df = pd.concat([trades_df, pd.DataFrame([current_trade])])
        #     trading_days = 0
        #     equity = current_trade["equity"]
        #     net_equity = current_trade["net_equity"]
        #     current_trade = {}
        # elif position_opened is True and buy_price * (1 - stop_loss) >= row.Low:
        #     position_opened = False
        #     current_trade["sell_date"] = index
        #     current_trade["sell_price"] = buy_price * (1 - stop_loss)
        #     current_trade["sell_reason"] = "Stop loss triggered"

        #     ret = (current_trade["sell_price"] / current_trade["buy_price"]) - 1
        #     current_trade["return"] = ret
        #     current_trade["equity"] = equity * (1 + ret)
        #     current_trade["fees"] = abs(net_equity * -2 * taker_fees)
        #     current_trade["net_equity"] = (
        #         current_trade["equity"] - current_trade["fees"]
        #     )

        #     trades_df = pd.concat([trades_df, pd.DataFrame([current_trade])])
        #     trading_days = 0
        #     equity = current_trade["equity"]
        #     net_equity = current_trade["net_equity"]
        #     current_trade = {}
        else:
            trading_days += 1

        previous_row = row

    assert len(trades_df) > 0, "No trades were generated"

    trades_df["trade_duration"] = trades_df["sell_date"] - trades_df["buy_date"]

    good_trades = trades_df.loc[trades_df["return"] > 0]
    bad_trades = trades_df.loc[trades_df["return"] < 0]
    total_trades = len(trades_df)

    winrate = 100 * len(good_trades) / total_trades

    strategy_final_return = 100 * trades_df["equity"].iloc[-1] / initial_equity
    strategy_final_return_net = 100 * trades_df["net_equity"].iloc[-1] / initial_equity
    buy_and_hold_return = 100 * ohlcv_df["Close"].iloc[-1] / ohlcv_df["Close"].iloc[0]

    print(f"{'  General informations  ':-^50}")
    print(f"Period: [{str(ohlcv_df.index[0])}] -> [{str(ohlcv_df.index[-1])}]")
    print(f"Intial balance: {initial_equity} $")

    print(f"\n{'  Strategy performance  ':-^50}")
    print(f"Final balance: {trades_df['equity'].iloc[-1]:.2f} $")
    print(f"Strategy return: {strategy_final_return:.2f} %")
    print(f"Strategy fees: {trades_df['fees'].sum():.2f} $")
    print(f"Final net balance: {trades_df['net_equity'].iloc[-1]:.2f} $")
    print(f"Strategy net return: {strategy_final_return_net:.2f} %")
    print(f"Buy and Hold return: {buy_and_hold_return:.2f} %")
    print(f"Strategy winrate: {winrate:.2f} %")
    print(f"\n{'  Trades informations  ':-^50}")
    print(
        f"Mean trade duration: {str((trades_df['trade_duration']).mean()).split('.')[0]}"
    )
    print(f"Total trades: {total_trades}")
    print(f"Total bad trades: {len(bad_trades)}")
    print(f"Total good trades: {len(good_trades)}")
    print(f"Mean good trades return: {100*good_trades['return'].mean():.2f} %")
    print(f"Mean bad trades return: {100*bad_trades['return'].mean():.2f} %")
    print(f"Median good trades return: {100*good_trades['return'].median():.2f} %")
    print(f"Median bad trades return: {100*bad_trades['return'].median():.2f} %")
    print(
        f"Mean good trade duration: {str((good_trades['trade_duration']).mean()).split('.')[0]}"
    )
    print(
        f"Mean good trade duration: {str((bad_trades['trade_duration']).mean()).split('.')[0]}"
    )
    print(f"Exit reasons repartition: {trades_df.reasons.value_counts()}")

    return trades_df
