"""
This module handles all the data coming from the backetester,
paper trading, and live trading. This will make accessing the
necessary trade data and metrics much easier.

Created by Zach Hatzenbeller 2025-07-06
"""

# Standard Library Imports
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime

# Third Party Import
import pandas as pd
import numpy as np


@dataclass
class Trade:
    ticker: str = None
    type: str = None  # "buy" or "sell"
    entry_datatime: datetime = None
    exit_datatime: datetime = None
    entry_price: float = 0
    exit_price: float = 0
    quantity: float = 0
    cost: float = 0
    pnl: float = 0
    stop_loss: float = 0  # price at which you've decided to exit the trade to limit potential losses
    target_price: float = 0 # price at which you've decided to exit the trade to secure profits
    fees: float = 0 # brokerage fees associated with trade
    slippage: float = 0 # difference between expected price and execution price
    entry_signal: dict = None # dictionary of entry strategy signal data for trade
    exit_signal: dict = field(default_factory=dict) # dictionary of exit strategy signal data for trade


class DataHandler:
    def __init__(self):
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.portfolio_values: List[Dict] = []

    def log_trade(self, trade: Trade):
        self.trades.append(trade)

    def record_portfolio_value(
        self, portfolio_value: float, capital: float, index: int
    ):
        self.equity_curve.append(portfolio_value)
        self.portfolio_values.append(
            {
                "portfolio_value": portfolio_value,
                "capital": capital,
                "index": index,
            }
        )

    def get_trade_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def get_equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.portfolio_values)

    def compute_metrics(self) -> Dict[str, float]:
        if not self.equity_curve:
            return {}

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        total_return = (equity[-1] - equity[0]) / equity[0]
        sharpe_ratio = (
            np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
        )  # daily freq
        drawdown = 1 - equity / np.maximum.accumulate(equity)
        max_drawdown = np.max(drawdown)

        trades_df = self.get_trade_df()
        win_rate = (trades_df["pnl"] > 0).mean() if not trades_df.empty else 0
        avg_win = (
            trades_df[trades_df["pnl"] > 0]["pnl"].mean() if not trades_df.empty else 0
        )
        avg_loss = (
            trades_df[trades_df["pnl"] < 0]["pnl"].mean() if not trades_df.empty else 0
        )

        return {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Win Rate": win_rate,
            "Avg Win": avg_win,
            "Avg Loss": avg_loss,
        }
