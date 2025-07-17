"""
This module handles all the data coming from the backetester,
paper trading, and live trading. This will make accessing the
necessary trade data and metrics much easier.

Created by Zach Hatzenbeller 2025-07-06
"""

# Repository Imports
from utils.metrics import (
    sharpe_ratio,
    sortino_ratio,
    beta,
    alpha,
    max_drawdown,
    win_loss_rates,
)

# Standard Library Imports
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

# Third Party Import
import pandas as pd
import numpy as np


@dataclass
class Trade:
    ticker: str = None
    type: str = None  # "buy" or "sell"
    entry_datatime: datetime = None
    exit_datatime: datetime = None
    entry_close: float = 0
    entry_price: float = 0
    exit_close: float = 0
    exit_price: float = 0
    quantity: float = 0
    cost: float = 0
    pnl: float = 0
    stop_loss: float = (
        0  # price at which you've decided to exit the trade to limit potential losses
    )
    target_price: float = (
        0  # price at which you've decided to exit the trade to secure profits
    )
    fees: float = 0  # brokerage fees associated with trade
    slippage: float = 0  # difference between expected price and execution price
    entry_signal: dict = None  # dictionary of entry strategy signal data for trade
    exit_signal: dict = field(
        default_factory=dict
    )  # dictionary of exit strategy signal data for trade


class DataHandler:
    def __init__(self, market_reference_data: pd.DataFrame):
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.portfolio_values: List[Dict] = []
        self.drawdown: List[float] = []

        # market reference data for computing metrics
        self.market_reference = market_reference_data

    def log_trade(self, trade: Trade):
        self.trades.append(trade)

    def record_equity_value(self, portfolio_value: float):
        self.equity_curve.append(portfolio_value)

    def record_portfolio_value(
        self, symbol: str, portfolio_value: float, capital: float, index: int
    ):
        self.portfolio_values.append(
            {
                "symbol": symbol,
                "portfolio_value": portfolio_value,
                "capital": capital,
                "index": index,
            }
        )

    def get_trade_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def get_equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.portfolio_values)

    def get_asset_equities(self):
        assets_equities = defaultdict(list)
        for value in self.portfolio_values:
            assets_equities[value["symbol"]].append(
                (value["index"], value["portfolio_value"])
            )

        return dict(assets_equities)

    def compute_metrics(self, risk_free_rate: float = 4.5) -> Dict[str, float]:
        if not self.equity_curve:
            return {}

        # get all data from trades made
        trades_df = self.get_trade_df()

        # calculate portfolio returns
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        downside_returns = returns[returns < 0]

        # calculate asset returns
        asset_returns = {}
        asset_equities = self.get_asset_equities()
        for symbol, value in asset_equities.items():
            _, asset_equity = zip(*value)
            asset_equity = np.array(asset_equity)
            asset_returns[symbol] = np.diff(asset_equity) / asset_equity[:-1]

        # calculate market reference returns
        market_reference_returns = self.market_reference["close"].pct_change().values

        # calculate the total return over the trading period
        total_return = (equity[-1] - equity[0]) / equity[0]

        # Calculate sharpe ratio
        sharpe = sharpe_ratio(portfolio_returns=returns, annual_rf_rate=risk_free_rate)

        # calculate the sortino ratio
        sortino = sortino_ratio(
            downside_portfolio_returns=downside_returns, annual_rf_rate=risk_free_rate
        )

        # calculate asset and portfolio betas
        portfolio_beta, asset_betas = beta(
            returns_dict=asset_returns,
            market_reference_returns=market_reference_returns,
        )

        # calculate portfolio alpha
        alpha_ = alpha(
            portfolio_returns=returns,
            market_reference_returns=market_reference_returns,
            annual_rf_rate=risk_free_rate,
            portfolio_beta=portfolio_beta,
        )

        # calculate the max drawdown for the portfolio its drawdown curve
        max_dd, self.drawdown = max_drawdown(equity)

        # calculate the average win, loss and win rate
        pnl = trades_df["pnl"].values
        win_rate, avg_win, avg_loss = win_loss_rates(pnl=pnl)

        # win_rate = (trades_df["pnl"] > 0).mean() if not trades_df.empty else 0
        # avg_win = (
        #     trades_df[trades_df["pnl"] > 0]["pnl"].mean() if not trades_df.empty else 0
        # )
        # avg_loss = (
        #     trades_df[trades_df["pnl"] < 0]["pnl"].mean() if not trades_df.empty else 0
        # )

        # calculate profit factor
        gross_profits = (trades_df["pnl"] > 0).sum() if not trades_df.empty else 0
        gross_losses = (trades_df["pnl"] < 0).sum() if not trades_df.empty else 0
        profit_factor = gross_profits / gross_losses

        return {
            "Total Return (%)": total_return*100,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Beta": portfolio_beta,
            "Asset Betas": asset_betas,
            "Alpha": alpha_,
            "Max Drawdown (%)": max_dd*100,
            "Win Rate (%)": win_rate*100,
            "Avg Win": f"${avg_win:,.2f}",
            "Avg Loss": f"${avg_loss:,.2f}",
            "Profit Factor": profit_factor,
        }
