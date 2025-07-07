"""
Module that contains the backtesting class to evaluate various
strategies and return appropriate results regarding strategy performance

Created by Zach Hatzenbeller 2025-07-05
"""

# Repository Imports
from core.strategy import Strategy
from core.risk_management import RiskManager
from core.data_handler import DataHandler, Trade

# Standard Library Imports


# Third Party Imports
import pandas as pd


class Backtester:
    def __init__(
        self,
        strategy: Strategy,
        risk_manager: RiskManager,
        data_handler: DataHandler,
        initial_capital: float = 100000,
        trade_fee: float = 0.005,
        slippage: float = 0.005,
    ):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.data_handler = data_handler
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trade_fee = trade_fee
        self.slippage = slippage
        self.current_position = Trade()

        # Use the data after the indicators have been calculated
        self.data = self.strategy.indicators_df

    def run_backtest(self):
        for idx, row in self.data.iterrows():
            current_price = row["close"]
            signal = self.strategy.generate_signals(row)

            # Step 1: Manage existing position (check SL/TP)
            if self.current_position.type is not None:
                self.manage_open_positions(row, idx)

            # Step 2: Open new position if no current position and signal is valid
            if signal == "buy" and self.current_position.type is None:
                self.execute_trade(signal, row, idx)
            elif signal == "sell" and self.current_position.type is not None:
                self.execute_trade(signal, row, idx)

            # Step 3: Update portfolio value
            if self.current_position.type == "buy":
                self.update_portfolio_value(current_price, idx)

            # Step 4: Check drawdown and exit if breached
            # if not self.risk_manager.check_drawdown_limit(
            #     self.data_handler.equity_curve,
            #     self.initial_capital,
            #     self.current_capital,
            # ) and not has_position:
            #     print(f"Max drawdown hit. Ending backtest at index {idx}")
            #     break


    def execute_trade(self, signal: str, row: pd.Series, index: int):
        current_price = row["close"]
        # stop/take-profit logic
        stop_price = current_price * (1 - self.risk_manager.stop_loss_pct)
        take_profit_price = current_price * (1 + self.risk_manager.target_price_pct) 

        # Validate the trade with RiskManager
        validated, cost = self.risk_manager.validate_trade(self.current_capital)

        if not validated:
            return "Trade is not Validated"
        
        if signal == "buy" and self.current_position.type is not None:
            return "Already has position"

        # Create the position
        if signal == "buy":
            self.current_position.ticker = row["symbol"]
            self.current_position.type = "open"
            self.current_position.entry_datatime = row["timestamp"]
            self.current_position.entry_price = current_price * (1+self.slippage)
            self.current_position.quantity = (cost / current_price)
            self.current_position.cost = cost
            self.current_position.stop_loss = stop_price
            self.current_position.target_price = take_profit_price
            self.current_position.fees = cost * self.trade_fee
            self.current_position.slippage = current_price * self.slippage
            self.current_position.entry_signal = {}
            self.current_capital -= cost  # subtract initial capital

        elif signal == "sell":
            self.current_position.type = "closed"
            self.current_position.exit_datatime = row["timestamp"]
            self.current_position.exit_price = current_price * (1-self.slippage)
            self.current_position.fees += self.current_position.cost * self.trade_fee
            self.current_position.slippage += current_price * self.slippage
            self.current_position.pnl = (
                (self.current_position.quantity * self.current_position.exit_price)
                - (self.current_position.quantity * self.current_position.entry_price)
                - self.current_position.fees
            )
            self.current_capital += self.current_position.cost  # subtract initial capital
            self.current_capital += self.current_position.pnl  # add the profit/loss to capital
            self.update_portfolio_value(self.current_position.exit_price, index)
            self.data_handler.log_trade(self.current_position)
            self.current_position = Trade()

    def manage_open_positions(self, row: pd.Series, index: int):
        current_price = row["close"]

        hit_sl = self.risk_manager.enforce_stop_loss(
            current_price, self.current_position.stop_loss
        )
        hit_tp = self.risk_manager.enforce_take_profit(
            current_price, self.current_position.target_price
        )

        if hit_sl or hit_tp:
            self.current_position.type = "closed"
            self.current_position.exit_datatime = row["timestamp"]
            self.current_position.exit_price = current_price * (1-self.slippage)
            self.current_position.fees += self.current_position.exit_price * self.current_position.quantity * self.trade_fee
            self.current_position.slippage += current_price * self.slippage
            self.current_position.pnl = (
                (self.current_position.quantity * self.current_position.exit_price)
                - (self.current_position.quantity * self.current_position.entry_price)
                - self.current_position.fees
            )
            self.current_capital += self.current_position.cost  # subtract initial capital
            self.current_capital += self.current_position.pnl  # add the profit/loss to capital
            self.update_portfolio_value(self.current_position.exit_price, index)
            self.data_handler.log_trade(self.current_position)
            self.current_position = Trade()

    def update_portfolio_value(self, current_price, index):
        # Assume all open positions are valued at current price
        open_value = self.current_position.quantity * current_price
        total_value = self.current_capital + open_value

        self.data_handler.record_portfolio_value(
            total_value, self.current_capital, index
        )
