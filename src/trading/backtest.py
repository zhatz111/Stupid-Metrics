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
        self.trade_fee = trade_fee
        self.slippage = slippage
        

        # Use the data after the indicators have been calculated
        self.data = self.strategy.indicators_df.dropna()
        self.allocations = len(self.data["symbol"].unique())

        self.current_prices = {}
        self.current_capital = {}
        self.open_positions = {}
        for symbol in self.data["symbol"].unique():
            self.current_prices[symbol] = {"index": 0, "price": 0}

            # pre-allocate capital to each asset being traded
            self.current_capital[symbol] = initial_capital/self.allocations

            # instantiate a trade object for each symbol
            self.open_positions[symbol] = Trade()
        
        self.backtest_len = min(self.data.groupby('symbol').size())
        self.update_portfolio_value()

    def run_backtest(self):
        for idx in range(self.backtest_len):
            for symbol, data in self.data.groupby("symbol"):
                # extract current row from each symbols dataframe
                row = data.iloc[idx]
                current_price = row["close"]
                self.current_prices[symbol] = {"index": idx, "price": current_price}

                signal = self.strategy.generate_signals(row)

                # Step 1: Manage existing position (check SL/TP)
                if self.open_positions[symbol].type is not None:
                    self.manage_open_positions(symbol, row, idx)

                if signal == "buy" and self.open_positions[symbol].type is None:
                    self.execute_trade(symbol, signal, row, idx)
                elif (
                    signal == "sell"
                    and self.open_positions[symbol].type == "open"
                    and not (self.open_positions[symbol].exit_datatime == row["timestamp"])
                ):
                    self.execute_trade(symbol, signal, row, idx)

            

            # if not positions_open_check:
            self.update_portfolio_value()

            for key, value in self.open_positions.items():
                if value.type == "closed":
                    self.open_positions[key] = Trade()

            positions_open_check = all(position.type is None for _, position in self.open_positions.items())

            # Step 4: Check drawdown and exit if breached
            if self.data_handler.equity_curve:
                if not self.risk_manager.check_drawdown_limit(
                    self.data_handler.equity_curve,
                ) and positions_open_check:
                    self.close_all_positions(idx)
                    self.update_portfolio_value()
                    print(f"Max drawdown hit. Ending backtest at index {idx}")
                    break

    def execute_trade(self, symbol, signal: str, row: pd.Series, index: int):
        current_price = row["close"]
        # stop/take-profit logic
        stop_price = current_price * (1 - self.risk_manager.stop_loss_pct)
        take_profit_price = current_price * (1 + self.risk_manager.target_price_pct) 

        # Validate the trade with RiskManager
        validated, cost = self.risk_manager.validate_trade(self.current_capital[symbol], self.allocations)

        if not validated:
            return "Trade is not Validated"
        
        if signal == "buy" and self.open_positions[symbol].type is not None:
            return "Already has position"

        # Create the position
        if signal == "buy":
            self.open_positions[symbol].ticker = row["symbol"]
            self.open_positions[symbol].type = "open"
            self.open_positions[symbol].entry_datatime = row["timestamp"]
            self.open_positions[symbol].entry_close = current_price
            self.open_positions[symbol].entry_price = current_price * (1+self.slippage)
            self.open_positions[symbol].quantity = (cost / current_price)
            self.open_positions[symbol].cost = cost
            self.open_positions[symbol].stop_loss = stop_price
            self.open_positions[symbol].target_price = take_profit_price
            self.open_positions[symbol].fees = cost * self.trade_fee
            self.open_positions[symbol].slippage = current_price * self.slippage
            self.open_positions[symbol].entry_signal = {}
            self.current_capital[symbol] -= cost  # subtract initial capital

        elif signal == "sell":
            self.open_positions[symbol].type = "closed"
            self.open_positions[symbol].exit_datatime = row["timestamp"]
            self.open_positions[symbol].exit_close = current_price
            self.open_positions[symbol].exit_price = current_price * (1-self.slippage)
            self.open_positions[symbol].fees += self.open_positions[symbol].cost * self.trade_fee
            self.open_positions[symbol].slippage += current_price * self.slippage
            self.open_positions[symbol].pnl = (
                (self.open_positions[symbol].quantity * self.open_positions[symbol].exit_price)
                - (self.open_positions[symbol].quantity * self.open_positions[symbol].entry_price)
                - self.open_positions[symbol].fees
            )
            # self.update_portfolio_value(self.open_positions[symbol].exit_price, index)
            self.current_capital[symbol] += self.open_positions[symbol].cost  # subtract initial capital
            self.current_capital[symbol] += self.open_positions[symbol].pnl  # add the profit/loss to capital
            self.data_handler.log_trade(self.open_positions[symbol])

    def manage_open_positions(self, symbol, row: pd.Series, index: int):
        current_price = row["close"]

        hit_sl = self.risk_manager.enforce_stop_loss(
            current_price, self.open_positions[symbol].stop_loss
        )
        hit_tp = self.risk_manager.enforce_take_profit(
            current_price, self.open_positions[symbol].target_price
        )

        if hit_sl or hit_tp:
            self.open_positions[symbol].type = "closed"
            self.open_positions[symbol].exit_datatime = row["timestamp"]
            self.open_positions[symbol].exit_close = current_price
            self.open_positions[symbol].exit_price = current_price * (1-self.slippage)
            self.open_positions[symbol].fees += self.open_positions[symbol].exit_price * self.open_positions[symbol].quantity * self.trade_fee
            self.open_positions[symbol].slippage += current_price * self.slippage
            self.open_positions[symbol].pnl = (
                (self.open_positions[symbol].quantity * self.open_positions[symbol].exit_price)
                - (self.open_positions[symbol].quantity * self.open_positions[symbol].entry_price)
                - self.open_positions[symbol].fees
            )
            # self.update_portfolio_value(self.open_positions[symbol].exit_price, index)
            self.current_capital[symbol] += self.open_positions[symbol].cost  # add back initial capital
            self.current_capital[symbol] += self.open_positions[symbol].pnl  # add the profit/loss to capital
            self.data_handler.log_trade(self.open_positions[symbol])
    
    def close_all_positions(self, index):
        for symbol, trade in self.open_positions.items():
            symbol_data = self.data.copy().loc[self.data["symbol"]==symbol, :].reset_index(drop=True)
            row = symbol_data.iloc[index]
            current_price = row["close"]
            if trade.type == "open":
                self.open_positions[symbol].type = "closed"
                self.open_positions[symbol].exit_datatime = row["timestamp"]
                self.open_positions[symbol].exit_close = current_price
                self.open_positions[symbol].exit_price = current_price * (1-self.slippage)
                self.open_positions[symbol].fees += self.open_positions[symbol].exit_price * self.open_positions[symbol].quantity * self.trade_fee
                self.open_positions[symbol].slippage += current_price * self.slippage
                self.open_positions[symbol].pnl = (
                    (self.open_positions[symbol].quantity * self.open_positions[symbol].exit_price)
                    - (self.open_positions[symbol].quantity * self.open_positions[symbol].entry_price)
                    - self.open_positions[symbol].fees
                )
                # self.update_portfolio_value(self.open_positions[symbol].exit_price, index)
                self.current_capital[symbol] += self.open_positions[symbol].cost  # add back initial capital
                self.current_capital[symbol] += self.open_positions[symbol].pnl  # add the profit/loss to capital
                self.data_handler.log_trade(self.open_positions[symbol])

    def update_portfolio_value(self):
        total_value = 0

        for symbol, value in self.current_prices.items():
            idx = value["index"]
            current_price = value["price"]
            if self.open_positions[symbol].type not in ["closed", None]:
                open_value = self.open_positions[symbol].quantity * current_price
                curr_value = self.current_capital[symbol] + open_value
                self.data_handler.record_portfolio_value(
                    symbol, curr_value, self.current_capital[symbol], idx
                )
                total_value += curr_value
            else:
                total_value += self.current_capital[symbol]
                self.data_handler.record_portfolio_value(
                    symbol, self.current_capital[symbol], self.current_capital[symbol], idx
                )

        self.data_handler.record_equity_value(total_value)
