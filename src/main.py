"""
Main module to run backtesting, paper trading, and live trading

Created by Zach Hatzenbeller 2025-07-06
"""

# Repository Imports
from core.data_loader import DataLoader
from core.data_handler import DataHandler
from core.risk_management import RiskManager
from core.strategy import MARSdx
from trading.backtest import Backtester

# Standard Library Imports
import os
from dotenv import load_dotenv

# Third Party Imports
import pandas as pd
import numpy as np

load_dotenv()

api_key = os.getenv("APCA-API-KEY-ID")
secret_key = os.getenv("APCA-API-SECRET-KEY")

data_retriever = DataLoader(
    api_key=api_key,
    secret_key=secret_key,
    resolution=1,
    time_length=365,
    time_frame_unit="Day",
    asset_class="crypto"
)

bars = data_retriever.load_data(symbols=["BTC/USD"])

data_handler = DataHandler()
risk_manager = RiskManager()
strategy = MARSdx(bars)
back_test = Backtester(
    strategy=strategy,
    risk_manager=risk_manager,
    data_handler=data_handler,
    initial_capital=10_000
)

back_test.run_backtest()

# print(data_handler.trades)
print(data_handler.equity_curve[-1])
print(data_handler.compute_metrics())
