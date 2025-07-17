"""
Main module to run backtesting, paper trading, and live trading

Created by Zach Hatzenbeller 2025-07-06
"""

# Repository Imports
from core.data_loader import DataLoader
from core.data_handler import DataHandler
from core.risk_management import RiskManager
from core.strategy import MARSdx
from core.visualizer import TradingVisualizer
from trading.backtest import Backtester

# Standard Library Imports
import os
from dotenv import load_dotenv

# Third Party Imports


load_dotenv()

api_key = os.getenv("APCA-API-KEY-ID")
secret_key = os.getenv("APCA-API-SECRET-KEY")


symbols = ["NVDA", "AAPL", "AMZN", "MSFT"]  # BTC/USD


data_retriever = DataLoader(
    api_key=api_key,
    secret_key=secret_key,
    resolution=1,
    time_length=1008,
    time_frame_unit="Day",
    asset_class="stock",
)

bars = data_retriever.load_data(symbols=symbols)

data_handler = DataHandler(market_reference_data=data_retriever.market_reference)
risk_manager = RiskManager(
    max_risk_per_trade=0.01,
    drawdown_limit=0.20,
    leverage=1.0,
    stop_loss_pct=0.02,
    target_price_pct=0.04,
)
strategy = MARSdx(
    data=bars,
    ema_length=2,
    sma_length=10,
    rsi_length=4,
    adx_length=4,
)
back_test = Backtester(
    strategy=strategy,
    risk_manager=risk_manager,
    data_handler=data_handler,
    initial_capital=10_000,
    trade_fee=0.001,
    slippage=0.001,
)

back_test.run_backtest()

# print(data_handler.equity_curve)
# print(data_handler.equity_curve[-1])
print(data_handler.compute_metrics())
print(data_handler.get_trade_df())

visualize_test = TradingVisualizer(backtester=back_test)
visualize_test.plot_equity_curve(ticker=symbols[0])
