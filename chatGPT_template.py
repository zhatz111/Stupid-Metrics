# src/config/settings.py
API_KEY = "your_alpaca_api_key"
SECRET_KEY = "your_alpaca_secret_key"
BASE_URL = "https://paper-api.alpaca.markets"

SYMBOL = "AAPL"
TIMEFRAME = "1Min"
MAX_POSITION_SIZE = 100
STOP_LOSS_PCT = 0.02


# src/core/broker_interface.py
import alpaca_trade_api as tradeapi
from config import settings

class BrokerInterface:
    def __init__(self, paper=True):
        self.api = tradeapi.REST(settings.API_KEY, settings.SECRET_KEY, settings.BASE_URL)

    def get_account(self):
        return self.api.get_account()

    def submit_order(self, symbol, qty, side, type, time_in_force):
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=type,
            time_in_force=time_in_force
        )


# src/core/data_loader.py
import pandas as pd
from config import settings
from alpaca_trade_api.rest import REST

def get_historical_data(symbol, timeframe, limit=1000):
    api = REST(settings.API_KEY, settings.SECRET_KEY, settings.BASE_URL)
    barset = api.get_bars(symbol, timeframe, limit=limit)
    df = barset.df
    return df[df['symbol'] == symbol]


# src/core/strategy.py
def generate_signal(data):
    # Example: simple moving average crossover
    data['sma_short'] = data['close'].rolling(window=5).mean()
    data['sma_long'] = data['close'].rolling(window=20).mean()
    
    if data['sma_short'].iloc[-1] > data['sma_long'].iloc[-1]:
        return 'buy'
    elif data['sma_short'].iloc[-1] < data['sma_long'].iloc[-1]:
        return 'sell'
    return 'hold'


# src/core/risk_management.py
def apply_risk_management(account, price, max_position_size, stop_loss_pct):
    cash = float(account.cash)
    qty = min(max_position_size, int(cash / price))
    stop_price = price * (1 - stop_loss_pct)
    return qty, stop_price


# src/trading/backtest.py
def run_backtest(data, strategy_func):
    signals = []
    for i in range(20, len(data)):
        signal = strategy_func(data.iloc[:i])
        signals.append(signal)
    data['signal'] = ['hold'] * 20 + signals
    return data


# src/trading/paper_trade.py
from core.broker_interface import BrokerInterface
from core.strategy import generate_signal
from core.data_loader import get_historical_data
from core.risk_management import apply_risk_management
from utils.logger import get_logger

logger = get_logger(__name__)

broker = BrokerInterface()
data = get_historical_data("AAPL", "1Min")
signal = generate_signal(data)
account = broker.get_account()

if signal in ['buy', 'sell']:
    qty, stop_price = apply_risk_management(account, data['close'].iloc[-1], 100, 0.02)
    broker.submit_order("AAPL", qty, signal, "market", "gtc")
    logger.info(f"Submitted {signal} order for {qty} shares at stop {stop_price}")


# src/trading/live_trade.py
# Similar to paper_trade.py, but connect to live endpoint and add extra safeguards


# src/optimization/optimizer.py
def optimize_strategy(data, strategy_func, param_grid):
    best_score = float('-inf')
    best_params = None
    for params in param_grid:
        score = backtest_score(data, lambda d: strategy_func(d, **params))
        if score > best_score:
            best_score = score
            best_params = params
    return best_params


def backtest_score(data, strategy_func):
    results = run_backtest(data, strategy_func)
    return results['signal'].value_counts().get('buy', 0) - results['signal'].value_counts().get('sell', 0)


# src/utils/logger.py
import logging
import os
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        base_dir = Path(__file__).resolve().parent.parent
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(logs_dir / "trading.log")
        file_handler.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


# src/utils/metrics.py
def calculate_sharpe(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()


# src/utils/helpers.py
import time
import functools

def retry(exception_to_check, tries=3, delay=2):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(tries):
                try:
                    return func(*args, **kwargs)
                except exception_to_check as e:
                    time.sleep(delay)
            raise
        return wrapper
    return decorator_retry


# src/main.py
from trading.paper_trade import broker, data, signal

if __name__ == "__main__":
    print("Running trading bot...")
    print(f"Latest signal: {signal}")
