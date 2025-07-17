"""
Module for strategy abstract class and different strategy implementations

Created by Zach Hatzenbeller 2025-07-03

Note: In pandas-ta library numpy 2.x versions through an error upon import
due to import line in squeeze_pro.py file which attempts to import: "from numpy import NaN as npNaN"
but in 2.x versions this was changed to "from numpy import nan as npNaN". This was modified manually
in the python file so be aware of this if the fix has not been released yet.
"""

# Standard Library Imports
from abc import ABC, abstractmethod
import pandas as pd
import pandas_ta as ta


class Strategy(ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.indicators = self.required_indicators()
        self.indicators_df = self.compute_indicators()

    @abstractmethod
    def required_indicators(self) -> list[dict]:
        """
        Uses the pandas-ta library for computing technical analysis indicators.
        Must return a list of dictionaries describing required indicators.
        E.g., [{'name': 'rsi', 'params': {'length': 14}}]
        """
        pass

    def helper_compute_indicators(self, group_df) -> pd.DataFrame:
        for ind in self.indicators:
            ind["column_name"] = []  # initialize here
            name = ind["name"]
            params = ind.get("params", {})
            func = getattr(group_df.ta, name)

            result = func(**params, append=True)

            # Capture the resulting column names
            if isinstance(result, pd.Series):
                ind["column_name"].append(result.name)
            elif isinstance(result, pd.DataFrame):
                ind["column_name"].extend(result.columns)
        return group_df
    
    def compute_indicators(self):
        df = self.data.copy()
        return df.groupby("symbol").apply(self.helper_compute_indicators).reset_index(drop=True)

    @abstractmethod
    def generate_signals(self):
        """Generate buy/sell signals"""
        pass

    # @abstractmethod
    # def backtest(self):
    #     """Backtest the strategy logic"""
    #     pass


class MARSdx(Strategy):
    def __init__(
        self,
        data: pd.DataFrame,
        ema_length: int = 7,
        sma_length: int = 50,
        rsi_length: int = 14,
        adx_length: int = 14,
    ):
        self.ema_length = ema_length
        self.sma_length = sma_length
        self.rsi_length = rsi_length
        self.adx_length = adx_length
        super().__init__(data)

    def required_indicators(self):
        return [
            {"name": "ema", "params": {"length": self.ema_length}},
            {"name": "sma", "params": {"length": self.sma_length}},
            {"name": "rsi", "params": {"length": self.rsi_length}},
            {"name": "adx", "params": {"length": self.adx_length}},
        ]

    def generate_signals(self, row):
        ema = next(ind["column_name"][0] for ind in self.indicators if ind["name"] == "ema")
        sma = next(ind["column_name"][0] for ind in self.indicators if ind["name"] == "sma")
        rsi = next(ind["column_name"][0] for ind in self.indicators if ind["name"] == "rsi")
        adx = next(ind["column_name"][0] for ind in self.indicators if ind["name"] == "adx")


        if (
            (row["close"] > row[sma])
            and (row["close"] > row[ema])
            and (row[rsi] > row[adx])
        ):
            return "buy"
        elif row[rsi] < row[adx] and (row["close"] < row[ema]):
            return "sell"
        else:
            return None

    # def backtest(self):
    #     return super().backtest()