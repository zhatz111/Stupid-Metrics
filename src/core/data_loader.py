"""
Module to load candle bar data needed for trading

Created by Zach Hatzenbeller 2025-07-06
"""


# Repository Imports


# Standard Library Imports
import pytz
from typing import List
from datetime import datetime, timedelta

# Third Party Imports
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient


class DataLoader:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        resolution: int,
        time_length: int,
        time_frame_unit: str = "Day",
        asset_class: str = "stock",
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.asset_class = asset_class
        self.resolution = resolution
        self.time_length = time_length
        self.time_frame_unit = time_frame_unit

        # Time Specifications for data
        self.time_frame = TimeFrame(self.resolution, TimeFrameUnit(self.time_frame_unit))
        self.start_time = datetime.now(pytz.utc) - timedelta(days=self.time_length)
        self.end_time = datetime.now(pytz.utc) - timedelta(days=1)

    def load_data(self, symbols: List[str]):
        
        if self.asset_class == "stock":
            # Creating request object
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=self.time_frame,
                start=self.start_time,
                end=self.end_time,
            )

            client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
            bars = client.get_stock_bars(request_params)
        elif self.asset_class == "crypto":
            # Creating request object
            request_params = CryptoBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=self.time_frame,
                start=self.start_time,
                end=self.end_time,
            )

            client = CryptoHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
            bars = client.get_crypto_bars(request_params)
        else:
            ValueError("Please specify valid data class to retrieve bars from.")

        # Convert to dataframe
        return bars.df.reset_index()

    def get_live_data(self):
        pass