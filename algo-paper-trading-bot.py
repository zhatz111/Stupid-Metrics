"""_summary_

Returns:
    _type_: _description_
"""

# standard library imports
import os
import pytz
import time
import smtplib
from io import BytesIO
from typing import Tuple
from pathlib import Path
from datetime import datetime, timedelta

# Imports for sending emails
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

# Third Party Import
import numpy as np
import pandas as pd
from scipy.linalg import solve
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from rich import print as print_
import matplotlib.dates as mdates
from scipy.signal import savgol_filter

# Alpaca Data Imports
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical import CryptoHistoricalDataClient

# Alpaca Trading Imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.data.requests import CryptoLatestQuoteRequest


load_dotenv()


def one_sided_savitzky_golay(y, window_length, polyorder, deriv=0):
    """
    Applies a one-sided Savitzky-Golay filter to the input array.

    Args:
        y (np.ndarray): The input array to filter.
        window_length (int): The size of the filter window (must be odd).
        polyorder (int): The polyorder of the polynomial to fit.
        deriv (int, optional): The polyorder of the deriv to compute
                                     (default is 0 for smoothing).

    Returns:
        np.ndarray: The filtered array or deriv.
    """
    if window_length % 2 != 1 or window_length < 1:
        raise ValueError("Window size must be a positive odd number")
    if window_length < polyorder + 1:
        raise ValueError("Window size is too small for the polynomial polyorder")

    half_window = (window_length - 1) // 2

    # Construct the Vandermonde design matrix
    x = np.arange(window_length)
    V = np.stack([x**i for i in range(polyorder + 1)], axis=1)

    # Calculate filter coefficients
    try:
        coeffs = solve(V.T @ V, V.T)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Singular matrix: try decreasing polynomial polyorder or window size"
        )

    # Extract the deriv coefficients
    if deriv > polyorder:
        raise ValueError("deriv polyorder is too high for the polynomial polyorder")

    deriv_coeffs = coeffs[deriv]

    # Apply the filter to each point, considering the one-sided window
    filtered_values = np.zeros_like(y)
    for i in range(len(y)):
        start = max(0, i - window_length + 1)
        end = i + 1
        window = y[start:end]

        if len(window) < window_length:
            pad_size = window_length - len(window)
            padded_window = np.pad(window, (pad_size, 0), "edge")
            filtered_values[i] = np.dot(deriv_coeffs, padded_window)
        else:
            filtered_values[i] = np.dot(deriv_coeffs, window)

    return filtered_values


class AlgoTrading:
    """_summary_"""

    def __init__(
        self,
        api_key: str,
        api_secrets: str,
        investment: float,
        resolution: int,
        time_length: int,
        first_mov_avg: int,
        deriv: int,
        second_mov_avg: int,
        deriv_cutoff: float,
        symbol: str,
        win_length: int,
        data_path: Path,
        save_path: Path,
    ):
        self.api_key = api_key
        self.api_secrets = api_secrets
        self.investment = investment
        self.initial_investment = investment
        self.resolution = resolution
        self.time_length = time_length
        self.first_mov_avg = first_mov_avg
        self.second_mov_avg = second_mov_avg
        self.deriv_cutoff = deriv_cutoff
        self.symbol = symbol
        self.win_length = win_length
        self.data_path = data_path
        self.deriv = deriv
        self.curr_order = {}
        self.has_position = False
        self.save_path = save_path
        self.tmpfile = BytesIO()
        self.buying_power = 0
        self.cash = 0
        self.bid_price = 0
        self.ask_price = 0

    def get_crypto_data(self) -> Tuple[pd.DataFrame, float]:
        """
        This function gets crypto data for the user based on the
        resolution, time lenght and crypto symbol specified

        Returns:
            Tuple[pd.DataFrame, float]: returns a tuple of a dataframe
            with crypto bar data and the current price of the asset
        """
        time_frame = TimeFrame(self.resolution, TimeFrameUnit("Min"))
        start_time = datetime.now(pytz.utc) - timedelta(days=self.time_length)
        end_time = datetime.now(pytz.utc)

        # No keys required for crypto data
        client = CryptoHistoricalDataClient()

        # Creating request object
        request_params = CryptoBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=time_frame,
            start=start_time,
            end=end_time,
        )

        # Retrieve daily bars for Bitcoin in a DataFrame and printing it
        bars = client.get_crypto_bars(request_params)
        # Convert to dataframe
        data = bars.df.reset_index()

        self.first_mov_avg_res = (
            self.first_mov_avg * int(60 / self.resolution) * 24
        )  # minute
        self.second_mov_avg_res = (
            self.second_mov_avg * int(60 / self.resolution) * 24
        )  # minute

        # using the second derivative instead of the first as it better approximates the peaks and dips
        data["Moving Avg (First)"] = (
            data["close"].rolling(int(self.first_mov_avg_res)).mean()
        )
        data["Moving Avg (First) STD"] = (
            data["close"].rolling(int(self.first_mov_avg_res)).std()
        )
        data["Moving Avg (First) Deriv"] = savgol_filter(
            data["Moving Avg (First)"],
            window_length=self.win_length,
            polyorder=2,
            deriv=self.deriv,
        )

        # using the second derivative instead of the first as it better approximates the peaks and dips
        data["Moving Avg (Second)"] = (
            data["close"].rolling(int(self.second_mov_avg_res)).mean()
        )
        data["Moving Avg (Second) STD"] = (
            data["close"].rolling(int(self.second_mov_avg_res)).std()
        )
        data["Moving Avg (Second) Deriv"] = savgol_filter(
            data["Moving Avg (Second)"],
            window_length=self.win_length,
            polyorder=2,
            deriv=self.deriv,
        )

        curr_row = data.iloc[-1, :]

        return data, curr_row

    def get_account_details(self):
        trading_client = TradingClient(self.api_key, self.api_secrets, paper=True)

        # Get our account information.
        account = trading_client.get_account()

        # Check if our account is restricted from trading.
        if account.trading_blocked:
            raise ValueError("Account is currently restricted from trading.")

        self.buying_power = float(account.buying_power)
        self.cash = float(account.cash)

    def graph_data(self, save_path, time_len=5):
        data_, _ = self.get_crypto_data()
        start_time = datetime.now(pytz.utc) - timedelta(days=time_len)
        end_time = datetime.now(pytz.utc)
        graph_data = data_[data_["timestamp"] > start_time]

        save_path.mkdir(parents=True, exist_ok=True)
        order_data = self.import_order_data(all_orders=True)

        if not isinstance(order_data, pd.DataFrame):
            return False

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 12))
        order_dict = order_data.to_dict("index")

        sum_profit = 0
        successful_trades = 0
        best_trade = -np.inf

        failed_trades = 0
        worst_trade = np.inf

        ax[0].plot(
            graph_data["timestamp"],
            graph_data["close"],
            c="darkgray",
            label=f"Close Price ({self.resolution} minute)",
        )

        ax[0].plot(
            graph_data["timestamp"],
            graph_data["Moving Avg (First)"],
            c="dodgerblue",
            linestyle="--",
            label=f"Moving Average ({self.first_mov_avg_res / (int(60 / self.resolution) * 24):0.02f} days)",
        )

        ax[0].plot(
            graph_data["timestamp"],
            graph_data["Moving Avg (Second)"],
            c="chocolate",
            linestyle="--",
            label=f"Moving Average ({self.second_mov_avg_res / (int(60 / self.resolution) * 24):0.02f} days)",
        )

        ax[0].grid()
        ax[0].legend()
        ax[0].set_xlim(start_time, end_time)
        # ax[0].set_ylim(min(graph_data["close"])-min(graph_data["close"])*0.2, max(graph_data["close"])+max(graph_data["close"])*0.2)
        ax[0].set_xlabel("Datetime")
        ax[0].set_ylabel(f"{self.symbol}")

        ax[0].tick_params(axis="x", labelrotation=60)

        ax[1].plot(
            graph_data["timestamp"],
            graph_data["Moving Avg (First) Deriv"],
            c="darkgray",
            label=f"Moving Average ({self.first_mov_avg_res / (int(60 / self.resolution) * 24)} days)",
        )
        ax[1].axhline(0, linestyle="--", c="k")

        ax[1].grid()
        ax[1].set_xlim(start_time, end_time)
        # ax[1].set_ylim(min(graph_data["Moving Avg (First) Deriv"])*1.2, max(graph_data["Moving Avg (First) Deriv"])*1.2)
        ax[1].set_xlabel("Datetime")

        if self.deriv == 1:
            ax[1].set_ylabel("First Derivative")
        elif self.deriv == 2:
            ax[1].set_ylabel("Second Derivative")

        ax[1].tick_params(axis="x", labelrotation=60)

        for _, data in order_dict.items():
            if (
                datetime.strptime(data["Buy Datetime"], "%Y-%m-%d %H:%M:%S %z")
                > start_time
            ):
                if not bool(np.isnan(data["Total Profit"])):
                    if data["Total Profit"] >= 0:
                        successful_trades += 1
                        sum_profit += data["Total Profit"]
                        if best_trade < data["Total Profit"]:
                            best_trade = data["Total Profit"]

                        ax[0].axvline(
                            datetime.strptime(
                                data["Buy Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            linestyle="--",
                            c="green",
                        )
                        ax[1].axvline(
                            datetime.strptime(
                                data["Buy Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            linestyle="--",
                            c="green",
                        )
                        ax[0].axvline(
                            datetime.strptime(
                                data["Sell Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            linestyle="--",
                            c="green",
                        )
                        ax[1].axvline(
                            datetime.strptime(
                                data["Sell Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            linestyle="--",
                            c="green",
                        )
                        ax[0].axvspan(
                            datetime.strptime(
                                data["Buy Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            datetime.strptime(
                                data["Sell Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            linestyle="--",
                            facecolor="palegreen",
                            alpha=0.3,
                        )
                        ax[1].axvspan(
                            datetime.strptime(
                                data["Buy Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            datetime.strptime(
                                data["Sell Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            linestyle="--",
                            facecolor="palegreen",
                            alpha=0.3,
                        )
                    else:
                        failed_trades += 1
                        sum_profit += data["Total Profit"]
                        if worst_trade > data["Total Profit"]:
                            worst_trade = data["Total Profit"]

                        ax[0].axvline(
                            datetime.strptime(
                                data["Buy Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            linestyle="--",
                            c="red",
                        )
                        ax[1].axvline(
                            datetime.strptime(
                                data["Buy Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            linestyle="--",
                            c="red",
                        )
                        ax[0].axvline(
                            datetime.strptime(
                                data["Sell Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            linestyle="--",
                            c="red",
                        )
                        ax[1].axvline(
                            datetime.strptime(
                                data["Sell Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            linestyle="--",
                            c="red",
                        )
                        ax[0].axvspan(
                            datetime.strptime(
                                data["Buy Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            datetime.strptime(
                                data["Sell Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            linestyle="--",
                            facecolor="lightcoral",
                            alpha=0.3,
                        )
                        ax[1].axvspan(
                            datetime.strptime(
                                data["Buy Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            datetime.strptime(
                                data["Sell Datetime"], "%Y-%m-%d %H:%M:%S %z"
                            ),
                            linestyle="--",
                            facecolor="lightcoral",
                            alpha=0.3,
                        )
                else:
                    ax[0].axvline(
                        datetime.strptime(data["Buy Datetime"], "%Y-%m-%d %H:%M:%S %z"),
                        linestyle="--",
                        c="green",
                    )
                    ax[1].axvline(
                        datetime.strptime(data["Buy Datetime"], "%Y-%m-%d %H:%M:%S %z"),
                        linestyle="--",
                        c="green",
                    )
        title = (
            f"{self.symbol} | {successful_trades} Profitable Trades (Best: ${best_trade:,.2f}), "
            f"{failed_trades} Unprofitable Trades (Worst: ${worst_trade:,.2f}) | "
            f"Total Profit: ${sum_profit:,.2f} over {time_len} days"
        )

        title_2 = (
            f"Resolution: {self.resolution}, First Moving Average: {self.first_mov_avg:0.02f}, Second Moving Average: {self.second_mov_avg:0.02f}, "
            f"Derivative Cutoff: {self.deriv_cutoff:0.02f}, Savgol Window Length: {self.win_length}, Derivative: {self.deriv}"
        )

        ax[0].set_title(title)
        ax[1].set_title(title_2)

        # Set x-axis to display hour ticks
        hours = mdates.HourLocator(interval=4)  # Ticks every hour
        h_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M")  # Format as HH:MM

        ax[0].xaxis.set_major_locator(hours)
        ax[0].xaxis.set_major_formatter(h_fmt)

        ax[1].xaxis.set_major_locator(hours)
        ax[1].xaxis.set_major_formatter(h_fmt)

        fig.tight_layout()

        fig.savefig(
            Path(
                save_path,
                f"{self.symbol.replace('/', '')}_{datetime.now().strftime('%Y-%m-%d_%H%M')}_EST.png",
            ),
            format="png",
        )
        fig.savefig(self.tmpfile, format="png")

        return True

    def send_email_update(self, buy=True):
        # Email credentials
        SMTP_SERVER = os.getenv("MAIL_SERVER")
        SMTP_PORT = os.getenv("MAIL_PORT")
        EMAIL_ADDRESS = os.getenv("MAIL_USERNAME")
        EMAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
        RECIPIENT = os.getenv("RECIPIENT")

        msg = MIMEMultipart()

        saved_status = self.graph_data(save_path=self.save_path)

        if saved_status:
            # Attach image
            self.tmpfile.seek(0)
            image_attachment = MIMEImage(
                self.tmpfile.read(),
                name=f"{self.symbol.replace('/', '')}_{datetime.now().strftime('%Y-%m-%d_%H%M')}_EST.png",
            )
            msg.attach(image_attachment)

        # Email details
        subject = f"Trade Execution {datetime.now().strftime('%Y-%m-%d %H:%M')} EST"

        if buy:
            html_content = f"""\
            <html>
                <body>
                    <h3>Buy Executed:</h3>
                    <h3><span style="color:orange;">{self.curr_order["Buy Datetime"]}</span></h3>
                    <h3>Quantity: <span style="color:orange;">{self.curr_order["Buy Quantity"]:0.04f}</span></h3>
                    <h3>Current Buying Power: <span style="color:green;">{self.buying_power:,.02f}</span></h3>
                    <h3>Current Cash: <span style="color:green;">{self.cash:,.02f}</span></h3>
                    <p>------------------------------------------</p>
                    <p>Symbol: <span style="color:dodgerblue;">{self.curr_order["Symbol"]}</span></p>
                    <p>Buy Order ID: <span style="color:magenta;">{self.curr_order["Buy Order ID"]}</span></p>
                    <p>Buy Datetime: <span style="color:dodgerblue;">{self.curr_order["Buy Datetime"]}</span></p>
                    <p>Buy Price: <span style="color:dodgerblue;">${self.curr_order["Buy Price"]:,.02f}</span></p>
                    <p>Buy Quantity: <span style="color:dodgerblue;">{self.curr_order["Buy Quantity"]:0.04f}</span></p>
                    <p>Derivative ({self.deriv}) at Purchase: <span style="color:dodgerblue;">{self.curr_order["Deriv at Purchase"]:0.04f}</span></p>
                    <p><span style="font-weight:bold;">Current Bid Price:</span> <span style="color:green;font-weight:bold;">${self.bid_price:,.02f}</span></p>
                    <p><span style="font-weight:bold;">Current Ask Price:</span> <span style="color:green;font-weight:bold;">${self.ask_price:,.02f}</span></p>
                </body>
            </html>
            """
        else:
            if self.curr_order["Total Profit"] > 0:
                profit_color = "green"
            else:
                profit_color = "red"
            html_content = f"""\
            <html>
                <body>
                    <h3>Sell Executed:</h3>
                    <h3><span style="color:{profit_color};">{self.curr_order["Sell Datetime"]}</span></h3>
                    <h3>Quantity: <span style="color:{profit_color};">{self.curr_order["Quantity Sold"]:0.04f}</span>,
                    Profit: <span style="color:{profit_color};">${self.curr_order["Total Profit"]:,.02f}</span></h3>
                    <h3>Current Buying Power: <span style="color:green;">{self.buying_power:,.02f}</span></h3>
                    <h3>Current Cash: <span style="color:green;">{self.cash:,.02f}</span></h3>
                    <p>------------------------------------------</p>
                    <p>Symbol: <span style="color:dodgerblue;">{self.curr_order["Symbol"]}</span></p>
                    <p>Buy Order ID: <span style="color:magenta;">{self.curr_order["Buy Order ID"]}</span></p>
                    <p>Buy Datetime: <span style="color:dodgerblue;">{self.curr_order["Buy Datetime"]}</span></p>
                    <p>Buy Price: <span style="color:{profit_color};">${self.curr_order["Buy Price"]:,.02f}</span></p>
                    <p>Buy Quantity: <span style="color:dodgerblue;">{self.curr_order["Buy Quantity"]:0.04f}</span></p>
                    <p>Derivative ({self.deriv}) at Purchase: <span style="color:dodgerblue;">{self.curr_order["Deriv at Purchase"]:0.04f}</span></p>
                    <p>Sell Order ID: <span style="color:magenta;">{self.curr_order["Sell Order ID"]}</span></p>
                    <p>Sell Datetime: <span style="color:dodgerblue;">{self.curr_order["Sell Datetime"]}</span></p>
                    <p>Sell Price: <span style="color:{profit_color};">${self.curr_order["Sell Price"]:,.02f}</span></p>
                    <p>Quantity Sold: <span style="color:dodgerblue;">{self.curr_order["Quantity Sold"]:0.04f}</span></p>
                    <p>Percent Change: <span style="color:dodgerblue;">{self.curr_order["Percent Change"]:0.04f}</span></p>
                    <p>Total Profit: <span style="color:{profit_color};">{self.curr_order["Total Profit"]:,.02f}</span></p>
                    <p>Derivative ({self.deriv}) at Sell: <span style="color:dodgerblue;">{self.curr_order["Deriv at Sell"]:0.04f}</span></p>
                    <p><span style="font-weight:bold;">Current Bid Price:</span> <span style="color:green;font-weight:bold;">${self.bid_price:,.02f}</span></p>
                    <p><span style="font-weight:bold;">Current Ask Price:</span> <span style="color:green;font-weight:bold;">${self.ask_price:,.02f}</span></p>
                </body>
            </html>
            """

        msg["From"] = EMAIL_ADDRESS
        msg["To"] = RECIPIENT
        msg["Subject"] = subject
        msg.attach(MIMEText(html_content, "html"))

        # Send email
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()  # Secure connection
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.sendmail(EMAIL_ADDRESS, RECIPIENT, msg.as_string())
            print("Email sent successfully!")
        except Exception as e:
            print(f"Error: {e}")

    def execute_trade(self, cash_amount: float):
        """_summary_

        Args:
            quantity (float): amount of asset to purchase

        Returns:
            _type_: returns the submitted order to alpaca
        """
        trading_client = TradingClient(self.api_key, self.api_secrets, paper=True)

        # preparing market order
        market_order_request = MarketOrderRequest(
            symbol=self.symbol,
            notional=cash_amount,
            side=OrderSide.BUY,
            type="market",
            time_in_force=TimeInForce.GTC,
        )

        # Market order
        return trading_client.submit_order(order_data=market_order_request)

    def sell_position(self):
        """_summary_

        Args:
            quantity (float): amount of asset to purchase

        Returns:
            _type_: returns the submitted sell order to alpaca
        """

        trading_client = TradingClient(self.api_key, self.api_secrets, paper=True)

        position = trading_client.get_open_position(self.symbol.replace("/", ""))
        quantity = position.qty

        # preparing market order
        sell_order_request = MarketOrderRequest(
            symbol=self.symbol,
            qty=quantity,
            side=OrderSide.SELL,
            type="market",
            time_in_force=TimeInForce.GTC,
        )

        # Market order
        return trading_client.submit_order(order_data=sell_order_request)

    def get_last_order_data(self):
        trading_client = TradingClient(self.api_key, self.api_secrets, paper=True)

        # Get the last 100 closed orders
        get_orders_data = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=1,
            nested=True,  # show nested multi-leg orders
            symbols=[self.symbol],
        )

        return trading_client.get_orders(filter=get_orders_data)

    def reset_order(self):
        self.curr_order = {
            "Symbol": "",
            "Buy Order ID": "",
            "Buy Datetime": "",
            "Buy Price": "",
            "Buy Quantity": "",
            "Deriv at Purchase": "",
            "Sell Order ID": "",
            "Sell Datetime": "",
            "Sell Price": "",
            "Quantity Sold": "",
            "Percent Change": "",
            "Total Profit": "",
            "Deriv at Sell": "",
            "Close Price at Sell": "",
            "Slippage": "",
        }

    def reset_fig_data(self):
        self.tmpfile = BytesIO()

    def get_latest_quote(self):
        client = CryptoHistoricalDataClient()
        quote_request = CryptoLatestQuoteRequest(symbol_or_symbols=[self.symbol])
        latest_quote = client.get_crypto_latest_quote(quote_request)

        return latest_quote[self.symbol].bid_price, latest_quote[self.symbol].ask_price

    def import_order_data(self, all_orders=False):
        order_file_path = Path(
            self.data_path, f"{self.symbol.replace('/', '')}_order_data.csv"
        )

        check_last_order = self.get_last_order_data()

        if order_file_path.is_file() and check_last_order:
            df_order = pd.read_csv(order_file_path)

            if check_last_order[0].side == "buy" and (
                df_order.iloc[-1, :]["Buy Order ID"]
                == check_last_order[0].client_order_id
            ):
                self.has_position = True
                self.curr_order = df_order.iloc[-1, :].to_dict()
            else:
                self.has_position = False
                self.curr_order = df_order.iloc[-1, :].to_dict()
            if all_orders:
                return df_order
        elif check_last_order and (check_last_order[0].side == "buy"):
            _, curr_row = self.get_crypto_data()
            self.reset_order()
            self.curr_order["Symbol"] = check_last_order[0].symbol
            self.curr_order["Buy Order ID"] = check_last_order[0].client_order_id
            self.curr_order["Buy Datetime"] = check_last_order[0].filled_at.strftime(
                "%Y-%m-%d %H:%M:%S %z"
            )
            self.curr_order["Buy Price"] = float(check_last_order[0].filled_avg_price)
            self.curr_order["Buy Quantity"] = float(check_last_order[0].filled_qty)
            self.curr_order["Deriv at Purchase"] = float(
                curr_row["Moving Avg (First) Deriv"]
            )
        else:
            self.reset_order()
        return []

    def export_data(self, historic_only=False):
        # Fetch historic data
        data, curr_row = self.get_crypto_data()

        if not bool(self.curr_order):
            self.reset_order()

        # Define file paths
        historic_file_path = Path(
            self.data_path, f"{self.symbol.replace('/', '')}_historic_data.csv"
        )
        order_file_path = Path(
            self.data_path, f"{self.symbol.replace('/', '')}_order_data.csv"
        )

        if historic_file_path.is_file():
            df_historic = pd.read_csv(historic_file_path)
            curr_row_df = curr_row.to_frame().T  # Transpose to make it a row
            time_str = df_historic.iloc[-1, :]["timestamp"]
            time_str_fixed = (
                time_str[:-3] + time_str[-2:]
            )  # Convert '+00:00' to '+0000'
            dt = datetime.strptime(time_str_fixed, "%Y-%m-%d %H:%M:%S%z")
            if curr_row["timestamp"].to_pydatetime() != dt:
                data = pd.concat([df_historic, curr_row_df], ignore_index=True)
        data.to_csv(historic_file_path, index=False)  # Ensure historic data is saved

        if historic_only:
            return  # Avoid redundant writes

        # Avoid writing empty orders
        if not all(value == "" for value in self.curr_order.values()):
            df_order = pd.DataFrame([self.curr_order])
        else:
            df_order = pd.DataFrame()  # Empty dataframe to prevent issues

        # Load existing order data
        if order_file_path.is_file():
            df_existing = pd.read_csv(order_file_path)

            buy_order_id = self.curr_order.get("Buy Order ID")
            if buy_order_id and buy_order_id in df_existing["Buy Order ID"].values:
                df_existing.loc[
                    df_existing["Buy Order ID"] == buy_order_id, df_order.columns
                ] = df_order.values
            elif not df_order.empty:
                df_existing = pd.concat([df_existing, df_order], ignore_index=True)
        else:
            df_existing = df_order  # Only store new order if file doesn't exist

        # Save updated order data
        if not df_existing.empty:
            df_existing.to_csv(order_file_path, index=False)

    def mean_reversion_crypto_algo(self):
        """_summary_"""
        # get the historical crypto data from the alpaca api
        _, curr_row = self.get_crypto_data()

        # In case script is interuppted or data is lost check to see if
        # the last order was a buy or sell to determine if holding a current
        # position or not

        _ = self.import_order_data()
        self.get_account_details()
        self.bid_price, self.ask_price = self.get_latest_quote()

        print_("-------------------------------------------------")
        print_(f"[bold blue]Execution Time: {datetime.now()} EST[/]")

        if (
            (curr_row["Moving Avg (First)"] > curr_row["Moving Avg (Second)"])
            and not self.has_position
            and (curr_row["Moving Avg (First) Deriv"] > 0)
        ):
            self.has_position = True
            self.reset_order()

            # add logic to execute order then fill in order details after it is accepted
            # qty = (self.cash/2) / curr_row["close"]

            # Execute the trades based on the condition
            # Only use 95% of the available cash in the account to trade with
            _ = self.execute_trade(cash_amount=round(self.cash * 0.95, 2))
            time.sleep(2)
            last_order_algo = self.get_last_order_data()

            self.curr_order["Symbol"] = last_order_algo[0].symbol
            self.curr_order["Buy Order ID"] = last_order_algo[0].client_order_id
            self.curr_order["Buy Datetime"] = last_order_algo[0].filled_at.strftime(
                "%Y-%m-%d %H:%M:%S %z"
            )
            self.curr_order["Buy Price"] = float(last_order_algo[0].filled_avg_price)
            self.curr_order["Buy Quantity"] = float(last_order_algo[0].filled_qty)
            self.curr_order["Deriv at Purchase"] = float(
                curr_row["Moving Avg (First) Deriv"]
            )
            print_(
                f"New Order Bought at [bold green]{self.curr_order['Buy Datetime']}[/], Quantity: [bold green]{self.curr_order['Buy Quantity']:0.04f}[/]"
            )
            self.get_account_details()
            self.export_data()
            self.send_email_update(buy=True)
            self.reset_fig_data()

        if self.has_position:
            if (
                curr_row["Moving Avg (First) Deriv"]
                < self.curr_order["Deriv at Purchase"] * self.deriv_cutoff
            ):
                # Sell the current position
                _ = self.sell_position()
                time.sleep(2)
                position_sold = self.get_last_order_data()

                self.curr_order["Sell Order ID"] = position_sold[0].client_order_id
                self.curr_order["Sell Datetime"] = position_sold[0].filled_at.strftime(
                    "%Y-%m-%d %H:%M:%S %z"
                )
                self.curr_order["Sell Price"] = float(position_sold[0].filled_avg_price)
                self.curr_order["Quantity Sold"] = float(position_sold[0].filled_qty)
                self.curr_order["Close Price at Sell"] = curr_row["close"]
                self.curr_order["Slippage"] = (
                    100
                    * (self.curr_order["Close Price at Sell"] - self.curr_order["Sell Price"])
                    / self.curr_order["Close Price at Sell"]
                )
                self.curr_order["Percent Change"] = (
                    100
                    * (self.curr_order["Sell Price"] - self.curr_order["Buy Price"])
                    / self.curr_order["Buy Price"]
                )
                self.curr_order["Total Profit"] = (
                    self.curr_order["Percent Change"] * self.investment / 100
                )
                self.curr_order["Deriv at Sell"] = curr_row["Moving Avg (First) Deriv"]

                self.investment += self.curr_order["Total Profit"]

                if self.curr_order["Total Profit"] > 0:
                    print_(
                        f"Current Order Sold at [bold green]{self.curr_order['Sell Datetime']}[/], Quantity: [bold green]{self.curr_order['Quantity Sold']:0.04f}[/], Profit: [bold green]${self.curr_order['Total Profit']:,.02f}[/]"
                    )
                else:
                    print_(
                        f"Current Order Sold at [bold red]{self.curr_order['Sell Datetime']}[/], Quantity: [bold red]{self.curr_order['Quantity Sold']:0.04f}[/], Profit: [bold red]${self.curr_order['Total Profit']:,.02f}[/]"
                    )
                self.has_position = False
                self.get_account_details()
                self.export_data()
                self.send_email_update(buy=False)
                self.reset_fig_data()

        print_(f"Current Buying Power: [bold green]${self.buying_power:,.02f}[/]")
        print_(f"Current Cash: [bold green]${self.cash:,.02f}[/]")
        print_("-------------------------------------------------")
        for key, value in self.curr_order.items():
            if isinstance(value, float):
                print_(f"{key}: {value:,.04f}")
            else:
                print_(f"{key}: {value}")
        print_("-------------------------------------------------")
        print_(f"Close Price: [bold green]${curr_row['close']:,.02f}[/]")
        print_(
            f"Moving Avg. {self.first_mov_avg_res / (int(60 / self.resolution) * 24):0.02f} day: [bold green]${curr_row['Moving Avg (First)']:,.02f}[/]"
        )
        print_(
            f"Moving Avg. {self.second_mov_avg_res / (int(60 / self.resolution) * 24):0.02f} day: [bold green]${curr_row['Moving Avg (Second)']:,.02f}[/]"
        )
        print_(f"Moving Avg. Deriv: {curr_row['Moving Avg (First) Deriv']:0.04f}")
        print_(
            f"[bold]Current Bid Price:[/bold] [bold green]${self.bid_price:,.02f}[/]"
        )
        print_(
            f"[bold]Current Ask Price:[/bold] [bold green]${self.ask_price:,.02f}[/]"
        )
        print_("-------------------------------------------------")
        print_("")

        self.export_data(historic_only=True)


if __name__ == "__main__":
    # Import Environment Variables Needed
    # Alpaca API keys
    API_KEY = os.getenv("APCA-API-KEY-ID")
    API_SECRET = os.getenv("APCA-API-SECRET-KEY")
    INVESTMENT = 10_000  # dollars
    RESOLUTION = 50  # minute
    TIME_LENGTH = 30  # days
    FIRST_MOV_AVG_DAY = 0.5  # days
    SECOND_MOV_AVG_DAY = 7  # days
    DERIV_CUTOFF = 0.8
    WIN_LENGTH = 9
    DERIVATIVE = 1
    TICKER_SYMBOL = "BTC/USD"
    DATA_FOLDER = "live-paper-v3"

    # Set timezone to Eastern Time
    eastern = pytz.timezone('America/New_York')

    data_path_ = Path(Path.cwd(), "data/mean_reversion_algo/paper_trading", DATA_FOLDER)
    data_path_.mkdir(parents=True, exist_ok=True)

    fig_save_path = Path(data_path_, "figures")
    fig_save_path.mkdir(parents=True, exist_ok=True)

    algo_trading = AlgoTrading(
        api_key=API_KEY,
        api_secrets=API_SECRET,
        investment=INVESTMENT,
        resolution=RESOLUTION,
        time_length=TIME_LENGTH,
        first_mov_avg=FIRST_MOV_AVG_DAY,
        second_mov_avg=SECOND_MOV_AVG_DAY,
        deriv_cutoff=DERIV_CUTOFF,
        symbol=TICKER_SYMBOL,
        win_length=WIN_LENGTH,
        data_path=data_path_,
        save_path=fig_save_path,
        deriv=DERIVATIVE,
    )

    print_(
        f"[bold green]Algo Trading Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EST[/] for Account: [bold cyan]{DATA_FOLDER}[/]"
    )
    algo_trading.mean_reversion_crypto_algo()

    # make sure the job starts at the 10s place so that the
    # api query timestamps are aligned with when the script runs
    while True:
        now = datetime.now(eastern)  # Get current time in EST
    
        # Define trading window: 9:30 AM to 4:00 PM EST
        start_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        end_time = now.replace(hour=16, minute=0, second=0, microsecond=0)

        if now.minute % 10 == 0 and now.second == 5:
            algo_trading.mean_reversion_crypto_algo()
        time.sleep(1)  # Check every second
