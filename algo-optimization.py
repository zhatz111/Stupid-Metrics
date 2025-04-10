# standard library imports
import ray
import pytz
import json
import warnings
import itertools
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from datetime import datetime, timedelta

# Third Party Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import savgol_filter

# Alpaca Data Imports
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical import CryptoHistoricalDataClient


class AlgoOptimization:
    """This class is used to optimize algorithmic trading strategies.
    The current parameters to optimize are data resolution, window for
    the first moving average, window for the second moving average,
    window length for the savgol filter, and cutoff for the derivative
    to determine when to sell.
    The parameters when instantiating the class are given in that order:
        - Resoluition
        - First moving average window length
        - Second moving average window length
        - Savgol window length
        - Derivative cutoff
    """

    def __init__(
        self,
        initial_investment: float,
        time_length: int,
        ticker_symbol: str,
        graph_length: int,
        data_folder: str,
        opt_ranges: list = [],
        initial_params: list = [],
    ):
        self.initial_investment = initial_investment
        self.ticker_symbol = ticker_symbol
        self.graph_length = graph_length
        self.data_folder = data_folder
        self.opt_ranges = opt_ranges
        self.results: dict = {}
        self.best_parameters: dict = {}
        self.initial_time = datetime.now(pytz.utc) - timedelta(days=self.time_length)
        self.start_time = self.initial_time + timedelta(days=self.graph_length)
        self.end_time = datetime.now(pytz.utc)

        if isinstance(time_length, list):
            self.time_length = time_length[0]
            self.second_time_length = time_length[1]
            self.third_time_length = time_length[2]
        else:
            self.time_length = time_length


        if len(initial_params) == 5:
            self.resolution = initial_params[0]  # minute
            self.first_mov_avg = initial_params[1]  # days
            self.second_mov_avg = initial_params[2]  # days
            self.win_length = initial_params[3]  # period
            self.deriv_cutoff = initial_params[4]
        else:
            raise ValueError(
                "Initial Parameters given does not match required length. See Class documentation for correct parameters and order."
            )

    def reset_order(self):
        curr_order = {
            "Buy Datetime": "",
            "Buy Price": "",
            "Deriv at Purchase": "",
            "Sell Datetime": "",
            "Sell Price": "",
            "Percent Change": "",
            "Order Profit": "",
            "Deriv at Sell": "",
            "Just Sold": False,
            "Has Position": False,
            "Total Losses": 0,
            "Total Gains": 0,
        }
        return curr_order

    def get_crypto_data(
        self,
        resolution: int,
        start_time: datetime,
        end_time: datetime,
    ) -> Tuple[pd.DataFrame, float]:
        """
        This function gets crypto data for the user based on the
        resolution, time lenght and crypto symbol specified

        Returns:
            Tuple[pd.DataFrame, float]: returns a tuple of a dataframe
            with crypto bar data and the current price of the asset
        """
        if resolution < 60:
            time_frame = TimeFrame(resolution, TimeFrameUnit("Min"))
        else:
            time_frame = TimeFrame(int(resolution / 60), TimeFrameUnit("Hour"))

        # No keys required for crypto data
        client = CryptoHistoricalDataClient()

        # Creating request object
        request_params = CryptoBarsRequest(
            symbol_or_symbols=self.ticker_symbol,
            timeframe=time_frame,
            start=start_time,
            end=end_time,
        )

        # Retrieve daily bars for Bitcoin in a DataFrame and printing it
        bars = client.get_crypto_bars(request_params)
        # Convert to dataframe
        data = bars.df.reset_index()

        return data

    # look into increasing savgol polyorder to 3? Change derivative calc though to np.diff
    def calculate_columns(
        self,
        data,
        resolution,
        first_mov_avg,
        second_mov_avg,
        win_length,
    ):
        first_mov_avg_res = first_mov_avg * int(60 / resolution) * 24  # minute
        second_mov_avg_res = second_mov_avg * int(60 / resolution) * 24  # minute

        # using the second derivative instead of the first as it better approximates the peaks and dips
        data["Moving Avg (First)"] = (
            data["close"].rolling(int(first_mov_avg_res)).mean()
        )
        data["Moving Avg (First) STD"] = (
            data["close"].rolling(int(first_mov_avg_res)).std()
        )
        data["Moving Avg (First) Deriv"] = savgol_filter(
            data["Moving Avg (First)"],
            window_length=win_length,
            polyorder=2,
            deriv=1,
        )
        data["Moving Avg (First) Deriv2"] = savgol_filter(
            data["Moving Avg (First)"],
            window_length=win_length,
            polyorder=2,
            deriv=2,
        )

        # using the second derivative instead of the first as it better approximates the peaks and dips
        data["Moving Avg (Second)"] = (
            data["close"].rolling(int(second_mov_avg_res)).mean()
        )
        data["Moving Avg (Second) STD"] = (
            data["close"].rolling(int(second_mov_avg_res)).std()
        )
        data["Moving Avg (Second) Deriv"] = savgol_filter(
            data["Moving Avg (Second)"],
            window_length=win_length,
            polyorder=2,
            deriv=1,
        )
        data["Moving Avg (Second) Deriv2"] = savgol_filter(
            data["Moving Avg (Second)"],
            window_length=win_length,
            polyorder=2,
            deriv=2,
        )

        curr_row = data.iloc[-1, :]

        return data, curr_row

    def mean_reversion_crypto_algo(
        self, curr_row, curr_order, deriv_cutoff, investment, sec_der_strat=False
    ):
        """_summary_"""
        if curr_order["Just Sold"]:
            curr_order["Just Sold"] = False

        # In case script is interuppted or data is lost check to see if
        # the last order was a buy or sell to determine if holding a current
        # position or not
        if sec_der_strat:
            if (
                (curr_row["Moving Avg (First)"] > curr_row["Moving Avg (Second)"])
                and not curr_order["Has Position"]
                and (curr_row["Moving Avg (First) Deriv"] > 0)
                and (curr_row["Moving Avg (First) Deriv2"] > 0)
            ):
                curr_order["Has Position"] = True

                curr_order["Buy Datetime"] = str(curr_row["timestamp"])
                curr_order["Buy Price"] = curr_row["close"]
                curr_order["Deriv at Purchase"] = curr_row["Moving Avg (First) Deriv"]
        else:
            if (
                (curr_row["Moving Avg (First)"] > curr_row["Moving Avg (Second)"])
                and not curr_order["Has Position"]
                and (curr_row["Moving Avg (First) Deriv"] > 0)
            ):
                curr_order["Has Position"] = True

                curr_order["Buy Datetime"] = str(curr_row["timestamp"])
                curr_order["Buy Price"] = curr_row["close"]
                curr_order["Deriv at Purchase"] = curr_row["Moving Avg (First) Deriv"]

        if curr_order["Has Position"]:
            if (
                curr_row["Moving Avg (First) Deriv"]
                < curr_order["Deriv at Purchase"] * deriv_cutoff
            ):
                # Sell the current position

                curr_order["Sell Datetime"] = str(curr_row["timestamp"])
                curr_order["Sell Price"] = (
                    curr_row["close"] * 0.999
                )  # Account for a 0.10% slippage of close price (better simulates trading sell price)
                curr_order["Percent Change"] = (
                    100
                    * (curr_order["Sell Price"] - curr_order["Buy Price"])
                    / curr_order["Buy Price"]
                )
                curr_order["Order Profit"] = (
                    curr_order["Percent Change"] * investment / 100
                )
                curr_order["Deriv at Sell"] = curr_row["Moving Avg (First) Deriv"]
                curr_order["Just Sold"] = True

                investment += curr_order["Order Profit"]
                curr_order["Has Position"] = False

                if curr_order["Order Profit"] < 0:
                    curr_order["Total Losses"] += curr_order["Order Profit"]
                else:
                    curr_order["Total Gains"] += curr_order["Order Profit"]

        return curr_order, investment

    @ray.remote  # Assigns a fraction of the GPU per task
    def optimization_loop(self, data, res, fma, sma, der, win):
        try:
            orders_dict = {}
            curr_order = reset_order()
            initial_investment = 10_000
            profit_factor = 0
            time_steps_ = float(
                np.floor((end_time - start_time) / timedelta(minutes=res))
            )

            for step in range(int(time_steps_)):
                curr_data_ = data.iloc[: -int(time_steps_) + step, :].copy()

                _, curr_row_ = calculate_columns(
                    data=curr_data_,
                    resolution=res,
                    first_mov_avg=fma,
                    second_mov_avg=sma,
                    win_length=win,
                    deriv=1,
                )

                curr_order, initial_investment = mean_reversion_crypto_algo(
                    curr_row=curr_row_,
                    curr_order=curr_order,
                    deriv_cutoff=der,
                    investment=initial_investment,
                )

                if curr_order["Just Sold"]:
                    orders_dict[step] = curr_order.copy()
                    if curr_order["Total Losses"] < 0:
                        profit_factor = np.abs(curr_order["Total Gains"]) / np.abs(
                            curr_order["Total Losses"]
                        )

            return {
                "Resolution": res,
                "First Moving Average Day": fma,
                "Second Moving Average Day": sma,
                "Savgol Window Length": win,
                "Derivative Cutoff": der,
                "Total Gains": curr_order["Total Gains"],
                "Total Losses": curr_order["Total Losses"],
                "Profit Factor": profit_factor,
            }

        except Exception as e:
            return {"error": str(e)}

    # Function to retrieve the crypto data (should be outside multiprocessing)
    def get_data_for_resolution(self, res):
        return get_crypto_data(
            resolution=res,
            start_time=initial_time,
            end_time=end_time,
            symbol=TICKER_SYMBOL,
        )

    def run_optimal_solution(self, params: dict, investment: float = 10_000):
        optimized_orders_dict = {}
        datasets = {}
        data = self.get_crypto_data(
            resolution=params["Resolution"],
            start_time=initial_time,
            end_time=end_time,
            symbol=TICKER_SYMBOL,
        )
        curr_order = self.reset_order()
        time_steps_ = float(
            np.floor((end_time - start_time) / timedelta(minutes=params["Resolution"]))
        )
        for step in range(int(time_steps_)):
            curr_data = data.copy().iloc[: -int(time_steps_) + step, :]

            _, curr_row = self.calculate_columns(
                data=curr_data,
                resolution=params["Resolution"],
                first_mov_avg=params["First Moving Average Day"],
                second_mov_avg=params["Second Moving Average Day"],
                win_length=params["Savgol Window Length"],
                deriv=1,
            )

            curr_order, investment = self.mean_reversion_crypto_algo(
                curr_row=curr_row,
                curr_order=curr_order,
                deriv_cutoff=params["Derivative Cutoff"],
                investment=investment,
                sec_der_strat=False,
            )

            if curr_order["Just Sold"]:
                optimized_orders_dict[step] = curr_order.copy()
                datasets[step] = curr_data.copy()

        true_dataset = pd.concat(datasets, ignore_index=True)

        return data, true_dataset, optimized_orders_dict

    def graph_optimal_solution(
        self,
        normal_data: pd.DataFrame,
        true_dataset: pd.DataFrame,
        optimized_orders: dict,
        save_path: Path,
    ):
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 20))
        data, _ = self.calculate_columns(
            data=normal_data,
            resolution=best_params["Resolution"],
            first_mov_avg=best_params["First Moving Average Day"],
            second_mov_avg=best_params["Second Moving Average Day"],
            win_length=best_params["Savgol Window Length"],
            deriv=1,
        )
        sum_profit = 0
        successful_trades = 0
        best_trade = -np.inf

        failed_trades = 0
        worst_trade = np.inf

        ax[0].plot(
            data["timestamp"],
            data["close"],
            c="darkgray",
            label=f"Open Price ({best_params['Resolution']} minute)",
        )

        ax[0].plot(
            data["timestamp"],
            data["Moving Avg (First)"],
            c="dodgerblue",
            linestyle="--",
            label=f"Moving Average ({best_params['First Moving Average Day']:0.2f} days)",
        )

        ax[0].plot(
            data["timestamp"],
            data["Moving Avg (Second)"],
            c="chocolate",
            linestyle="--",
            label=f"Moving Average ({best_params['Second Moving Average Day']:0.2f} days)",
        )

        ax[0].grid()
        ax[0].legend()
        ax[0].set_xlim(start_time, end_time)
        ax[0].set_xlabel("Datetime")
        ax[0].set_ylabel(f"{TICKER_SYMBOL}")

        axis_interval = int(TIME_LENGTH / 20) if TIME_LENGTH / 20 > 1 else 1
        ax[0].xaxis.set_major_locator(mdates.DayLocator(interval=axis_interval))
        ax[0].tick_params(axis="x", labelrotation=60)

        ax[1].plot(
            true_dataset["timestamp"],
            true_dataset["Moving Avg (First) Deriv"],
            c="darkgray",
            label=f"Moving Average ({best_params['First Moving Average Day']:0.2f} days)",
        )
        ax[1].axhline(0, linestyle="--", c="k")

        ax[1].grid()
        ax[1].set_xlim(start_time, end_time)
        ax[1].set_xlabel("Datetime")
        ax[1].set_ylabel("First Derivative")
        ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=axis_interval))
        ax[1].tick_params(axis="x", labelrotation=60)

        ax[2].plot(
            true_dataset["timestamp"],
            true_dataset["Moving Avg (First) Deriv2"],
            c="darkgray",
            label=f"Moving Average ({best_params['First Moving Average Day']:0.2f} days)",
        )
        ax[2].axhline(0, linestyle="--", c="k")

        ax[2].grid()
        ax[2].set_xlim(start_time, end_time)
        ax[2].set_xlabel("Datetime")
        ax[2].set_ylabel("Second Derivative")
        ax[2].xaxis.set_major_locator(mdates.DayLocator(interval=axis_interval))
        ax[2].tick_params(axis="x", labelrotation=60)

        for _, order in optimized_orders.items():
            try:
                order["Buy Datetime"] = pd.to_datetime(order["Buy Datetime"])
                order["Sell Datetime"] = pd.to_datetime(order["Sell Datetime"])
                if order["Order Profit"] >= 0:
                    successful_trades += 1
                    sum_profit += order["Order Profit"]
                    if best_trade < order["Order Profit"]:
                        best_trade = order["Order Profit"]

                    ax[0].axvline(order["Buy Datetime"], linestyle="--", c="green")
                    ax[1].axvline(order["Buy Datetime"], linestyle="--", c="green")
                    ax[2].axvline(order["Buy Datetime"], linestyle="--", c="green")

                    ax[0].axvline(order["Sell Datetime"], linestyle="--", c="green")
                    ax[1].axvline(order["Sell Datetime"], linestyle="--", c="green")
                    ax[2].axvline(order["Sell Datetime"], linestyle="--", c="green")
                    ax[0].axvspan(
                        order["Buy Datetime"],
                        order["Sell Datetime"],
                        linestyle="--",
                        facecolor="palegreen",
                        alpha=0.3,
                    )
                    ax[1].axvspan(
                        order["Buy Datetime"],
                        order["Sell Datetime"],
                        linestyle="--",
                        facecolor="palegreen",
                        alpha=0.3,
                    )
                    ax[2].axvspan(
                        order["Buy Datetime"],
                        order["Sell Datetime"],
                        linestyle="--",
                        facecolor="palegreen",
                        alpha=0.3,
                    )
                else:
                    failed_trades += 1
                    sum_profit += order["Order Profit"]
                    if worst_trade > order["Order Profit"]:
                        worst_trade = order["Order Profit"]

                    ax[0].axvline(order["Buy Datetime"], linestyle="--", c="red")
                    ax[1].axvline(order["Buy Datetime"], linestyle="--", c="red")
                    ax[2].axvline(order["Buy Datetime"], linestyle="--", c="red")

                    ax[0].axvline(order["Sell Datetime"], linestyle="--", c="red")
                    ax[1].axvline(order["Sell Datetime"], linestyle="--", c="red")
                    ax[2].axvline(order["Sell Datetime"], linestyle="--", c="red")

                    ax[0].axvspan(
                        order["Buy Datetime"],
                        order["Sell Datetime"],
                        linestyle="--",
                        facecolor="lightcoral",
                        alpha=0.3,
                    )
                    ax[1].axvspan(
                        order["Buy Datetime"],
                        order["Sell Datetime"],
                        linestyle="--",
                        facecolor="lightcoral",
                        alpha=0.3,
                    )
                    ax[2].axvspan(
                        order["Buy Datetime"],
                        order["Sell Datetime"],
                        linestyle="--",
                        facecolor="lightcoral",
                        alpha=0.3,
                    )
            except KeyError:
                ax[0].axvline(order["Buy Datetime"], linestyle="--", c="green")
                ax[1].axvline(order["Buy Datetime"], linestyle="--", c="green")
                ax[2].axvline(order["Buy Datetime"], linestyle="--", c="green")

        # Set x-axis to display hour ticks
        hours = mdates.DayLocator(interval=2)  # Ticks every hour
        h_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M")  # Format as HH:MM

        title = (
            f"{TICKER_SYMBOL} | {successful_trades} Profitable Trades (Best: ${best_trade:,.2f}), "
            f"{failed_trades} Unprofitable Trades (Worst: ${worst_trade:,.2f}) | "
            f"Order Profit: ${sum_profit:,.2f} over {TIME_LENGTH - GRAPH_LENGTH} days"
        )

        title_2 = (
            f"Resolution: {best_params['Resolution']}, First Moving Average: {best_params['First Moving Average Day']}, Second Moving Average: {best_params['Second Moving Average Day']}, "
            f"Derivative Cutoff: {best_params['Derivative Cutoff']}, Savgol Window Length: {best_params['Savgol Window Length']}"
        )

        {
            "Resolution": 10,
            "First Moving Average Day": 0.2,
            "Second Moving Average Day": 5,
            "Derivative Cutoff": 0.25,
            "Savgol Window Length": 5,
            "Derivative": 1,
            "Profit": np.float64(565.094483855095),
        }

        ax[0].set_title(title)
        ax[1].set_title(title_2)

        ax[0].xaxis.set_major_locator(hours)
        ax[0].xaxis.set_major_formatter(h_fmt)

        ax[1].xaxis.set_major_locator(hours)
        ax[1].xaxis.set_major_formatter(h_fmt)

        ax[2].xaxis.set_major_locator(hours)
        ax[2].xaxis.set_major_formatter(h_fmt)

        fig.tight_layout()

        fig.savefig(
            Path(
                save_path,
                f"{datetime.now().strftime('%Y-%m-%d_%H%M')}-{TIME_LENGTH - GRAPH_LENGTH}_day_optimization",
            )
        )

    def export_optimal_data(self, best_params, optimized_orders):
        with open(
            Path(
                json_path,
                f"{datetime.now().strftime('%Y-%m-%d_%H%M')}-{TIME_LENGTH - GRAPH_LENGTH}_day_optimization_parameters",
            ),
            "w",
        ) as file:
            json.dump(best_params, file, indent=4)
            file.close()

        try:
            for key, value in optimized_orders.items():
                value["Buy Datetime"] = value["Buy Datetime"].strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                value["Sell Datetime"] = value["Sell Datetime"].strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        except Exception:
            pass

        with open(
            Path(
                json_path,
                f"{datetime.now().strftime('%Y-%m-%d_%H%M')}-{TIME_LENGTH - GRAPH_LENGTH}_day_optimization_orders",
            ),
            "w",
        ) as file:
            json.dump(optimized_orders, file, indent=4)
            file.close()

    def main_ray(self):
        results = []
        futures = []

        # Store large dataset in Ray object store
        data_objects = {
            res: ray.put(
                self.get_crypto_data(res, initial_time, end_time, TICKER_SYMBOL)
            )
            for res in resolution_range
        }

        for res in resolution_range:
            data_id = data_objects[res]  # Use stored object

            # Generate parameter combinations
            combinations = list(
                itertools.product(
                    [data_id],
                    [res],
                    first_mov_avg_range,
                    second_mov_avg_range,
                    deriv_cutoff_range,
                    win_length_range,
                )
            )

            # Submit tasks in bulk
            futures.extend(
                [self.optimization_loop.remote(*combo) for combo in combinations]
            )

        # Display progress bar
        with tqdm(total=len(futures), desc="Processing", unit="task") as pbar:
            while futures:
                ready_futures, futures = ray.wait(
                    futures, num_returns=min(10, len(futures)), timeout=1
                )  # Fetch multiple results
                for done_future in ready_futures:
                    results.append(ray.get(done_future))  # Retrieve result
                    pbar.update(1)

        return results

    def main_nested(self):
        results = []
        total_tasks = (
            len(first_mov_avg_range)
            * len(second_mov_avg_range)
            * len(deriv_cutoff_range)
            * len(win_length_range)
            * len(resolution_range)
        )

        with tqdm(total=total_tasks, desc="Processing", unit="task") as pbar:
            for res in resolution_range:
                data = self.get_crypto_data(
                    resolution=res,
                    start_time=initial_time,
                    end_time=end_time,
                    symbol=TICKER_SYMBOL,
                )

                for fma in first_mov_avg_range:
                    for sma in second_mov_avg_range:
                        for der in deriv_cutoff_range:
                            for win in win_length_range:
                                result = self.optimization_loop(
                                    data, res, fma, sma, der, win
                                )
                                results.append(result)
                                pbar.update(1)  # Update progress bar
        return results


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    resolution_range = [5, 10, 30]
    first_mov_avg_range = [0.1, 0.3, 0.5]
    second_mov_avg_range = [5, 6, 7, 9]
    win_length_range = [5, 11, 13]
    deriv_cutoff_range = [0.2, 0.4, 0.6]

    INVESTMENT = 10_000  # dollars
    initial_investment = INVESTMENT
    TICKER_SYMBOL = "BTC/USD"
    DATA_FOLDER = "mean_reversion_algo_opt-2"
    TIME_LENGTH = 365  # days
    GRAPH_LENGTH = 5  # days

    algo_optimization = AlgoOptimization(
        initial_investment=INVESTMENT,
        time_length=TIME_LENGTH,
        ticker_symbol=TICKER_SYMBOL,
        graph_length=GRAPH_LENGTH,
        data_folder=DATA_FOLDER,
        opt_ranges=[
            resolution_range,
            first_mov_avg_range,
            second_mov_avg_range,
            win_length_range,
            deriv_cutoff_range,
        ],
    )

    fig_path = Path(Path.cwd(), f"data/optimization/{DATA_FOLDER}/figures")
    fig_path.mkdir(parents=True, exist_ok=True)
    json_path = Path(Path.cwd(), f"data/optimization/{DATA_FOLDER}/json")
    json_path.mkdir(parents=True, exist_ok=True)

    ray.shutdown()
    ray.init(num_gpus=1, num_cpus=8)  # Adjust based on your system
    results = main_ray()
    ray.shutdown()

    best_params = {}
    best_pf = 0
    try:
        for res in results:
            if res["Profit Factor"] > best_pf:
                best_pf = res["Profit Factor"]
                best_params = res.copy()
    except KeyError:
        pass

    export_optimal_data(best_params, results)

    # with open(
    #     Path(json_path, "2025-03-29_1013-85_day_optimization_parameters"), "r"
    # ) as file:
    #     best_params = json.load(file)

    data, true_data, optimized_orders = run_optimal_solution(best_params)
    graph_optimal_solution(
        normal_data=data,
        true_dataset=true_data,
        optimized_orders=optimized_orders,
        save_path=fig_path,
    )


# Change the moving average to exponential weighted moving average to weight the
# average price based on the current price. Should help with large changes in price
# so that i can get in an out of trades faster.
