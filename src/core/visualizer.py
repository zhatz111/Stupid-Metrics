"""
A module for visualizing the outputs of the backtests, paper trading, and live trading

Created by Zach Hatzenbeller 2025-07-10
"""

# Repository Imports
from trading.backtest import Backtester

# Standard Library Imports


# Third Party Imports
import matplotlib.pyplot as plt
import numpy as np


class TradingVisualizer:
    def __init__(self, backtester: Backtester):
        self.backtester = backtester

    def plot_equity_curve(self, ticker):
        equity = self.backtester.data_handler.equity_curve
        drawdown = self.backtester.data_handler.drawdown
        metrics = self.backtester.data_handler.compute_metrics()
        asset_equities = self.backtester.data_handler.get_asset_equities()
        normalized_equity = np.array(equity) - equity[0]

        buynhold_pnl = 0
        for _, data in self.backtester.data.groupby("symbol"):
            initial_price = data["close"].iloc[0]
            final_price = data["close"].iloc[-1]
            allocation = self.backtester.initial_capital/self.backtester.allocations
            qty = allocation/initial_price
            pnl = (qty*final_price) - allocation
            buynhold_pnl += pnl

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

        equity_diff = equity[-1] - equity[0]
        ax[0].plot(normalized_equity, color="black", label="Portfolio")
        ax[0].set_xlabel("Trade Number")
        ax[0].set_ylabel("Equity Value")
        ax[0].set_title(
            "Portfolio and Asset Equity Curves"
        )
        
        # f"Starting Equity: ${equity[0]:,.2f}, Final Equity: ${equity[-1]:,.2f}, PnL: ${equity_diff:,.2f}, Buy and Hold PnL: ${buynhold_pnl:,.2f}"

        for key, data in asset_equities.items():
            x, y = zip(*data)
            normalized_y = np.array(y) - y[0]
            ax[0].plot(x, normalized_y, label=f"{key}", alpha=0.5)
        
        ax[0].legend()

        x = np.arange(0, len(drawdown), 1)
        y = np.array(drawdown) * -100
        ax[1].plot(x, y, color="red")
        ax[1].fill_between(x, y, alpha=0.3, color="red")
        ax[1].set_xlabel("Trade Number")
        ax[1].set_ylabel("Drawdown %")
        ax[1].set_title(f"Max Drawdown: {max(drawdown) * 100:.2f}%")

        data_info = "Portfolio Metrics\n"
        align_key_width = 20
        align_val_width = 10

        # Add metrics (aligned floats)
        for key, value in metrics.items():
            if isinstance(value, float):
                data_info += f"{key:<{align_key_width}}{value:<{align_val_width}.3f}\n"
            elif isinstance(value, str):
                data_info += f"{key:<{align_key_width}}{value:<{align_val_width}}\n"


        # Add dollar-based values (aligned with commas and $)
        data_info += f"{'Starting Equity':<{align_key_width}}${equity[0]:<{align_val_width-1},.2f}\n"
        data_info += f"{'Final Equity':<{align_key_width}}${equity[-1]:<{align_val_width-1},.2f}\n"
        data_info += f"{'PnL':<{align_key_width}}${equity_diff:<{align_val_width-1},.2f}\n"
        data_info += f"{'Buy Hold PnL':<{align_key_width}}${buynhold_pnl:<{align_val_width-1},.2f}"



        # Define properties for the textbox
        box_properties = dict(
            boxstyle='round,pad=0.3',  # Rounded corners with padding
            facecolor='whitesmoke',    # Background color
            alpha=0.5,                 # Transparency
            edgecolor='black',         # Border color
            linewidth=1.0              # Border width
        )

        ax[0].text(
            0.01, 0.96, data_info,
            transform=ax[0].transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=box_properties,
            family='monospace'
        )

        plt.tight_layout()
        plt.show()
