"""
A module for handling the risk for each trade.

Created by Zach Hatzenbeller 2025-07-05
"""

# Repository Imports
from utils.metrics import max_drawdown


class RiskManager:
    def __init__(
        self,
        max_risk_per_trade=0.01,
        drawdown_limit=0.2,
        leverage=1.0,
        stop_loss_pct=0.02,
        target_price_pct=0.05,
    ):
        self.max_risk_per_trade = max_risk_per_trade  # e.g. 1% of capital
        self.drawdown_limit = drawdown_limit  # e.g. 20% max allowable drawdown
        self.stop_loss_pct = stop_loss_pct
        self.target_price_pct = target_price_pct
        self.leverage = leverage
        self.equity_curve = []  # track PnL over time for drawdown calcs

    def validate_trade(self, current_capital: float, num_allocations: int) -> bool:
        """
        Check if a trade can be taken (e.g., within limits).

        Calculate position size based on risk per trade and stop loss %.
        Basically I only want to risk losing 1% of my total capital. I can
        then calculate the total amount of money (max_position_size) that
        I can spend on this trade such that if I hit my stop loss, I have
        only lost 1% of the initial portfolio value.
        """
        allocation_amount = current_capital/num_allocations
        risk_amount = allocation_amount * self.max_risk_per_trade
        max_capital_risk = risk_amount / self.stop_loss_pct
        position_size = min(max_capital_risk, allocation_amount * self.leverage)
        return position_size <= (allocation_amount * self.leverage), position_size

    def check_drawdown_limit(
        self, equity_curve
    ) -> bool:
        """Return False if max drawdown exceeded."""
        max_dd, _ = max_drawdown(equity_curve)
        return max_dd <= self.drawdown_limit

    def enforce_stop_loss(self, current_price, stop_price) -> bool:
        """Determine if stop loss should trigger."""
        return current_price <= stop_price

    def enforce_take_profit(self, current_price, take_profit_price) -> bool:
        """Determine if take profit should trigger."""
        return current_price >= take_profit_price
