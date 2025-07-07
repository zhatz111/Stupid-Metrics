"""
Module for storing common portfolio metrics and indicators
Created by Zach Hatzenbeller 2025-07-02
"""

# Third Party Imports
import numpy as np


def sharpe_ratio(portfolio_return: float, risk_free_rate: float, std_portfolio: float):
    """_summary_

    Args:
        portfolio_return (float): _description_
        risk_free_rate (float): _description_
        std_portfolio (float): _description_

    Returns:
        _type_: _description_
    """
    return (portfolio_return - risk_free_rate) / std_portfolio


def sortino_ratio(
    portfolio_return: float, risk_free_rate: float, neg_std_portfolio: float
):
    """_summary_

    Args:
        portfolio_return (float): _description_
        risk_free_rate (float): _description_
        neg_std_portfolio (float): _description_

    Returns:
        _type_: _description_
    """
    return (portfolio_return - risk_free_rate) / neg_std_portfolio


def beta(returns_dict: dict, weights_dict: dict, market_reference_returns: list):
    """_summary_

    Args:
        returns_dict (dict): _description_
        weights_dict (dict): _description_
        market_reference_returns (list): _description_

    Returns:
        _type_: _description_
    """
    betas = {}
    for key, returns in returns_dict.items():
        cov_mat = np.cov(np.array(returns, market_reference_returns))
        cov_ = cov_mat[0, 1]
        var_ = np.nanstd(np.array(market_reference_returns)) ** 2
        betas[key] = cov_ / var_

    portfolio_beta = 0
    for key, beta in betas.items():
        portfolio_beta += beta * weights_dict[key]

    return portfolio_beta, betas


def alpha(
    portfolio_return: float,
    risk_free_rate: float,
    expected_market_return: float,
    portfolio_beta: float,
):
    """_summary_

    Args:
        portfolio_return (float): _description_
        risk_free_rate (float): _description_
        expected_market_return (float): _description_
        portfolio_beta (float): _description_

    Returns:
        _type_: _description_
    """
    return portfolio_return - (
        risk_free_rate + portfolio_beta * (expected_market_return - risk_free_rate)
    )


def max_drawdown(equity_curve: list):
    """_summary_

    Args:
        equity_curve (list): _description_

    Returns:
        _type_: _description_
    """
    peak = equity_curve[0]
    max_dd = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_dd = max(max_dd, drawdown)
    return max_dd


def profit_factor(gross_profits: float, gross_losses: float):
    """_summary_

    Args:
        gross_profits (float): _description_
        gross_losses (float): _description_

    Returns:
        _type_: _description_
    """
    return gross_profits/gross_losses
