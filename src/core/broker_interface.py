# Package Imports
from config import settings

# Alpaca Trading Imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.requests import CryptoLatestQuoteRequest
from alpaca.data.historical import CryptoHistoricalDataClient

class BrokerInterface:

    def __init__(self, paper=True):
        self.trading_client = TradingClient(settings.API_KEY, settings.API_SECRET, paper=paper)
        self.buying_power = 0
        self.cash = 0
    
    def get_account_details(self):

        # Get our account information.
        account = self.trading_client.get_account()

        # Check if our account is restricted from trading.
        if account.trading_blocked:
            raise ValueError("Account is currently restricted from trading.")

        self.buying_power = float(account.buying_power)
        self.cash = float(account.cash)

    def execute_trade(self, cash_amount: float):
        """_summary_

        Args:
            quantity (float): amount of asset to purchase

        Returns:
            _type_: returns the submitted order to alpaca
        """

        # preparing market order
        market_order_request = MarketOrderRequest(
            symbol=settings.SYMBOL,
            notional=cash_amount,
            side=OrderSide.BUY,
            type="market",
            time_in_force=TimeInForce.GTC,
        )

        # Market order
        return self.trading_client.submit_order(order_data=market_order_request)
    
    def sell_position(self):
        """_summary_

        Args:
            quantity (float): amount of asset to purchase

        Returns:
            _type_: returns the submitted sell order to alpaca
        """

        position = self.trading_client.get_open_position(settings.SYMBOL.replace("/", ""))
        quantity = position.qty

        # preparing market order
        sell_order_request = MarketOrderRequest(
            symbol=settings.SYMBOL,
            qty=quantity,
            side=OrderSide.SELL,
            type="market",
            time_in_force=TimeInForce.GTC,
        )

        # Market order
        return self.trading_client.submit_order(order_data=sell_order_request)
    
    def get_latest_quote(self):
        client = CryptoHistoricalDataClient()
        quote_request = CryptoLatestQuoteRequest(symbol_or_symbols=[self.symbol])
        latest_quote = client.get_crypto_latest_quote(quote_request)

        return latest_quote[settings.SYMBOL].bid_price, latest_quote[settings.SYMBOL].ask_price

