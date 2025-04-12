
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("APCA-API-KEY-ID")
API_SECRET = os.getenv("APCA-API-SECRET-KEY")

SYMBOL = "AAPL"
TIMEFRAME = "1Min"
MAX_POSITION_SIZE = 100
STOP_LOSS_PCT = 0.02

INVESTMENT = 10_000  # dollars
RESOLUTION = 60  # minute
TIME_LENGTH = 30  # days
FIRST_MOV_AVG_DAY = 0.5  # days
SECOND_MOV_AVG_DAY = 7  # days
DERIV_CUTOFF = 0.8
WIN_LENGTH = 9
DERIVATIVE = 1
TICKER_SYMBOL = "BTC/USD"
DATA_FOLDER = "live-paper-v3"