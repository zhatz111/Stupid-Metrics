
import os
import pytz
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("APCA-API-KEY-ID")
API_SECRET = os.getenv("APCA-API-SECRET-KEY")

INVESTMENT = 10_000  # dollars
SYMBOL = "BTC/USD"
DATA_RESOLUTION = 60  # minute
MAX_POSITION_SIZE = 100
STOP_LOSS_PCT = 0.02

# Backtesting/Optimization Parameters
TIME_LENGTH_BACKTESTING = 30  # days
FIRST_MOV_AVG_RES = 0.5  # days
SECOND_MOV_AVG_RES = 7  # days
DERIV_CUTOFF = 0.8
WIN_LENGTH = 9
DERIVATIVE = 1

DATA_FOLDER = "live-paper-v3"

# Set timezone to Eastern Time
eastern = pytz.timezone('America/New_York')

# Data and figure file paths to save too
data_path_ = Path(Path.cwd(), "data/mean_reversion_algo/paper_trading", DATA_FOLDER)
data_path_.mkdir(parents=True, exist_ok=True)

fig_save_path = Path(data_path_, "figures")
fig_save_path.mkdir(parents=True, exist_ok=True)