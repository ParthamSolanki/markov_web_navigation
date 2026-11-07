from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT / "data"
SEQ_FILE = DATA_DIR / "msnbc990928.seq"

TABLES = ROOT / ("tables")
FIGURES = ROOT / ("figures")

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/133/msnbc+com+anonymous+web+data.zip"
)

K_VALUES = [1, 2, 3, 4, 5, 6]
TAU_VALUES = [0, 5, 10]
NUM_STATES = 17
ALPHA = 1.0

CATEGORIES = [
    "frontpage",
    "news",
    "tech",
    "local",
    "opinion",
    "on-air",
    "misc",
    "weather",
    "health",
    "living",
    "business",
    "sports",
    "summary",
    "bbs",
    "travel",
    "msn-news",
    "msn-sports",
]
