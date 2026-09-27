from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "SQ3L_DB.db"

conn = sqlite3.connect(str(DB_PATH),check_same_thread=False)
memory = SqliteSaver(conn)
memory.setup()