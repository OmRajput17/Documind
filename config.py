import os
from dotenv import load_dotenv
from pathlib import Path

### Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = "data/"
VECTOR_STORE_PATH = BASE_DIR/"vectorstore"

LOG_DIR = Path(__file__).resolve().parent / "logs"