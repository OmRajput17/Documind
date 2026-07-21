import os
from dotenv import load_dotenv
from pathlib import Path

# Suppress HuggingFace Hub symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

### Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = "data/"
VECTOR_STORE_PATH = BASE_DIR/"vectorstore"

LOG_DIR = Path(__file__).resolve().parent / "logs"


### Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 240
TEXT_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    " ",
    ""
]

## Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_PATH = BASE_DIR / "models"

