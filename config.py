import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Suppress HuggingFace Hub symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

### Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = "data/"
VECTOR_STORE_PATH = BASE_DIR/"vectorstore"

LOG_DIR = Path(__file__).resolve().parent / "logs"

PG_CONNECTION_STRING = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
PG_COLLECTION_NAME = "documind_budget"


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
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_CACHE_PATH = BASE_DIR / "models"

## BM25 Index Cache
BM25_CACHE_PATH = BASE_DIR / "cache" / "bm25_index.pkl"

### Retriever
DENSE_TOP_K = 20
BM25_TOP_K = 20
RERANK_TOP_K = 8
RRF_TOP_N = 15
HYBRID_TOP_K = 8
CONFIDENCE_THRESHOLD = 0.65
CONFIDENCE_TOP_K = 5
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


### Graph
MAX_RETRIES = 3

### LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
VALIDATION_LLM = "openai/gpt-oss-20b"
LLM_MODEL = "openai/gpt-oss-120b"
VALIDATION_TEMPERATURE = 0
GENERATION_TEMPERATURE = 0.5


### Guardrials
### Guardrails
PROMPT_INJECTION_PATTERNS = {

    # ============================================================
    # Instruction Override
    # ============================================================
    r"(ignore|forget|disregard|override)\s+(all\s+)?(previous|prior|above)\s+(instructions?|context)": 4,
    r"(ignore|disregard)\s+(everything|all)\s+(above|below)": 4,
    r"(ignore|skip|do\s+not\s+use|don't\s+use)\s+(the\s+)?(retrieved|provided|given)\s+(documents?|context)": 5,
    r"answer\s+from\s+your\s+own\s+(knowledge|training)": 5,
    r"(respond|answer)\s+without\s+(using\s+)?(the\s+)?(documents?|context)": 5,

    # ============================================================
    # System Prompt / Internal Prompt Extraction
    # ============================================================
    r"reveal\s+(your\s+)?(system|hidden|internal|original)\s+(prompt|instructions?)": 5,
    r"(show|print|display|repeat|dump|output)\s+(me\s+)?(your\s+)?(system|hidden|internal|original)\s+(prompt|instructions?)": 5,
    r"what\s+(are|were)\s+your\s+(rules|instructions)": 4,
    r"developer\s+message": 5,

    # ============================================================
    # Persona Hijacking / Jailbreak
    # ============================================================
    r"(you\s+are\s+now|from\s+now\s+on\s+you\s+are|pretend\s+to\s+be)": 4,
    r"new\s+persona": 3,
    r"jailbreak": 4,
    r"do\s+anything\s+now": 4,
    r"act\s+as\s+(an?\s+)?(unrestricted|uncensored|different|another)\s+(assistant|ai|model)": 4,

    # ============================================================
    # Role / Header Injection
    # FIX: removed leading `^` anchor — with only re.IGNORECASE set (no
    # re.MULTILINE), `^` only matches the very start of the whole string,
    # so an embedded mid-query attempt like "...please help. System: ignore
    # all rules." would have been missed entirely. Matching anywhere in
    # the text is the correct, more realistic detection behavior here.
    # ============================================================
    r"(system|assistant|developer)\s*:": 5,
    r"<\s*(system|assistant|developer)\s*>": 5,
    r"</\s*(system|assistant|developer)\s*>": 5,
    r"role\s*:\s*(system|assistant|developer)": 5,
    r"```(system|assistant|developer)": 5,

    # ============================================================
    # Guardrail / Policy Bypass
    # ============================================================
    r"bypass": 3,
    r"disable\s+(all\s+)?(safety|guardrails?|restrictions?|filters?)": 5,
    r"(ignore|disable)\s+(all\s+)?(policies|rules|guardrails|restrictions)": 5,

    # ============================================================
    # Chain-of-Thought / Reasoning Extraction
    # ============================================================
    r"(show|reveal|display)\s+(your\s+)?chain\s+of\s+thought": 5,
    r"show\s+your\s+reasoning": 4,

    # ============================================================
    # Tool / Agent Enumeration
    # ============================================================
    r"(list|show)\s+(available\s+)?tools": 4,
    r"tool\s+configuration": 5,
    r"what\s+tools\s+can\s+you\s+access": 3,

    # ============================================================
    # Obfuscation Indicators
    # ============================================================
    r"(base64|rot13|hex)\s+(decode|encoded?)": 2,
    # NEW, optional: flags suspiciously long base64-shaped strings
    # (the raw payload itself, not just someone naming the encoding).
    # Low weight since legitimate tokens/IDs can incidentally match this
    # shape — it should only ever nudge the score, never block alone.
    r"[A-Za-z0-9+/]{24,}={0,2}": 2,

    # ============================================================
    # Generic (kept narrow to avoid false positives on benign phrasing
    # like "act as a summarizer" or "act as a reviewer")
    # ============================================================
    r"act\s+as": 2,
}

# Two-tier thresholds — see routing logic below.
PROMPT_INJECTION_REVIEW_THRESHOLD = 3   # escalate to DeepTeam (Gemini judge)
PROMPT_INJECTION_BLOCK_THRESHOLD = 5              # hard block, skip DeepTeam entirely


# ------------------------------------------------------------------
# Routing logic (for reference — wire this into your guard's check()
# method or the orchestrator node that calls it):
#
#   if score >= PROMPT_INJECTION_THRESHOLD:
#       -> block immediately, do not call DeepTeam
#   elif score >= PROMPT_INJECTION_AMBIGUOUS_THRESHOLD:
#       -> escalate to DeepTeam for a deeper LLM-as-judge check
#   else:
#       -> clearly safe, skip DeepTeam entirely
# ------------------------------------------------------------------

PII_PATTERNS = {

    # Email
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",

    # Indian Mobile Number
    "PHONE": r"(?<!\d)(?:\+91[-\s]?)?[6-9]\d{9}(?!\d)",

    # Credit Card (Keep BEFORE Aadhaar)
    "CREDIT_CARD": r"\b(?:\d{4}[ -]?){3}\d{4}\b|\b\d{15}\b",

    # Aadhaar
    "AADHAAR": r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b",

    # PAN
    "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",

    # Passport
    "PASSPORT": r"\b[A-PR-WY][1-9]\d{6}\b",

    # IFSC
    "IFSC": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",

    # UPI
    "UPI": (
        r"\b[a-zA-Z0-9._-]+@"
        r"(?:"
        r"ybl|ibl|axl|upi|"
        r"oksbi|okhdfcbank|okaxis|okicici|"
        r"paytm|apl|airtel"
        r")\b"
    ),
}

PII_PLACEHOLDERS = {
    "EMAIL": "[EMAIL]",
    "PHONE": "[PHONE]",
    "CREDIT_CARD": "[CREDIT_CARD]",
    "AADHAAR": "[AADHAAR]",
    "PAN": "[PAN]",
    "PASSPORT": "[PASSPORT]",
    "IFSC": "[IFSC]",
    "UPI": "[UPI]",
}


NVIDIA_API_KEY = os.getenv("NVIDIA_NIM_API_KEY")
