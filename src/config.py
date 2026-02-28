"""Centralized configuration management.

All configuration is loaded from environment variables with sensible defaults.
"""

import os
import sys
import random
import logging
import threading
from itertools import cycle
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Project Root ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Logging Setup ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --- Database Configuration ---
def get_db_config():
    """Load database configuration from environment variables.

    Returns:
        dict or None: Database config dict, or None if required variables are missing.
    """
    required_keys = ["DAILY_DATABASE", "DAILY_USER", "DAILY_PASSWORD", "DAILY_HOST"]
    missing = [k for k in required_keys if k not in os.environ]
    if missing:
        logger.error("缺少数据库环境变量: %s", ", ".join(missing))
        return None

    return {
        "dbname": os.environ["DAILY_DATABASE"],
        "user": os.environ["DAILY_USER"],
        "password": os.environ["DAILY_PASSWORD"],
        "host": os.environ["DAILY_HOST"],
        "port": os.environ.get("DAILY_PORT", "5432"),
    }


# --- Redis Configuration ---
def get_redis_client():
    """Create and return a Redis client, or None if unavailable.

    Returns:
        redis.Redis or None: Connected Redis client, or None on failure.
    """
    try:
        import redis

        host = os.environ.get("REDIS_HOST")
        if not host:
            logger.info("未配置Redis（REDIS_HOST为空），跳过Redis连接。")
            return None

        client = redis.Redis(
            host=host,
            port=int(os.environ.get("REDIS_PORT", 6380)),
            password=os.environ.get("REDIS_PASSWORD", ""),
            ssl=os.environ.get("REDIS_SSL", "true").lower() == "true",
            db=0,
            decode_responses=True,
        )
        client.ping()
        logger.info("成功连接到Redis。")
        return client
    except Exception as e:
        logger.warning("连接到Redis失败: %s", e)
        return None


# --- LLM Configuration ---
LLM_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

SCREENING_MODEL = os.environ.get("SCREENING_MODEL", "gemini-2.5-flash-lite")
ANALYSIS_MODEL = os.environ.get("ANALYSIS_MODEL", "gemini-2.5-pro")

# API key prefix for discovery from environment
API_KEY_PREFIX = os.environ.get("API_KEY_PREFIX", "Y")


class APIKeyPool:
    """Thread-safe pool of API keys loaded from environment variables."""

    def __init__(self, prefix=None):
        prefix = prefix or API_KEY_PREFIX
        key_names = [k for k in os.environ.keys() if k.startswith(prefix)]
        if not key_names:
            raise ValueError(
                f"未在环境变量中找到任何以 '{prefix}' 开头的API密钥。"
            )
        random.shuffle(key_names)
        self._pool = cycle(key_names)
        self._lock = threading.Lock()
        logger.info("成功加载 %d 个API密钥。", len(key_names))

    def get_next_key(self):
        """Get the next API key value (thread-safe).

        Returns:
            str: An API key value.
        """
        with self._lock:
            key_name = next(self._pool)
        return os.environ[key_name]


# --- Prompt File Paths ---
PROMPT_DIR = Path(os.environ.get("PROMPT_DIR", str(PROJECT_ROOT / "prompts")))
SCREENING_PROMPT_FILE = PROMPT_DIR / os.environ.get(
    "SCREENING_PROMPT_FILE", "screening.md"
)
ANALYSIS_PROMPT_FILE = PROMPT_DIR / os.environ.get(
    "ANALYSIS_PROMPT_FILE", "analysis.md"
)


def load_prompt(filepath):
    """Load a prompt template from a file.

    Args:
        filepath: Path to the prompt file.

    Returns:
        str: The prompt text content.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"提示词文件不存在: {path}")
    return path.read_text(encoding="utf-8")


# --- Notion Configuration ---
NOTION_API_KEY = os.environ.get("NOTION_API_KEY", "")
NOTION_PAGE_ID = os.environ.get("NOTION_PAGE_ID", "")

# --- Crawler Configuration ---
CRAWLER_BASE_URL = os.environ.get("CRAWLER_BASE_URL", "")
CRAWLER_MAX_PAGES = int(os.environ.get("CRAWLER_MAX_PAGES", "18"))
CRAWLER_CHUNK_SIZE = int(os.environ.get("CRAWLER_CHUNK_SIZE", "50"))
CRAWLER_MAX_WORKERS = int(os.environ.get("CRAWLER_MAX_WORKERS", "1"))
CRAWLER_RATE_LIMIT_SECONDS = float(os.environ.get("CRAWLER_RATE_LIMIT_SECONDS", "4.2"))

# --- HTTP Headers ---
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}
