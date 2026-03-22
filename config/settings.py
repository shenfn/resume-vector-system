"""
全局配置模块
集中管理所有配置项，包括 API 密钥、路径、模型参数等。
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")

# ============================================================
# 项目根目录
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ============================================================
# MiniMax API 配置
# ============================================================
MINIMAX_API_KEY: str = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL: str = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")
MINIMAX_MODEL: str = os.getenv("MINIMAX_MODEL", "MiniMax-Text-01")

# API 调用参数
API_MAX_RETRIES: int = 3
API_RETRY_DELAY_BASE: float = 2.0  # 指数退避基数（秒）
API_TIMEOUT: int = 120  # 单次请求超时（秒）
API_TEMPERATURE: float = 0.3  # 提取任务使用低温度以保证一致性

# ============================================================
# ChromaDB 配置
# ============================================================
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(PROJECT_ROOT / "storage" / "chroma_db")
)
CHROMA_COLLECTION_EXPERIENCES: str = "experience_entries"
CHROMA_COLLECTION_SKILLS: str = "skill_tags"

# ============================================================
# 文件扫描配置
# ============================================================
SUPPORTED_EXTENSIONS: tuple = (".md", ".txt", ".mdx")
SKIP_DIRS: tuple = (".obsidian", ".trash", ".git", "__pycache__", "node_modules")
MIN_CONTENT_LENGTH: int = 50  # 过短的文件直接跳过（字符数）

# ============================================================
# 预处理配置
# ============================================================
# 大文件阈值（字符数）
FILE_SIZE_SMALL: int = 5000
FILE_SIZE_MEDIUM: int = 30000
# 超过此阈值的文件需要按章节拆分
FILE_SIZE_LARGE: int = 30000

# 噪音过滤：最短有效片段长度
CHUNK_MIN_LENGTH: int = 80

# ============================================================
# LLM 提取配置
# ============================================================
# 单次 API 调用的最大输入字符数（留 buffer 给 prompt 和输出）
EXTRACTION_MAX_INPUT_CHARS: int = 12000
# 提取置信度阈值：低于此值的条目不入库
EXTRACTION_CONFIDENCE_THRESHOLD: float = 0.5

# ============================================================
# 匹配与检索配置
# ============================================================
# 语义检索返回的 Top-K 数量
RETRIEVAL_TOP_K: int = 8
# 关键词匹配权重 vs 语义匹配权重
KEYWORD_WEIGHT: float = 0.4
SEMANTIC_WEIGHT: float = 0.6

# ============================================================
# 存储路径
# ============================================================
STORAGE_DIR: Path = PROJECT_ROOT / "storage"
EXPORTS_DIR: Path = STORAGE_DIR / "exports"
CACHE_DIR: Path = STORAGE_DIR / "cache"

# 确保目录存在
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 日志配置
# ============================================================
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger("resume-vector-system")
