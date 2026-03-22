"""
Obsidian 文件扫描器模块。
递归扫描指定目录下的所有文本文件，生成文件清单。
支持增量扫描：通过文件哈希值判断是否已处理过。

参考 resume-experience-extractor 的 extract_content.py 设计，
但增强了增量扫描、文件分级和元数据收集能力。
"""

import os
import hashlib
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime

from config.settings import (
    SUPPORTED_EXTENSIONS,
    SKIP_DIRS,
    MIN_CONTENT_LENGTH,
    FILE_SIZE_SMALL,
    FILE_SIZE_MEDIUM,
    CACHE_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class FileManifest:
    """文件清单条目，记录每个待处理文件的元数据。"""
    file_path: str          # 文件绝对路径
    relative_path: str      # 相对于 Obsidian 根目录的路径
    file_name: str          # 文件名（不含路径）
    size_bytes: int         # 文件大小（字节）
    char_count: int         # 字符数
    modified_time: str      # 最后修改时间（ISO 格式）
    content_hash: str       # 内容 SHA256 哈希值
    size_category: str      # 文件大小分级：small / medium / large
    content: str = ""       # 文件原始内容（扫描时读取）


class ObsidianScanner:
    """
    Obsidian 文件扫描器。
    
    功能：
    1. 递归扫描指定目录下所有支持格式的文件
    2. 按文件大小自动分级（small / medium / large）
    3. 支持增量扫描：记录已处理文件的哈希值，跳过未修改的文件
    4. 生成结构化的文件清单
    """

    def __init__(self):
        """初始化扫描器，加载已处理文件的缓存记录。"""
        self._cache_file = CACHE_DIR / "scan_cache.json"
        self._processed_hashes: dict = self._load_cache()

    def scan(self, obsidian_path: str, incremental: bool = True) -> List[FileManifest]:
        """
        扫描 Obsidian 目录，返回文件清单列表。

        Args:
            obsidian_path: Obsidian vault 的根目录路径
            incremental: 是否启用增量扫描（仅处理新增/修改的文件）

        Returns:
            FileManifest 列表，按文件大小从小到大排序

        Raises:
            FileNotFoundError: 目录不存在时抛出
        """
        root = Path(obsidian_path)
        if not root.exists():
            raise FileNotFoundError(f"Obsidian 目录不存在: {obsidian_path}")
        if not root.is_dir():
            raise NotADirectoryError(f"路径不是目录: {obsidian_path}")

        manifests: List[FileManifest] = []
        skipped_count = 0
        error_count = 0

        for dirpath, dirnames, filenames in os.walk(root):
            # 跳过隐藏目录和系统目录
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRS and not d.startswith(".")
            ]

            for fname in filenames:
                if not self._is_supported_file(fname):
                    continue

                fpath = Path(dirpath) / fname
                try:
                    manifest = self._build_manifest(fpath, root)
                except Exception as e:
                    logger.warning("读取文件失败 %s: %s", fpath, e)
                    error_count += 1
                    continue

                # 跳过过短的文件
                if manifest.char_count < MIN_CONTENT_LENGTH:
                    logger.debug("跳过过短文件: %s (%d 字符)", manifest.relative_path, manifest.char_count)
                    skipped_count += 1
                    continue

                # 增量扫描：跳过未修改的文件
                if incremental and self._is_already_processed(manifest.content_hash):
                    logger.debug("跳过未修改文件: %s", manifest.relative_path)
                    skipped_count += 1
                    continue

                manifests.append(manifest)

        # 按文件字符数从小到大排序（先处理小文件，快速出结果）
        manifests.sort(key=lambda m: m.char_count)

        logger.info(
            "扫描完成: 共发现 %d 个待处理文件, 跳过 %d 个, 失败 %d 个",
            len(manifests), skipped_count, error_count
        )

        return manifests

    def mark_processed(self, content_hash: str, source_file: str) -> None:
        """
        标记文件为已处理，写入缓存。

        Args:
            content_hash: 文件内容的 SHA256 哈希值
            source_file: 来源文件相对路径
        """
        self._processed_hashes[content_hash] = {
            "source_file": source_file,
            "processed_at": datetime.now().isoformat(),
        }
        self._save_cache()

    def clear_cache(self) -> None:
        """清除所有已处理文件的缓存记录。"""
        self._processed_hashes = {}
        self._save_cache()
        logger.info("扫描缓存已清除")

    def get_scan_stats(self, manifests: List[FileManifest]) -> dict:
        """
        生成扫描统计报告。

        Args:
            manifests: 文件清单列表

        Returns:
            统计信息字典
        """
        if not manifests:
            return {"total_files": 0, "total_chars": 0, "by_category": {}}

        total_chars = sum(m.char_count for m in manifests)
        by_category = {}
        for m in manifests:
            cat = m.size_category
            if cat not in by_category:
                by_category[cat] = {"count": 0, "total_chars": 0}
            by_category[cat]["count"] += 1
            by_category[cat]["total_chars"] += m.char_count

        return {
            "total_files": len(manifests),
            "total_chars": total_chars,
            "by_category": by_category,
        }

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _is_supported_file(self, filename: str) -> bool:
        """判断文件是否为支持的格式。"""
        return any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)

    def _build_manifest(self, fpath: Path, root: Path) -> FileManifest:
        """
        构建单个文件的清单条目。

        Args:
            fpath: 文件绝对路径
            root: Obsidian 根目录路径

        Returns:
            FileManifest 对象
        """
        # 尝试多种编码读取
        content = self._read_file_content(fpath)
        stat = fpath.stat()

        char_count = len(content)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # 文件大小分级
        if char_count <= FILE_SIZE_SMALL:
            size_category = "small"
        elif char_count <= FILE_SIZE_MEDIUM:
            size_category = "medium"
        else:
            size_category = "large"

        return FileManifest(
            file_path=str(fpath.resolve()),
            relative_path=str(fpath.relative_to(root)),
            file_name=fpath.name,
            size_bytes=stat.st_size,
            char_count=char_count,
            modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            content_hash=content_hash,
            size_category=size_category,
            content=content,
        )

    def _read_file_content(self, fpath: Path) -> str:
        """
        读取文件内容，自动尝试多种编码。

        Args:
            fpath: 文件路径

        Returns:
            文件文本内容

        Raises:
            UnicodeDecodeError: 所有编码都失败时抛出
        """
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
        for enc in encodings:
            try:
                with open(fpath, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise UnicodeDecodeError(
            "all", b"", 0, 1,
            f"无法用任何支持的编码读取文件: {fpath}"
        )

    def _is_already_processed(self, content_hash: str) -> bool:
        """判断文件是否已处理过。"""
        return content_hash in self._processed_hashes

    def _load_cache(self) -> dict:
        """从磁盘加载扫描缓存。"""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("加载扫描缓存失败，将使用空缓存: %s", e)
        return {}

    def _save_cache(self) -> None:
        """将扫描缓存写入磁盘。"""
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._processed_hashes, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error("保存扫描缓存失败: %s", e)
