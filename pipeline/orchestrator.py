"""
管道编排器模块。
串联扫描、预处理、提取、向量化四个模块，实现一键全自动精炼管道。

用户只需提供：
1. Obsidian vault 目录路径
2. PDF 简历文件路径

系统自动完成：扫描 → 过滤 → 提取 → 向量化 → 生成报告
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

import fitz  # PyMuPDF

from config.settings import STORAGE_DIR, CACHE_DIR
from pipeline.scanner import ObsidianScanner, FileManifest
from pipeline.preprocessor import Preprocessor, CleanedChunk
from pipeline.extractor import Extractor, ExperienceEntry
from pipeline.vectorizer import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    """管道执行报告。"""
    status: str = "pending"          # pending / running / completed / failed
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0

    # 扫描阶段
    scanned_files: int = 0
    total_chars_scanned: int = 0

    # 预处理阶段
    cleaned_chunks: int = 0
    total_chars_before_clean: int = 0
    total_chars_after_clean: int = 0
    noise_ratio: float = 0.0

    # 提取阶段
    identified_chunks: int = 0
    extracted_entries: int = 0
    api_calls: int = 0
    tokens_used: int = 0

    # 向量化阶段
    vectorized_entries: int = 0

    # 错误信息
    error_message: str = ""

    def to_dict(self) -> dict:
        """转为字典。"""
        return asdict(self)


class PipelineOrchestrator:
    """
    全自动精炼管道编排器。

    完整管道流程：
    1. 解析 PDF 简历 → 获取简历文本内容
    2. 扫描 Obsidian 目录 → 文件清单
    3. 预处理（噪音过滤）→ 清洁文本片段
    4. LLM 提取 → 结构化经历条目
    5. 向量化存入 ChromaDB
    6. 输出处理报告
    """

    def __init__(self):
        """初始化所有子模块。"""
        self._scanner = ObsidianScanner()
        self._preprocessor = Preprocessor()
        self._vector_store = VectorStore()
        self._report = PipelineReport()
        self._report_file = STORAGE_DIR / "pipeline_report.json"

    def run_full_pipeline(
        self,
        obsidian_path: str,
        resume_pdf_path: str,
        incremental: bool = True,
    ) -> PipelineReport:
        """
        执行完整的精炼管道。

        Args:
            obsidian_path: Obsidian vault 的根目录路径
            resume_pdf_path: PDF 简历文件路径
            incremental: 是否启用增量扫描

        Returns:
            PipelineReport 管道执行报告
        """
        self._report = PipelineReport(status="running")
        self._report.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        start_ts = time.time()

        try:
            # Step 0: 解析 PDF 简历
            logger.info("=" * 60)
            logger.info("Step 0: 解析 PDF 简历")
            resume_content = self._parse_resume_pdf(resume_pdf_path)
            logger.info("简历解析完成: %d 字符", len(resume_content))

            # Step 1: 扫描 Obsidian 目录
            logger.info("=" * 60)
            logger.info("Step 1: 扫描 Obsidian 目录: %s", obsidian_path)
            manifests = self._scanner.scan(obsidian_path, incremental=incremental)
            scan_stats = self._scanner.get_scan_stats(manifests)
            self._report.scanned_files = scan_stats["total_files"]
            self._report.total_chars_scanned = scan_stats["total_chars"]
            logger.info("扫描结果: %d 个文件, %d 字符", scan_stats["total_files"], scan_stats["total_chars"])

            if not manifests:
                logger.info("没有需要处理的新文件")
                self._report.status = "completed"
                self._report.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
                self._report.duration_seconds = time.time() - start_ts
                self._save_report()
                return self._report

            # Step 2: 预处理（噪音过滤）
            logger.info("=" * 60)
            logger.info("Step 2: 预处理（噪音过滤）")
            chunks = self._preprocessor.process(manifests)
            prep_stats = self._preprocessor.get_preprocessing_report(chunks)
            self._report.cleaned_chunks = prep_stats["total_chunks"]
            self._report.total_chars_before_clean = prep_stats["total_original_chars"]
            self._report.total_chars_after_clean = prep_stats["total_cleaned_chars"]
            self._report.noise_ratio = prep_stats.get("overall_noise_ratio", 0.0)
            logger.info(
                "预处理结果: %d 个片段, 噪音比例 %.1f%%",
                prep_stats["total_chunks"],
                prep_stats.get("overall_noise_ratio", 0) * 100,
            )

            if not chunks:
                logger.info("预处理后没有有效片段")
                self._report.status = "completed"
                self._report.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
                self._report.duration_seconds = time.time() - start_ts
                self._save_report()
                return self._report

            # Step 3: LLM 提取
            logger.info("=" * 60)
            logger.info("Step 3: LLM 精炼提取")
            extractor = Extractor(resume_content=resume_content)
            entries = extractor.extract_all(chunks)
            ext_stats = extractor.get_extraction_stats()
            self._report.extracted_entries = len(entries)
            self._report.api_calls = ext_stats["total_api_calls"]
            self._report.tokens_used = ext_stats["total_tokens_used"]
            logger.info(
                "提取结果: %d 条经历条目, %d 次 API 调用, %d tokens",
                len(entries), ext_stats["total_api_calls"], ext_stats["total_tokens_used"],
            )

            # Step 4: 向量化存入 ChromaDB
            logger.info("=" * 60)
            logger.info("Step 4: 向量化存入 ChromaDB")
            added = self._vector_store.add_entries(entries)
            self._report.vectorized_entries = added
            logger.info("向量化结果: 成功存入 %d 条", added)

            # Step 5: 标记已处理文件
            for manifest in manifests:
                self._scanner.mark_processed(manifest.content_hash, manifest.relative_path)

            # 导出经历条目到 JSON 文件（便于检查）
            self._export_entries(entries)

            # 完成
            self._report.status = "completed"
            self._report.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self._report.duration_seconds = time.time() - start_ts
            self._save_report()

            logger.info("=" * 60)
            logger.info("管道执行完成! 耗时 %.1f 秒", self._report.duration_seconds)
            logger.info("=" * 60)

            return self._report

        except Exception as e:
            self._report.status = "failed"
            self._report.error_message = str(e)
            self._report.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self._report.duration_seconds = time.time() - start_ts
            self._save_report()
            logger.error("管道执行失败: %s", e, exc_info=True)
            raise

    def run_incremental(self, obsidian_path: str, resume_pdf_path: str) -> PipelineReport:
        """
        增量更新：只处理新增/修改的文件。

        Args:
            obsidian_path: Obsidian vault 的根目录路径
            resume_pdf_path: PDF 简历文件路径

        Returns:
            PipelineReport 管道执行报告
        """
        return self.run_full_pipeline(obsidian_path, resume_pdf_path, incremental=True)

    def run_full_rebuild(self, obsidian_path: str, resume_pdf_path: str) -> PipelineReport:
        """
        全量重建：清空缓存和向量库，重新处理所有文件。

        Args:
            obsidian_path: Obsidian vault 的根目录路径
            resume_pdf_path: PDF 简历文件路径

        Returns:
            PipelineReport 管道执行报告
        """
        logger.info("执行全量重建：清空缓存和向量库")
        self._scanner.clear_cache()
        self._vector_store.delete_all()
        return self.run_full_pipeline(obsidian_path, resume_pdf_path, incremental=False)

    def get_vector_store(self) -> VectorStore:
        """获取向量存储实例（供 matcher 模块使用）。"""
        return self._vector_store

    def get_last_report(self) -> Optional[PipelineReport]:
        """
        获取最近一次管道执行报告。

        Returns:
            PipelineReport 或 None
        """
        if self._report_file.exists():
            try:
                with open(self._report_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                report = PipelineReport()
                for k, v in data.items():
                    if hasattr(report, k):
                        setattr(report, k, v)
                return report
            except Exception as e:
                logger.warning("读取管道报告失败: %s", e)
        return None

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _parse_resume_pdf(self, pdf_path: str) -> str:
        """
        解析 PDF 简历，提取纯文本内容。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            简历文本内容

        Raises:
            FileNotFoundError: 文件不存在
            RuntimeError: PDF 解析失败
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"简历 PDF 文件不存在: {pdf_path}")

        try:
            doc = fitz.open(str(path))
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            content = "\n".join(text_parts).strip()
            if not content:
                raise RuntimeError(f"PDF 文件内容为空: {pdf_path}")
            return content
        except Exception as e:
            raise RuntimeError(f"解析 PDF 失败: {e}") from e

    def _export_entries(self, entries: List[ExperienceEntry]) -> None:
        """
        将提取的经历条目导出为 JSON 文件（便于人工审查）。

        Args:
            entries: 经历条目列表
        """
        export_path = STORAGE_DIR / "extracted_entries.json"
        try:
            data = [e.to_dict() for e in entries]
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("经历条目已导出到: %s", export_path)
        except IOError as e:
            logger.error("导出经历条目失败: %s", e)

    def _save_report(self) -> None:
        """保存管道执行报告到磁盘。"""
        try:
            with open(self._report_file, "w", encoding="utf-8") as f:
                json.dump(self._report.to_dict(), f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error("保存管道报告失败: %s", e)
