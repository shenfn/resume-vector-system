"""
向量化与 ChromaDB 管理模块。
将提取的经历条目向量化并存入 ChromaDB，支持持久化存储和语义检索。

ChromaDB 集合设计：
- experience_entries：存储精炼后的项目经历条目（主集合）
- skill_tags：存储技能标签索引（辅助检索）
"""

import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings

from config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_EXPERIENCES,
    CHROMA_COLLECTION_SKILLS,
)
from pipeline.extractor import ExperienceEntry

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB 向量存储管理器。

    功能：
    1. 管理 ChromaDB 客户端和集合的生命周期
    2. 将 ExperienceEntry 向量化并存入数据库
    3. 提供语义检索接口
    4. 支持数据导出和统计
    """

    def __init__(self):
        """初始化 ChromaDB 客户端，创建或加载持久化集合。"""
        persist_dir = Path(CHROMA_PERSIST_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(persist_dir))

        # 主集合：经历条目
        self._experiences = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_EXPERIENCES,
            metadata={"description": "精炼后的项目经历条目，用于语义检索和简历匹配"},
        )

        # 辅助集合：技能标签
        self._skills = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_SKILLS,
            metadata={"description": "技能标签索引，用于关键词维度的辅助检索"},
        )

        logger.info(
            "ChromaDB 初始化完成: 经历条目 %d 条, 技能标签 %d 条",
            self._experiences.count(),
            self._skills.count(),
        )

    def add_entries(self, entries: List[ExperienceEntry]) -> int:
        """
        批量添加经历条目到向量数据库。

        Args:
            entries: ExperienceEntry 列表

        Returns:
            成功添加的条目数
        """
        if not entries:
            return 0

        added_count = 0
        for i, entry in enumerate(entries):
            try:
                entry_id = f"exp_{i}_{hash(entry.resume_bullet) & 0xFFFFFFFF:08x}"

                # 主向量文档：resume_bullet + solution（捕捉项目核心语义）
                main_document = f"{entry.resume_bullet}\n{entry.solution}"

                # 元数据：存储结构化信息，用于过滤和展示
                metadata = {
                    "project_name": entry.project_name or "",
                    "tech_stack": json.dumps(entry.tech_stack, ensure_ascii=False),
                    "role": entry.role or "",
                    "core_challenge": entry.core_challenge or "",
                    "outcome": entry.outcome or "",
                    "skill_tags": json.dumps(entry.skill_tags, ensure_ascii=False),
                    "business_value": entry.business_value or "",
                    "detail_level": entry.detail_level or "brief",
                    "source_file": entry.source_file or "",
                    "section_path": entry.section_path or "",
                    "confidence": entry.confidence,
                }

                self._experiences.upsert(
                    ids=[entry_id],
                    documents=[main_document],
                    metadatas=[metadata],
                )

                # 辅助集合：技能标签向量
                if entry.skill_tags:
                    skill_doc = " ".join(entry.skill_tags)
                    self._skills.upsert(
                        ids=[f"skill_{entry_id}"],
                        documents=[skill_doc],
                        metadatas={"experience_id": entry_id, "source_file": entry.source_file or ""},
                    )

                added_count += 1
            except Exception as e:
                logger.error("添加条目失败 [%s]: %s", entry.project_name, e)

        logger.info("向量化完成: 成功添加 %d/%d 条经历条目", added_count, len(entries))
        return added_count

    def search_by_text(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        通过文本进行语义检索。

        Args:
            query: 查询文本（通常是 JD 的关键描述）
            top_k: 返回的最大结果数

        Returns:
            检索结果列表，每个元素包含 id、document、metadata、distance
        """
        if self._experiences.count() == 0:
            logger.warning("经历条目库为空，无法执行检索")
            return []

        results = self._experiences.query(
            query_texts=[query],
            n_results=min(top_k, self._experiences.count()),
        )

        return self._format_search_results(results)

    def search_by_skills(self, skill_keywords: List[str], top_k: int = 8) -> List[Dict[str, Any]]:
        """
        通过技能关键词进行检索。

        Args:
            skill_keywords: 技能关键词列表
            top_k: 返回的最大结果数

        Returns:
            检索结果列表
        """
        if self._skills.count() == 0:
            return []

        query = " ".join(skill_keywords)
        results = self._skills.query(
            query_texts=[query],
            n_results=min(top_k, self._skills.count()),
        )

        # 通过技能索引反查经历条目
        experience_ids = []
        for metadata_list in results.get("metadatas", [[]]):
            for meta in metadata_list:
                exp_id = meta.get("experience_id", "")
                if exp_id:
                    experience_ids.append(exp_id)

        if not experience_ids:
            return []

        # 去重
        unique_ids = list(dict.fromkeys(experience_ids))
        try:
            exp_results = self._experiences.get(ids=unique_ids[:top_k])
            return self._format_get_results(exp_results)
        except Exception as e:
            logger.error("反查经历条目失败: %s", e)
            return []

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """
        获取所有经历条目。

        Returns:
            所有经历条目的列表
        """
        if self._experiences.count() == 0:
            return []

        results = self._experiences.get()
        return self._format_get_results(results)

    def delete_all(self) -> None:
        """清空所有集合中的数据。"""
        self._client.delete_collection(CHROMA_COLLECTION_EXPERIENCES)
        self._client.delete_collection(CHROMA_COLLECTION_SKILLS)
        # 重新创建空集合
        self._experiences = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_EXPERIENCES,
            metadata={"description": "精炼后的项目经历条目"},
        )
        self._skills = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_SKILLS,
            metadata={"description": "技能标签索引"},
        )
        logger.info("所有向量数据已清空")

    def get_stats(self) -> dict:
        """
        获取向量数据库统计信息。

        Returns:
            统计信息字典
        """
        all_entries = self.get_all_entries()

        # 统计技能分布
        skill_counter: Dict[str, int] = {}
        project_names: List[str] = []
        source_files: set = set()

        for entry in all_entries:
            meta = entry.get("metadata", {})
            project_names.append(meta.get("project_name", "未命名"))
            source_files.add(meta.get("source_file", ""))

            try:
                tags = json.loads(meta.get("skill_tags", "[]"))
                for tag in tags:
                    skill_counter[tag] = skill_counter.get(tag, 0) + 1
            except json.JSONDecodeError:
                pass

        # 按出现频次排序
        top_skills = sorted(skill_counter.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "total_experiences": self._experiences.count(),
            "total_skill_entries": self._skills.count(),
            "unique_projects": len(set(project_names)),
            "unique_source_files": len(source_files),
            "top_skills": top_skills,
            "project_names": list(set(project_names)),
        }

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _format_search_results(self, results: dict) -> List[Dict[str, Any]]:
        """将 ChromaDB query 结果格式化为统一的结果列表。"""
        formatted = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for idx in range(len(ids)):
            formatted.append({
                "id": ids[idx],
                "document": documents[idx] if idx < len(documents) else "",
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "distance": distances[idx] if idx < len(distances) else 0.0,
            })
        return formatted

    def _format_get_results(self, results: dict) -> List[Dict[str, Any]]:
        """将 ChromaDB get 结果格式化为统一的结果列表。"""
        formatted = []
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        for idx in range(len(ids)):
            formatted.append({
                "id": ids[idx],
                "document": documents[idx] if idx < len(documents) else "",
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
            })
        return formatted
