"""
语义检索模块。
从 ChromaDB 中检索与 JD 最匹配的经历条目。

检索策略（双路召回 + 融合排序）：
1. 语义检索：用 JD 核心职责描述做向量相似度搜索
2. 关键词检索：用 JD 技能关键词做技能标签匹配
3. 融合排序：关键词命中权重 0.4 + 语义相似度权重 0.6
"""

import json
import logging
from typing import List, Dict, Any

from config.settings import RETRIEVAL_TOP_K, KEYWORD_WEIGHT, SEMANTIC_WEIGHT
from pipeline.vectorizer import VectorStore
from matcher.jd_analyzer import JDRequirements

logger = logging.getLogger(__name__)


class Retriever:
    """
    语义检索器。

    功能：
    1. 接收 JDRequirements，生成检索查询
    2. 双路召回：语义检索 + 关键词检索
    3. 融合排序并返回 Top-K 结果
    """

    def __init__(self, vector_store: VectorStore):
        """
        初始化检索器。

        Args:
            vector_store: ChromaDB 向量存储实例
        """
        self._store = vector_store

    def retrieve(self, jd_requirements: JDRequirements, top_k: int = RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        """
        根据岗位要求检索最匹配的经历条目。

        Args:
            jd_requirements: 结构化的岗位要求
            top_k: 返回的最大结果数

        Returns:
            检索结果列表，每个元素包含：
            - id: 条目 ID
            - document: 原始文档文本
            - metadata: 结构化元数据
            - score: 融合后的相关度得分（0.0-1.0，越高越相关）
            - match_type: 匹配类型（semantic / keyword / both）
        """
        # 1. 语义检索：用 JD 核心职责做向量搜索
        search_query = jd_requirements.get_search_query()
        semantic_results = self._store.search_by_text(search_query, top_k=top_k * 2)
        logger.info("语义检索: 返回 %d 条结果", len(semantic_results))

        # 2. 关键词检索：用技能关键词做匹配
        all_keywords = jd_requirements.get_all_keywords()
        keyword_results = self._store.search_by_skills(all_keywords, top_k=top_k * 2)
        logger.info("关键词检索: 返回 %d 条结果", len(keyword_results))

        # 3. 融合排序
        fused = self._fuse_results(semantic_results, keyword_results, all_keywords)

        # 4. 截取 Top-K
        top_results = fused[:top_k]

        logger.info(
            "检索完成: 融合后 %d 条, 返回 Top-%d",
            len(fused), len(top_results),
        )

        return top_results

    def _fuse_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        keywords: List[str],
    ) -> List[Dict[str, Any]]:
        """
        融合语义检索和关键词检索结果。

        算法：
        - 语义得分 = 1 - distance（ChromaDB distance 越小越相似）
        - 关键词得分 = 技能标签与 JD 关键词的 Jaccard 相似度
        - 融合得分 = semantic_weight * 语义得分 + keyword_weight * 关键词得分

        Args:
            semantic_results: 语义检索结果
            keyword_results: 关键词检索结果
            keywords: JD 关键词列表

        Returns:
            融合排序后的结果列表
        """
        # 收集所有候选结果（按 ID 去重）
        candidates: Dict[str, Dict[str, Any]] = {}

        # 处理语义检索结果
        for result in semantic_results:
            rid = result["id"]
            distance = result.get("distance", 1.0)
            # ChromaDB 默认使用 L2 距离，转为相似度得分
            semantic_score = max(0.0, 1.0 - distance / 2.0)

            if rid not in candidates:
                candidates[rid] = {
                    "id": rid,
                    "document": result.get("document", ""),
                    "metadata": result.get("metadata", {}),
                    "semantic_score": semantic_score,
                    "keyword_score": 0.0,
                    "match_type": "semantic",
                }
            else:
                candidates[rid]["semantic_score"] = max(
                    candidates[rid]["semantic_score"], semantic_score
                )

        # 处理关键词检索结果
        keywords_lower = set(kw.lower() for kw in keywords)
        for result in keyword_results:
            rid = result["id"]

            # 计算关键词得分：技能标签与 JD 关键词的 Jaccard 相似度
            metadata = result.get("metadata", {})
            try:
                skill_tags = json.loads(metadata.get("skill_tags", "[]"))
            except json.JSONDecodeError:
                skill_tags = []
            tags_lower = set(tag.lower() for tag in skill_tags)

            intersection = len(tags_lower & keywords_lower)
            union = len(tags_lower | keywords_lower)
            keyword_score = intersection / max(union, 1)

            if rid not in candidates:
                candidates[rid] = {
                    "id": rid,
                    "document": result.get("document", ""),
                    "metadata": metadata,
                    "semantic_score": 0.0,
                    "keyword_score": keyword_score,
                    "match_type": "keyword",
                }
            else:
                candidates[rid]["keyword_score"] = max(
                    candidates[rid]["keyword_score"], keyword_score
                )
                candidates[rid]["match_type"] = "both"

        # 计算融合得分并排序
        fused_list = []
        for cand in candidates.values():
            fused_score = (
                SEMANTIC_WEIGHT * cand["semantic_score"]
                + KEYWORD_WEIGHT * cand["keyword_score"]
            )
            cand["score"] = round(fused_score, 4)
            fused_list.append(cand)

        # 按融合得分降序排序
        fused_list.sort(key=lambda x: x["score"], reverse=True)

        return fused_list
