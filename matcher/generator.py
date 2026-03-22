"""
简历内容生成模块。
基于检索结果和 JD 要求，生成针对性的简历内容。

集成 jd-match-analyzer skill 的核心方法论：
1. 对照 JD 核心要求，识别匹配点和差距
2. 用 STAR 格式改写经历条目，突出与岗位要求的匹配点
3. 自然融入 JD 高频关键词，优先补充量化成果
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from pipeline.extractor import LLMClient
from config.prompts import RESUME_GENERATION_PROMPT
from matcher.jd_analyzer import JDRequirements

logger = logging.getLogger(__name__)


@dataclass
class MatchedExperience:
    """匹配后的单条经历。"""
    project_name: str = ""
    relevance_score: float = 0.0
    optimized_bullet: str = ""
    matched_requirements: List[str] = field(default_factory=list)
    tech_stack_highlight: List[str] = field(default_factory=list)


@dataclass
class GapItem:
    """差距分析条目。"""
    requirement: str = ""
    status: str = "not_matched"  # fully_matched / partially_matched / not_matched
    suggestion: str = ""


@dataclass
class MatchResult:
    """完整的匹配结果。"""
    jd_position: str = ""
    jd_company: str = ""
    selected_experiences: List[MatchedExperience] = field(default_factory=list)
    skill_summary: str = ""
    gap_analysis: List[GapItem] = field(default_factory=list)
    raw_retrieval_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "jd_position": self.jd_position,
            "jd_company": self.jd_company,
            "selected_experiences": [asdict(e) for e in self.selected_experiences],
            "skill_summary": self.skill_summary,
            "gap_analysis": [asdict(g) for g in self.gap_analysis],
        }

    def to_display_text(self) -> str:
        """
        生成用于界面展示的可读文本。
        
        Returns:
            格式化的匹配结果文本
        """
        lines = []
        lines.append(f"## 岗位匹配结果：{self.jd_position}")
        if self.jd_company:
            lines.append(f"**公司**：{self.jd_company}")
        lines.append("")

        # 技能摘要
        if self.skill_summary:
            lines.append(f"**技能匹配摘要**：{self.skill_summary}")
            lines.append("")

        # 推荐经历
        lines.append("### 推荐项目经历")
        for i, exp in enumerate(self.selected_experiences, 1):
            lines.append(f"\n**{i}. {exp.project_name}**（相关度：{exp.relevance_score:.0%}）")
            lines.append(f"> {exp.optimized_bullet}")
            if exp.matched_requirements:
                lines.append(f"  - 匹配要求：{', '.join(exp.matched_requirements)}")
            if exp.tech_stack_highlight:
                lines.append(f"  - 技术亮点：{', '.join(exp.tech_stack_highlight)}")

        # 差距分析
        if self.gap_analysis:
            lines.append("\n### 差距分析")
            for gap in self.gap_analysis:
                status_icon = {
                    "fully_matched": "✅",
                    "partially_matched": "⚠️",
                    "not_matched": "❌",
                }.get(gap.status, "❓")
                lines.append(f"- {status_icon} **{gap.requirement}**：{gap.suggestion}")

        return "\n".join(lines)


class ResumeGenerator:
    """
    简历内容生成器。

    功能：
    1. 接收检索结果和 JD 要求
    2. 调用 LLM 生成针对性的简历项目经历
    3. 输出匹配点、差距分析和优化建议
    """

    def __init__(self, resume_content: str = ""):
        """
        初始化生成器。

        Args:
            resume_content: 用户现有简历文本（作为风格参考）
        """
        self._llm = LLMClient()
        self._resume_content = resume_content

    def generate(
        self,
        jd_requirements: JDRequirements,
        retrieval_results: List[Dict[str, Any]],
    ) -> MatchResult:
        """
        根据 JD 要求和检索结果，生成优化的简历内容。

        Args:
            jd_requirements: 结构化的岗位要求
            retrieval_results: 从 ChromaDB 检索到的经历条目列表

        Returns:
            MatchResult 完整的匹配结果
        """
        if not retrieval_results:
            logger.warning("没有检索到任何经历条目，无法生成简历内容")
            return MatchResult(
                jd_position=jd_requirements.position,
                jd_company=jd_requirements.company,
                gap_analysis=[
                    GapItem(
                        requirement="经历条目库",
                        status="not_matched",
                        suggestion="经历条目库为空，请先运行精炼管道处理 Obsidian 笔记",
                    )
                ],
            )

        # 准备检索结果的文本描述
        experience_entries = self._format_retrieval_results(retrieval_results)

        logger.info(
            "开始生成简历内容: 岗位=%s, 候选经历=%d条",
            jd_requirements.position, len(retrieval_results),
        )

        prompt = RESUME_GENERATION_PROMPT.format(
            jd_analysis=jd_requirements.to_json(),
            experience_entries=experience_entries,
            resume_content=self._resume_content[:3000],
        )

        response = self._llm.chat(prompt, max_tokens=4096)
        if not response:
            logger.error("LLM 未返回有效响应")
            return MatchResult(
                jd_position=jd_requirements.position,
                jd_company=jd_requirements.company,
                raw_retrieval_results=retrieval_results,
            )

        try:
            result_data = self._parse_json_response(response)
            match_result = self._build_match_result(result_data, jd_requirements, retrieval_results)
            logger.info(
                "简历内容生成完成: %d 条推荐经历, %d 条差距分析",
                len(match_result.selected_experiences),
                len(match_result.gap_analysis),
            )
            return match_result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error("解析生成结果失败: %s, 原始响应: %s", e, response[:300])
            # 降级处理：直接使用检索结果构建基础匹配
            return self._build_fallback_result(jd_requirements, retrieval_results)

    def _format_retrieval_results(self, results: List[Dict[str, Any]]) -> str:
        """
        将检索结果格式化为 LLM 可读的文本。

        Args:
            results: 检索结果列表

        Returns:
            格式化的文本
        """
        formatted_entries = []
        for i, result in enumerate(results, 1):
            meta = result.get("metadata", {})
            entry_text = f"""条目 {i}:
- 项目名称: {meta.get('project_name', '未命名')}
- 技术栈: {meta.get('tech_stack', '[]')}
- 角色: {meta.get('role', '')}
- 核心挑战: {meta.get('core_challenge', '')}
- 成果: {meta.get('outcome', '')}
- 技能标签: {meta.get('skill_tags', '[]')}
- 业务价值: {meta.get('business_value', '')}
- 简历条目: {result.get('document', '')}
- 匹配得分: {result.get('score', 0.0):.4f}"""
            formatted_entries.append(entry_text)

        return "\n\n".join(formatted_entries)

    def _build_match_result(
        self,
        result_data: dict,
        jd_requirements: JDRequirements,
        retrieval_results: List[Dict[str, Any]],
    ) -> MatchResult:
        """
        从 LLM 响应构建 MatchResult 对象。

        Args:
            result_data: LLM 返回的 JSON 数据
            jd_requirements: 岗位要求
            retrieval_results: 原始检索结果

        Returns:
            MatchResult 对象
        """
        # 解析推荐经历
        experiences = []
        for exp_data in result_data.get("selected_experiences", []):
            experiences.append(MatchedExperience(
                project_name=exp_data.get("project_name", ""),
                relevance_score=float(exp_data.get("relevance_score", 0.0)),
                optimized_bullet=exp_data.get("optimized_bullet", ""),
                matched_requirements=exp_data.get("matched_requirements", []),
                tech_stack_highlight=exp_data.get("tech_stack_highlight", []),
            ))

        # 解析差距分析
        gaps = []
        for gap_data in result_data.get("gap_analysis", []):
            gaps.append(GapItem(
                requirement=gap_data.get("requirement", ""),
                status=gap_data.get("status", "not_matched"),
                suggestion=gap_data.get("suggestion", ""),
            ))

        return MatchResult(
            jd_position=jd_requirements.position,
            jd_company=jd_requirements.company,
            selected_experiences=experiences,
            skill_summary=result_data.get("skill_summary", ""),
            gap_analysis=gaps,
            raw_retrieval_results=retrieval_results,
        )

    def _build_fallback_result(
        self,
        jd_requirements: JDRequirements,
        retrieval_results: List[Dict[str, Any]],
    ) -> MatchResult:
        """
        降级处理：LLM 解析失败时，直接使用检索结果构建基础匹配。

        Args:
            jd_requirements: 岗位要求
            retrieval_results: 检索结果

        Returns:
            基础的 MatchResult 对象
        """
        logger.warning("使用降级策略构建匹配结果")

        experiences = []
        for result in retrieval_results[:5]:
            meta = result.get("metadata", {})
            experiences.append(MatchedExperience(
                project_name=meta.get("project_name", "未命名"),
                relevance_score=result.get("score", 0.0),
                optimized_bullet=result.get("document", ""),
                matched_requirements=[],
                tech_stack_highlight=[],
            ))

        return MatchResult(
            jd_position=jd_requirements.position,
            jd_company=jd_requirements.company,
            selected_experiences=experiences,
            skill_summary="(降级模式：LLM 解析失败，显示原始检索结果)",
            gap_analysis=[],
            raw_retrieval_results=retrieval_results,
        )

    def _parse_json_response(self, response: str) -> dict:
        """解析 LLM 返回的 JSON 响应。"""
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            text = "\n".join(lines).strip()
        return json.loads(text)
