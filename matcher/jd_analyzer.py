"""
JD 解析模块。
分析招聘 JD，提取结构化的岗位要求。

集成 jd-match-analyzer skill 的核心方法论：
1. 提炼 JD 最核心的 5-8 个硬性/软性要求
2. 提取技能、技术栈、年限、关键词、量化成果、软技能等维度
3. 生成用于语义匹配的关键词列表
"""

import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

from pipeline.extractor import LLMClient
from config.prompts import JD_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class JDRequirements:
    """结构化的岗位要求。"""
    position: str = ""
    company: str = ""
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    experience_requirements: List[str] = field(default_factory=list)
    key_responsibilities: List[str] = field(default_factory=list)
    soft_skills: List[str] = field(default_factory=list)
    match_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return asdict(self)

    def to_json(self) -> str:
        """转换为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "JDRequirements":
        """从字典创建实例，忽略未知字段。"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def get_all_keywords(self) -> List[str]:
        """
        汇总所有关键词（用于语义检索）。
        合并 required_skills + preferred_skills + match_keywords，去重。
        """
        all_kw = set()
        all_kw.update(self.required_skills)
        all_kw.update(self.preferred_skills)
        all_kw.update(self.match_keywords)
        return list(all_kw)

    def get_search_query(self) -> str:
        """
        生成用于向量检索的查询文本。
        将核心职责和技能要求拼接为一段自然语言描述。
        """
        parts = []
        if self.position:
            parts.append(f"岗位：{self.position}")
        if self.key_responsibilities:
            parts.append("核心职责：" + "；".join(self.key_responsibilities))
        if self.required_skills:
            parts.append("必须技能：" + "、".join(self.required_skills))
        if self.experience_requirements:
            parts.append("经验要求：" + "；".join(self.experience_requirements))
        return "\n".join(parts)


class JDAnalyzer:
    """
    JD 分析器。

    功能：
    1. 接收完整的 JD 文本
    2. 调用 LLM 提取结构化岗位要求
    3. 生成用于语义检索的查询文本和关键词列表
    """

    def __init__(self):
        """初始化 JD 分析器。"""
        self._llm = LLMClient()

    def analyze(self, jd_content: str) -> JDRequirements:
        """
        分析 JD 文本，提取结构化的岗位要求。

        Args:
            jd_content: 完整的招聘 JD 文本

        Returns:
            JDRequirements 结构化岗位要求

        Raises:
            ValueError: JD 内容为空或解析失败
        """
        if not jd_content or not jd_content.strip():
            raise ValueError("JD 内容不能为空")

        logger.info("开始分析 JD...")

        prompt = JD_ANALYSIS_PROMPT.format(jd_content=jd_content)
        response = self._llm.chat(prompt, max_tokens=2048)

        if not response:
            raise ValueError("LLM 未返回有效响应，请检查 API 配置")

        try:
            result = self._parse_json_response(response)
            requirements = JDRequirements.from_dict(result)

            logger.info(
                "JD 分析完成: 岗位=%s, 必须技能=%d项, 加分技能=%d项, 关键词=%d个",
                requirements.position,
                len(requirements.required_skills),
                len(requirements.preferred_skills),
                len(requirements.match_keywords),
            )

            return requirements

        except (json.JSONDecodeError, ValueError) as e:
            logger.error("解析 JD 分析结果失败: %s, 原始响应: %s", e, response[:300])
            raise ValueError(f"JD 分析结果解析失败: {e}") from e

    def _parse_json_response(self, response: str) -> dict:
        """
        解析 LLM 返回的 JSON 响应。

        Args:
            response: LLM 响应文本

        Returns:
            解析后的字典

        Raises:
            json.JSONDecodeError: 解析失败
        """
        text = response.strip()

        # 移除 markdown 代码块标记
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            text = "\n".join(lines).strip()

        return json.loads(text)
