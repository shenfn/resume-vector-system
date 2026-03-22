"""
LLM 精炼提取器模块。
调用 MiniMax API，从清洁后的文本片段中提取结构化的项目经历条目。

集成 resume-experience-extractor skill 的核心方法论：
1. 技术还原：将技术细节转化为专业项目描述
2. 商业叙事：强调项目的业务价值和成果
3. STAR 格式：情境-任务-行动-结果

提取流程分两层：
- 第一层：项目识别（判断片段是否包含有价值的项目经历）
- 第二层：结构化提取（从有价值片段中提取完整经历条目）
"""

import json
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from openai import OpenAI

from config.settings import (
    MINIMAX_API_KEY,
    MINIMAX_BASE_URL,
    MINIMAX_MODEL,
    API_MAX_RETRIES,
    API_RETRY_DELAY_BASE,
    API_TIMEOUT,
    API_TEMPERATURE,
    EXTRACTION_MAX_INPUT_CHARS,
    EXTRACTION_CONFIDENCE_THRESHOLD,
)
from config.prompts import (
    PROJECT_IDENTIFICATION_PROMPT,
    EXPERIENCE_EXTRACTION_PROMPT,
)
from pipeline.preprocessor import CleanedChunk

logger = logging.getLogger(__name__)


@dataclass
class ExperienceEntry:
    """结构化的项目经历条目，可直接用于简历匹配和生成。"""
    project_name: str = ""
    tech_stack: List[str] = field(default_factory=list)
    role: str = ""
    core_challenge: str = ""
    solution: str = ""
    outcome: str = ""
    skill_tags: List[str] = field(default_factory=list)
    resume_bullet: str = ""
    business_value: str = ""
    detail_level: str = "brief"
    source_file: str = ""
    section_path: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperienceEntry":
        """从字典创建实例，忽略未知字段。"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class LLMClient:
    """
    MiniMax API 客户端封装。
    提供带重试机制和错误处理的 API 调用方法。
    """

    def __init__(self):
        """初始化 OpenAI 兼容客户端（MiniMax 使用 OpenAI 兼容接口）。"""
        if not MINIMAX_API_KEY:
            raise ValueError("MINIMAX_API_KEY 未配置，请检查 .env 文件")
        self._client = OpenAI(
            api_key=MINIMAX_API_KEY,
            base_url=MINIMAX_BASE_URL,
            timeout=API_TIMEOUT,
        )
        self._total_tokens_used = 0
        self._total_calls = 0

    def chat(self, prompt: str, max_tokens: int = 4096) -> Optional[str]:
        """
        发送聊天请求并返回模型响应文本。

        Args:
            prompt: 用户 prompt 文本
            max_tokens: 最大输出 token 数

        Returns:
            模型响应文本，失败时返回 None
        """
        for attempt in range(1, API_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=MINIMAX_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=API_TEMPERATURE,
                    max_tokens=max_tokens,
                )
                self._total_calls += 1
                if response.usage:
                    self._total_tokens_used += response.usage.total_tokens
                content = response.choices[0].message.content
                return content.strip() if content else None
            except Exception as e:
                delay = API_RETRY_DELAY_BASE ** attempt
                logger.warning(
                    "API 调用失败 (第 %d/%d 次): %s, %.1f 秒后重试",
                    attempt, API_MAX_RETRIES, e, delay
                )
                if attempt < API_MAX_RETRIES:
                    time.sleep(delay)
                else:
                    logger.error("API 调用最终失败: %s", e)
                    return None

    @property
    def total_tokens_used(self) -> int:
        """返回累计消耗的 token 数。"""
        return self._total_tokens_used

    @property
    def total_calls(self) -> int:
        """返回累计 API 调用次数。"""
        return self._total_calls


class Extractor:
    """
    LLM 精炼提取器。
    
    功能：
    1. 第一层：项目识别——判断文本片段是否包含有价值的项目经历
    2. 第二层：结构化提取——从有价值片段中提取完整的经历条目
    3. 批量处理：自动合并小片段以减少 API 调用次数
    4. 结果缓存：避免重复处理相同内容
    """

    def __init__(self, resume_content: str = ""):
        """
        初始化提取器。

        Args:
            resume_content: 用户现有简历的文本内容（作为价值判断的锚点和风格参考）
        """
        self._llm = LLMClient()
        self._resume_content = resume_content

    def extract_all(self, chunks: List[CleanedChunk]) -> List[ExperienceEntry]:
        """
        批量提取所有片段中的项目经历。

        Args:
            chunks: 预处理后的干净文本片段列表

        Returns:
            ExperienceEntry 列表
        """
        all_entries: List[ExperienceEntry] = []
        identified_chunks: List[CleanedChunk] = []

        # 第一层：项目识别
        logger.info("开始第一层提取：项目识别（共 %d 个片段）", len(chunks))
        for i, chunk in enumerate(chunks):
            logger.info("识别进度: %d/%d - %s", i + 1, len(chunks), chunk.source_file)
            has_value, confidence = self._identify_project(chunk)
            if has_value and confidence >= EXTRACTION_CONFIDENCE_THRESHOLD:
                identified_chunks.append(chunk)
                logger.info(
                    "  ✓ 发现有价值内容 (置信度: %.2f): %s > %s",
                    confidence, chunk.source_file, chunk.section_path
                )
            else:
                logger.debug(
                    "  ✗ 无有价值内容: %s > %s",
                    chunk.source_file, chunk.section_path
                )

        logger.info(
            "第一层完成: %d/%d 个片段包含有价值内容",
            len(identified_chunks), len(chunks)
        )

        # 第二层：结构化提取
        logger.info("开始第二层提取：结构化提取")
        for i, chunk in enumerate(identified_chunks):
            logger.info("提取进度: %d/%d - %s", i + 1, len(identified_chunks), chunk.source_file)
            entries = self._extract_experiences(chunk)
            all_entries.extend(entries)
            logger.info("  提取到 %d 条经历条目", len(entries))

        logger.info(
            "提取完成: 共获得 %d 条经历条目, 消耗 %d tokens, %d 次 API 调用",
            len(all_entries), self._llm.total_tokens_used, self._llm.total_calls
        )

        return all_entries

    def _identify_project(self, chunk: CleanedChunk) -> tuple:
        """
        第一层：判断文本片段是否包含有价值的项目经历。

        Args:
            chunk: 清洁后的文本片段

        Returns:
            (has_value: bool, confidence: float) 元组
        """
        # 截断过长内容以控制 token 消耗
        note_content = chunk.content[:EXTRACTION_MAX_INPUT_CHARS]

        prompt = PROJECT_IDENTIFICATION_PROMPT.format(
            resume_content=self._resume_content[:3000],
            note_content=note_content,
        )

        response = self._llm.chat(prompt, max_tokens=256)
        if not response:
            return False, 0.0

        try:
            result = self._parse_json_response(response)
            has_value = result.get("has_value", False)
            confidence = float(result.get("confidence", 0.0))
            return has_value, confidence
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("解析项目识别结果失败: %s, 原始响应: %s", e, response[:200])
            return False, 0.0

    def _extract_experiences(self, chunk: CleanedChunk) -> List[ExperienceEntry]:
        """
        第二层：从有价值片段中提取结构化经历条目。

        Args:
            chunk: 已确认包含有价值内容的文本片段

        Returns:
            ExperienceEntry 列表
        """
        # 对于大片段，按 EXTRACTION_MAX_INPUT_CHARS 拆分处理
        content = chunk.content
        all_entries = []

        segments = self._split_content_for_extraction(content)
        for segment in segments:
            prompt = EXPERIENCE_EXTRACTION_PROMPT.format(
                resume_content=self._resume_content[:3000],
                note_content=segment,
                source_file=chunk.source_file,
            )

            response = self._llm.chat(prompt, max_tokens=4096)
            if not response:
                continue

            try:
                entries_data = self._parse_json_response(response)
                # 响应可能是单个对象或数组
                if isinstance(entries_data, dict):
                    entries_data = [entries_data]
                if not isinstance(entries_data, list):
                    logger.warning("提取结果格式异常，跳过: %s", type(entries_data))
                    continue

                for entry_dict in entries_data:
                    entry = ExperienceEntry.from_dict(entry_dict)
                    entry.source_file = chunk.source_file
                    entry.section_path = chunk.section_path
                    all_entries.append(entry)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "解析提取结果失败: %s, 来源: %s, 响应: %s",
                    e, chunk.source_file, response[:200]
                )

        return all_entries

    def _split_content_for_extraction(self, content: str) -> List[str]:
        """
        将过长内容拆分为适合单次 API 调用的片段。

        Args:
            content: 待拆分的文本

        Returns:
            文本片段列表
        """
        if len(content) <= EXTRACTION_MAX_INPUT_CHARS:
            return [content]

        # 按段落分割，尽量在段落边界拆分
        paragraphs = content.split("\n\n")
        segments = []
        current_segment = ""

        for para in paragraphs:
            if len(current_segment) + len(para) + 2 > EXTRACTION_MAX_INPUT_CHARS:
                if current_segment:
                    segments.append(current_segment)
                current_segment = para
            else:
                current_segment = current_segment + "\n\n" + para if current_segment else para

        if current_segment:
            segments.append(current_segment)

        return segments

    def _parse_json_response(self, response: str) -> Any:
        """
        解析 LLM 返回的 JSON 响应。
        处理常见的格式问题（如被 markdown 代码块包裹）。

        Args:
            response: LLM 响应文本

        Returns:
            解析后的 JSON 对象

        Raises:
            json.JSONDecodeError: 解析失败时抛出
        """
        text = response.strip()

        # 移除 markdown 代码块标记
        if text.startswith("```"):
            lines = text.split("\n")
            # 移除第一行（```json 或 ```）和最后一行（```）
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            text = "\n".join(lines).strip()

        return json.loads(text)

    def get_extraction_stats(self) -> dict:
        """返回提取统计信息。"""
        return {
            "total_api_calls": self._llm.total_calls,
            "total_tokens_used": self._llm.total_tokens_used,
        }
