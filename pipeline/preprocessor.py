"""
噪音过滤与预处理模块。
自动过滤错误日志、AI 对话、重复内容等噪音，输出干净的文本片段。

处理策略：
1. 小文件（< 5000 字）：整体作为一个片段处理
2. 中文件（5000-30000 字）：按 Markdown 标题拆分为章节
3. 大文件（> 30000 字）：按标题拆分 + 强力噪音过滤
"""

import re
import logging
from dataclasses import dataclass
from typing import List

from config.settings import CHUNK_MIN_LENGTH
from pipeline.scanner import FileManifest

logger = logging.getLogger(__name__)


@dataclass
class CleanedChunk:
    """预处理后的干净文本片段。"""
    content: str            # 清洁后的文本内容
    source_file: str        # 来源文件的相对路径
    section_path: str       # 章节路径（如 "项目概述 > 技术方案"）
    original_chars: int     # 原始字符数
    cleaned_chars: int      # 清洁后字符数
    noise_ratio: float      # 噪音比例（0.0-1.0）


class Preprocessor:
    """
    文本预处理器。
    
    功能：
    1. 按 Markdown 标题拆分大文件为章节
    2. 使用正则表达式过滤常见噪音模式
    3. 去除重复内容和无效片段
    4. 输出结构化的干净文本片段列表
    """

    # 噪音正则模式列表（按优先级排列）
    NOISE_PATTERNS = [
        # 1. Python/Java/JS 堆栈跟踪信息
        (r"Traceback \(most recent call last\):[\s\S]*?(?=\n\n|\Z)", "stack_trace"),
        (r"(?:Exception|Error|TypeError|ValueError|KeyError|AttributeError|ImportError|ModuleNotFoundError|RuntimeError|FileNotFoundError|PermissionError|OSError|IOError|ConnectionError|TimeoutError)\s*:.*(?:\n\s+.*)*", "exception"),
        # 2. 常见日志行（DEBUG/WARN/ERROR 级别）
        (r"^(?:\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}:\d{2}[.,]?\d*\s*)?(?:DEBUG|WARN(?:ING)?|ERROR|INFO|TRACE)\s*[:\|\[\]].+$", "log_line"),
        # 3. npm / pip / yarn 安装输出
        (r"(?:npm\s+(?:WARN|ERR!|notice)|pip\s+(?:install|download)|yarn\s+(?:add|install)).*(?:\n.*){0,5}", "package_install"),
        # 4. 大段 JSON/YAML/XML 配置块（超过 500 字符的代码块）
        (r"```(?:json|yaml|yml|xml|toml|ini|env|conf)[\s\S]{500,}?```", "config_block"),
        # 5. 重复的命令行提示符和输出
        (r"(?:^[>$#%]\s*.+\n?){3,}", "cli_output"),
        # 6. Git diff 输出
        (r"^(?:diff --git|index [a-f0-9]+|---\s+a/|\+\+\+\s+b/)[\s\S]*?(?=\n(?:diff --git|\Z))", "git_diff"),
        # 7. 长的纯 URL 列表
        (r"(?:https?://\S+\n?){5,}", "url_list"),
    ]

    # AI 对话模式
    AI_CONVERSATION_PATTERNS = [
        # 通用 AI 对话标记
        (r"^>\s*(?:User|Human|用户|我)\s*[:：][\s\S]*?(?=^>\s*(?:Assistant|AI|Claude|ChatGPT|助手|回答)|$)", "ai_user_msg"),
        (r"^>\s*(?:Assistant|AI|Claude|ChatGPT|助手|回答)\s*[:：][\s\S]*?(?=^>\s*(?:User|Human|用户|我)|$)", "ai_assistant_msg"),
        # Cursor/Windsurf 风格的对话
        (r"(?:^---\n)?(?:^(?:Human|Assistant|User|AI):.*\n(?:.*\n)*?)(?=^(?:Human|Assistant|User|AI):|^---|\Z)", "ai_dialog"),
    ]

    def __init__(self):
        """初始化预处理器，编译正则表达式。"""
        self._compiled_noise = [
            (re.compile(pattern, re.MULTILINE), label)
            for pattern, label in self.NOISE_PATTERNS
        ]
        self._compiled_ai = [
            (re.compile(pattern, re.MULTILINE), label)
            for pattern, label in self.AI_CONVERSATION_PATTERNS
        ]

    def process(self, manifests: List[FileManifest]) -> List[CleanedChunk]:
        """
        批量处理文件清单，返回清洁后的文本片段列表。

        Args:
            manifests: 文件清单列表

        Returns:
            CleanedChunk 列表
        """
        all_chunks: List[CleanedChunk] = []
        for manifest in manifests:
            try:
                chunks = self._process_single_file(manifest)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error("预处理文件失败 %s: %s", manifest.relative_path, e)

        logger.info(
            "预处理完成: 输入 %d 个文件, 输出 %d 个有效片段",
            len(manifests), len(all_chunks)
        )
        return all_chunks

    def _process_single_file(self, manifest: FileManifest) -> List[CleanedChunk]:
        """
        处理单个文件，根据文件大小选择不同策略。

        Args:
            manifest: 文件清单条目

        Returns:
            该文件产出的 CleanedChunk 列表
        """
        content = manifest.content
        relative_path = manifest.relative_path

        if manifest.size_category == "small":
            # 小文件：整体作为一个片段，直接过滤噪音
            cleaned = self._clean_text(content)
            if len(cleaned) < CHUNK_MIN_LENGTH:
                return []
            return [CleanedChunk(
                content=cleaned,
                source_file=relative_path,
                section_path="(全文)",
                original_chars=len(content),
                cleaned_chars=len(cleaned),
                noise_ratio=1.0 - len(cleaned) / max(len(content), 1),
            )]
        else:
            # 中/大文件：按标题拆分后逐段过滤
            sections = self._split_by_headings(content)
            chunks = []
            for section_title, section_content in sections:
                cleaned = self._clean_text(section_content)
                if len(cleaned) < CHUNK_MIN_LENGTH:
                    continue
                chunks.append(CleanedChunk(
                    content=cleaned,
                    source_file=relative_path,
                    section_path=section_title,
                    original_chars=len(section_content),
                    cleaned_chars=len(cleaned),
                    noise_ratio=1.0 - len(cleaned) / max(len(section_content), 1),
                ))
            return chunks

    def _split_by_headings(self, content: str) -> List[tuple]:
        """
        按 Markdown 标题拆分文本为章节。

        Args:
            content: Markdown 文本

        Returns:
            (章节标题, 章节内容) 的元组列表
        """
        # 匹配 Markdown 标题行（# ~ ####）
        heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(content))

        if not matches:
            # 没有标题，整体返回
            return [("(无标题)", content)]

        sections = []
        for i, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()
            if section_content:
                sections.append((title, section_content))

        # 如果标题前还有内容
        if matches[0].start() > 0:
            preamble = content[:matches[0].start()].strip()
            if preamble:
                sections.insert(0, ("(前言)", preamble))

        return sections

    def _clean_text(self, text: str) -> str:
        """
        对文本执行全部噪音过滤规则。

        Args:
            text: 原始文本

        Returns:
            过滤后的文本
        """
        cleaned = text

        # 1. 过滤噪音模式
        for pattern, label in self._compiled_noise:
            before_len = len(cleaned)
            cleaned = pattern.sub("", cleaned)
            removed = before_len - len(cleaned)
            if removed > 0:
                logger.debug("过滤 [%s]: 移除 %d 字符", label, removed)

        # 2. 过滤 AI 对话内容（保守处理：只过滤明确的对话格式）
        for pattern, label in self._compiled_ai:
            cleaned = pattern.sub("", cleaned)

        # 3. 移除空的代码块
        cleaned = re.sub(r"```\w*\s*```", "", cleaned)

        # 4. 移除连续空行（保留最多一个空行）
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        # 5. 移除行首行尾空白
        cleaned = cleaned.strip()

        return cleaned

    def get_preprocessing_report(self, chunks: List[CleanedChunk]) -> dict:
        """
        生成预处理统计报告。

        Args:
            chunks: 清洁后的片段列表

        Returns:
            统计信息字典
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_original_chars": 0,
                "total_cleaned_chars": 0,
                "avg_noise_ratio": 0.0,
            }

        total_original = sum(c.original_chars for c in chunks)
        total_cleaned = sum(c.cleaned_chars for c in chunks)
        avg_noise = sum(c.noise_ratio for c in chunks) / len(chunks)

        return {
            "total_chunks": len(chunks),
            "total_original_chars": total_original,
            "total_cleaned_chars": total_cleaned,
            "overall_noise_ratio": 1.0 - total_cleaned / max(total_original, 1),
            "avg_chunk_noise_ratio": round(avg_noise, 3),
            "unique_source_files": len(set(c.source_file for c in chunks)),
        }
