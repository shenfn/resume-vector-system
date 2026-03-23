# 简历向量匹配系统 (Resume Vector System)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

从 Obsidian 笔记中**全自动**提取项目经历，与岗位 JD 智能匹配，生成针对性简历内容。

> 解决一个核心痛点：Obsidian 里记录了大量项目经历，但内容杂乱（错误日志、AI 对话、调试过程混在一起），每次投简历都要手动翻找和整理。本系统实现从笔记到简历的全自动化管道。

## 核心功能

- **全自动精炼管道**：提供 Obsidian 路径 → 自动扫描、过滤噪音、提取经历、向量化存储，无需人工干预
- **智能岗位匹配**：粘贴 JD → 自动分析要求、语义检索、生成优化简历内容
- **知识库管理**：持久化存储经历条目，支持增量更新和全量重建
- **双路召回检索**：语义相似度 + 技能关键词双路融合，提升匹配精度

## 核心功能展示
<img width="1888" height="1265" alt="image" src="https://github.com/user-attachments/assets/1788b645-6a07-4d27-9f9d-a80f34c36400" />


## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/shenfn/resume-vector-system.git
cd resume-vector-system
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

复制 `.env.example` 为 `.env`，填入你自己的 MiniMax API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```
MINIMAX_API_KEY=你的MiniMax API Key
```

> ⚠️ `.env` 文件包含敏感信息，已被 `.gitignore` 排除，不会被提交到仓库。

### 4. 启动界面

```bash
streamlit run app.py
```

### 5. 使用流程

1. **Tab 1 - 内容库管理**：填入 Obsidian 路径和 PDF 简历路径，点击「增量更新」
2. **Tab 2 - 岗位匹配**：粘贴目标 JD，点击「一键匹配」
3. **Tab 3 - 系统测试**：查看数据库状态，执行检索测试

## 项目结构

```
resume-vector-system/
├── config/
│   ├── settings.py          # 全局配置（API、路径、模型参数）
│   └── prompts.py           # LLM Prompt 模板
├── pipeline/
│   ├── scanner.py           # Obsidian 文件扫描器（增量扫描、多编码）
│   ├── preprocessor.py      # 噪音过滤与预处理
│   ├── extractor.py         # LLM 精炼提取器（两层提取）
│   ├── vectorizer.py        # 向量化与 ChromaDB 管理
│   └── orchestrator.py      # 管道编排器（一键全自动）
├── matcher/
│   ├── jd_analyzer.py       # JD 解析（提取结构化要求）
│   ├── retriever.py         # 语义检索（双路召回融合排序）
│   └── generator.py         # 简历内容生成（STAR 格式）
├── storage/                  # 数据存储目录（自动创建，已 gitignore）
├── app.py                    # Streamlit 主界面
├── requirements.txt
├── .env.example              # 环境变量模板
└── .env                      # 环境变量（需自行创建，已 gitignore）
```

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| LLM | MiniMax API | OpenAI 兼容接口，可替换为其他模型 |
| 向量数据库 | ChromaDB | 本地持久化，轻量免费 |
| PDF 解析 | PyMuPDF | 速度快，格式保留好 |
| 界面 | Streamlit | 快速构建交互式 Web 应用 |
| 语言 | Python 3.10+ | |

## 处理流程

### 离线精炼管道（一次性处理）
```
Obsidian 目录 → 扫描 .md 文件 → 正则噪音过滤 → LLM 两层提取 → 向量化存入 ChromaDB
```

### 在线匹配管道（每次投简历时）
```
JD 文本 → 提取岗位要求 → 双路语义检索 → LLM 生成简历内容 → 匹配结果 + 差距分析
```

## 隐私说明

- 所有数据**仅存储在本地**，不会上传到任何第三方服务器
- API Key 通过 `.env` 文件管理，已被 `.gitignore` 排除
- 用户简历（PDF）不会被提交到版本控制
- `storage/` 目录下的所有运行时数据均已被 `.gitignore` 排除

## License

[MIT License](LICENSE)
