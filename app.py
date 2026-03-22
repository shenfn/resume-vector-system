"""
Streamlit 主界面。
提供三个 Tab：内容库管理、岗位匹配、系统测试。

用户交互流程：
1. Tab 1（内容库管理）：输入 Obsidian 路径和 PDF 简历路径 → 一键构建经历条目库
2. Tab 2（岗位匹配）：粘贴 JD → 一键匹配 → 查看生成的简历内容
3. Tab 3（系统测试）：查看数据库统计、执行测试验证
"""

import sys
import json
import logging
from pathlib import Path

import streamlit as st

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import STORAGE_DIR, EXPORTS_DIR
from pipeline.orchestrator import PipelineOrchestrator, PipelineReport
from pipeline.vectorizer import VectorStore
from matcher.jd_analyzer import JDAnalyzer
from matcher.retriever import Retriever
from matcher.generator import ResumeGenerator

logger = logging.getLogger(__name__)

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="简历向量匹配系统",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """初始化 Streamlit session state。"""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = PipelineOrchestrator()
    if "resume_content" not in st.session_state:
        st.session_state.resume_content = ""
    if "pipeline_report" not in st.session_state:
        st.session_state.pipeline_report = None
    if "match_result" not in st.session_state:
        st.session_state.match_result = None
    if "match_history" not in st.session_state:
        st.session_state.match_history = []


def main():
    """主入口函数。"""
    init_session_state()

    st.title("📄 简历向量匹配系统")
    st.markdown("*从 Obsidian 笔记中自动提取项目经历，与岗位 JD 智能匹配，生成针对性简历内容*")

    tab1, tab2, tab3 = st.tabs(["📦 内容库管理", "🎯 岗位匹配", "🧪 系统测试"])

    with tab1:
        render_pipeline_tab()

    with tab2:
        render_matching_tab()

    with tab3:
        render_test_tab()


# ============================================================
# Tab 1: 内容库管理
# ============================================================
def render_pipeline_tab():
    """渲染内容库管理 Tab。"""
    st.header("内容库管理")
    st.markdown("提供 Obsidian 目录路径和 PDF 简历路径，系统自动完成全部处理。")

    col1, col2 = st.columns(2)
    with col1:
        obsidian_path = st.text_input(
            "Obsidian Vault 路径",
            placeholder="例如: E:\\Obsidian\\我的笔记",
            help="Obsidian vault 的根目录路径，系统会递归扫描其中的所有 .md 文件",
        )
    with col2:
        resume_path = st.text_input(
            "PDF 简历路径",
            placeholder="例如: D:\\简历\\我的简历.pdf",
            help="你的 PDF 简历文件路径，用作内容提取的价值判断锚点和风格参考",
        )

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        run_incremental = st.button("🚀 增量更新", type="primary", use_container_width=True)
    with col_btn2:
        run_full = st.button("🔄 全量重建", use_container_width=True)
    with col_btn3:
        clear_cache = st.button("🗑️ 清空缓存", use_container_width=True)

    # 执行管道
    if run_incremental or run_full:
        if not obsidian_path or not resume_path:
            st.error("请填写 Obsidian 目录路径和 PDF 简历路径")
            return

        if not Path(obsidian_path).exists():
            st.error(f"Obsidian 目录不存在: {obsidian_path}")
            return

        if not Path(resume_path).exists():
            st.error(f"简历文件不存在: {resume_path}")
            return

        orchestrator = st.session_state.orchestrator

        with st.spinner("正在处理中，请稍候...（全自动，无需人工干预）"):
            progress_bar = st.progress(0, text="初始化...")
            try:
                if run_full:
                    progress_bar.progress(10, text="清空缓存和向量库...")
                    report = orchestrator.run_full_rebuild(obsidian_path, resume_path)
                else:
                    progress_bar.progress(10, text="开始增量扫描...")
                    report = orchestrator.run_incremental(obsidian_path, resume_path)

                progress_bar.progress(100, text="处理完成!")
                st.session_state.pipeline_report = report

                if report.status == "completed":
                    st.success("✅ 管道执行成功!")
                else:
                    st.error(f"❌ 管道执行失败: {report.error_message}")

            except Exception as e:
                progress_bar.progress(100, text="处理失败")
                st.error(f"❌ 执行出错: {e}")
                logger.error("管道执行异常: %s", e, exc_info=True)

    if clear_cache:
        st.session_state.orchestrator = PipelineOrchestrator()
        st.session_state.orchestrator._scanner.clear_cache()
        st.session_state.orchestrator._vector_store.delete_all()
        st.success("✅ 缓存和向量库已清空")

    # 显示执行报告
    report = st.session_state.pipeline_report
    if report is None:
        report = st.session_state.orchestrator.get_last_report()

    if report:
        render_pipeline_report(report)

    # 显示当前数据库状态
    render_db_stats()


def render_pipeline_report(report: PipelineReport):
    """渲染管道执行报告。"""
    st.subheader("📊 执行报告")

    status_color = {"completed": "🟢", "failed": "🔴", "running": "🟡"}.get(report.status, "⚪")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("状态", f"{status_color} {report.status}")
    with col2:
        st.metric("耗时", f"{report.duration_seconds:.1f} 秒")
    with col3:
        st.metric("扫描文件", report.scanned_files)
    with col4:
        st.metric("提取条目", report.extracted_entries)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("清洁片段", report.cleaned_chunks)
    with col6:
        st.metric("噪音比例", f"{report.noise_ratio:.1%}")
    with col7:
        st.metric("API 调用", report.api_calls)
    with col8:
        st.metric("Token 消耗", f"{report.tokens_used:,}")

    if report.error_message:
        st.error(f"错误信息: {report.error_message}")


def render_db_stats():
    """渲染数据库统计信息。"""
    vector_store = st.session_state.orchestrator.get_vector_store()
    stats = vector_store.get_stats()

    if stats["total_experiences"] == 0:
        st.info("📭 经历条目库为空，请先运行精炼管道处理 Obsidian 笔记。")
        return

    st.subheader("📚 经历条目库")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("经历条目总数", stats["total_experiences"])
    with col2:
        st.metric("独立项目数", stats["unique_projects"])
    with col3:
        st.metric("来源文件数", stats["unique_source_files"])

    # 技能分布
    if stats["top_skills"]:
        st.subheader("🏷️ 技能标签分布（Top 20）")
        skill_data = {skill: count for skill, count in stats["top_skills"]}
        st.bar_chart(skill_data)

    # 浏览条目
    with st.expander("📋 浏览所有经历条目", expanded=False):
        all_entries = vector_store.get_all_entries()
        for i, entry in enumerate(all_entries, 1):
            meta = entry.get("metadata", {})
            st.markdown(f"**{i}. {meta.get('project_name', '未命名')}**")
            st.markdown(f"> {entry.get('document', '')}")
            try:
                tags = json.loads(meta.get("skill_tags", "[]"))
                if tags:
                    st.caption(f"技能标签: {', '.join(tags)} | 来源: {meta.get('source_file', '')}")
            except json.JSONDecodeError:
                pass
            st.divider()


# ============================================================
# Tab 2: 岗位匹配
# ============================================================
def render_matching_tab():
    """渲染岗位匹配 Tab。"""
    st.header("岗位匹配")
    st.markdown("粘贴目标岗位的 JD，系统自动匹配最相关的经历并生成优化简历内容。")

    # 检查经历库是否为空
    vector_store = st.session_state.orchestrator.get_vector_store()
    if vector_store.get_stats()["total_experiences"] == 0:
        st.warning("⚠️ 经历条目库为空，请先在「内容库管理」Tab 中运行精炼管道。")
        return

    jd_content = st.text_area(
        "粘贴招聘 JD",
        height=300,
        placeholder="在此粘贴完整的招聘职位描述...",
    )

    if st.button("🎯 一键匹配", type="primary", use_container_width=True):
        if not jd_content or not jd_content.strip():
            st.error("请先粘贴 JD 内容")
            return

        with st.spinner("正在分析 JD 并匹配经历..."):
            try:
                # Step 1: 分析 JD
                jd_analyzer = JDAnalyzer()
                jd_requirements = jd_analyzer.analyze(jd_content)

                st.info(
                    f"📋 JD 分析完成: **{jd_requirements.position}** | "
                    f"必须技能: {', '.join(jd_requirements.required_skills[:5])} | "
                    f"关键词: {len(jd_requirements.match_keywords)} 个"
                )

                # Step 2: 语义检索
                retriever = Retriever(vector_store)
                retrieval_results = retriever.retrieve(jd_requirements)

                # Step 3: 生成简历内容
                # 读取简历内容（如果之前解析过）
                resume_content = st.session_state.get("resume_content", "")
                generator = ResumeGenerator(resume_content=resume_content)
                match_result = generator.generate(jd_requirements, retrieval_results)

                st.session_state.match_result = match_result

                # 保存到历史记录
                st.session_state.match_history.append({
                    "jd_position": match_result.jd_position,
                    "jd_content": jd_content[:200] + "...",
                    "result": match_result.to_dict(),
                })

            except Exception as e:
                st.error(f"❌ 匹配失败: {e}")
                logger.error("匹配执行异常: %s", e, exc_info=True)

    # 显示匹配结果
    match_result = st.session_state.match_result
    if match_result:
        st.divider()
        st.markdown(match_result.to_display_text())

        # 导出按钮
        col1, col2 = st.columns(2)
        with col1:
            result_json = json.dumps(match_result.to_dict(), ensure_ascii=False, indent=2)
            st.download_button(
                "📥 下载匹配结果 (JSON)",
                data=result_json,
                file_name=f"match_{match_result.jd_position}.json",
                mime="application/json",
            )
        with col2:
            display_text = match_result.to_display_text()
            st.download_button(
                "📥 下载匹配结果 (Markdown)",
                data=display_text,
                file_name=f"match_{match_result.jd_position}.md",
                mime="text/markdown",
            )

    # 历史记录
    if st.session_state.match_history:
        with st.expander("📜 匹配历史", expanded=False):
            for i, record in enumerate(reversed(st.session_state.match_history), 1):
                st.markdown(f"**{i}. {record['jd_position']}**")
                st.caption(record["jd_content"])
                st.divider()


# ============================================================
# Tab 3: 系统测试
# ============================================================
def render_test_tab():
    """渲染系统测试 Tab。"""
    st.header("系统测试")
    st.markdown("验证系统是否正常工作，检查经历条目库的质量。")

    vector_store = st.session_state.orchestrator.get_vector_store()
    stats = vector_store.get_stats()

    # 数据库状态
    st.subheader("📊 数据库状态")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("经历条目", stats["total_experiences"])
    with col2:
        st.metric("技能索引", stats["total_skill_entries"])
    with col3:
        st.metric("独立项目", stats["unique_projects"])
    with col4:
        st.metric("来源文件", stats["unique_source_files"])

    # 自由文本检索测试
    st.subheader("🔍 自由文本检索测试")
    st.markdown("输入任意文本，测试向量检索是否返回相关结果。")

    test_query = st.text_input(
        "测试查询",
        placeholder="例如: 微服务架构设计经验",
    )

    if st.button("执行检索测试") and test_query:
        results = vector_store.search_by_text(test_query, top_k=5)
        if results:
            st.success(f"检索到 {len(results)} 条结果")
            for i, result in enumerate(results, 1):
                meta = result.get("metadata", {})
                distance = result.get("distance", 0)
                st.markdown(f"**{i}. {meta.get('project_name', '未命名')}** (距离: {distance:.4f})")
                st.markdown(f"> {result.get('document', '')}")
                try:
                    tags = json.loads(meta.get("skill_tags", "[]"))
                    if tags:
                        st.caption(f"标签: {', '.join(tags)}")
                except json.JSONDecodeError:
                    pass
                st.divider()
        else:
            st.warning("未检索到结果，请确认经历条目库不为空。")

    # JD 匹配端到端测试
    st.subheader("🧪 端到端匹配测试")
    st.markdown("粘贴一段已知应该匹配的 JD，验证系统的完整匹配流程。")

    test_jd = st.text_area(
        "测试 JD",
        height=200,
        placeholder="粘贴一个你确信应该匹配到内容库中某些经历的 JD...",
        key="test_jd",
    )

    if st.button("执行端到端测试") and test_jd:
        with st.spinner("正在执行完整匹配流程..."):
            try:
                jd_analyzer = JDAnalyzer()
                jd_req = jd_analyzer.analyze(test_jd)
                st.json(jd_req.to_dict())

                retriever = Retriever(vector_store)
                results = retriever.retrieve(jd_req)

                st.markdown(f"**检索到 {len(results)} 条候选经历**")
                for r in results:
                    meta = r.get("metadata", {})
                    st.markdown(
                        f"- **{meta.get('project_name', '?')}** "
                        f"(得分: {r.get('score', 0):.4f}, 类型: {r.get('match_type', '?')})"
                    )

                resume_content = st.session_state.get("resume_content", "")
                generator = ResumeGenerator(resume_content=resume_content)
                match_result = generator.generate(jd_req, results)
                st.markdown(match_result.to_display_text())

                st.success("✅ 端到端测试通过")
            except Exception as e:
                st.error(f"❌ 测试失败: {e}")

    # 导出经历条目
    st.subheader("📥 数据导出")
    exported_file = STORAGE_DIR / "extracted_entries.json"
    if exported_file.exists():
        with open(exported_file, "r", encoding="utf-8") as f:
            data = f.read()
        st.download_button(
            "下载经历条目库 (JSON)",
            data=data,
            file_name="extracted_entries.json",
            mime="application/json",
        )
    else:
        st.info("暂无导出数据，请先运行精炼管道。")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    main()
