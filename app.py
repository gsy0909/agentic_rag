import asyncio
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.core.engine import OmniSearch
from src.io.data_loader import DataLoader

st.set_page_config(
    page_title="MRAG — Agentic RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── helpers ────────────────────────────

def _build_config(dataset_path, storage_path, api_key,
                  max_turns, w_bm25, w_vector, w_entity, w_ontology,
                  ontology_enabled, comp_enabled, comp_method, comp_threshold):
    return {
        "model": {
            "llm_name": "qwen3.5-flash",
            "embedding_name": "/root/autodl-tmp/LightRAG/models/Qwen3-Embedding-0.6B",
            "embedding_dim": 1024,
            "base_url": "https://www.packyapi.com/v1",
            "api_key": api_key,
            "api_mode": "auto",
        },
        "storage": {"path": storage_path, "force_rebuild_ontology": False},
        "search": {
            "max_turns": max_turns,
            "top_k": 100,
            "rrf_k": 60,
            "concept_bonus": 0.0025,
            "weights": {
                "bm25": w_bm25,
                "vector": w_vector,
                "entity": w_entity,
                "ontology": w_ontology,
            },
            "ontology_enabled": ontology_enabled,
            "compression": {
                "enabled": comp_enabled,
                "hard_method": comp_method,
                "similarity_threshold": comp_threshold,
                "compression_rate": 0.5,
                "skip_short_docs": 150,
            },
            "faiss": {"hnsw_m": 32, "ef_construction": 200, "ef_search": 64},
            "spacy": {"model": "en_core_web_lg"},
            "ontology": {},
            "llm_concurrency": 100,
            "subquery_concurrency": 4,
            "embedding_concurrency": 1,
        },
        "resources": {"global_llm_concurrency": 500, "global_gpu_concurrency": 1},
        "dataset": {"path": dataset_path, "name": "ui"},
    }


def _run_async(coro):
    """Run an async coroutine from Streamlit's sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
    except RuntimeError:
        pass
    return asyncio.run(coro)


def _display_result(item: dict):
    query = item["query"]
    result = item["result"]

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # Final answer at the top so users see it immediately
        st.markdown(result.get("answer", "—"))
        if result.get("thought"):
            with st.expander("💭 思考过程"):
                st.caption(result["thought"])

        turn_details = result.get("turn_details", [])
        if turn_details:
            total_turns = result.get("turns", len(turn_details))
            with st.expander(f"🔬 检索过程详情（共 {total_turns} 轮）", expanded=False):
                for turn in turn_details:
                    st.subheader(f"🔄 Turn {turn['turn']} / {total_turns}")

                    # Query plan
                    plan = turn.get("plan", {})
                    if plan:
                        st.markdown("**📋 查询计划**")
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            st.info(f"**Intent:** {plan.get('intent', '—')}")
                        with col_b:
                            st.info(f"**Rewritten:** {plan.get('rewritten_query', '—')}")
                        sqs = plan.get("sub_queries", [])
                        if sqs:
                            st.markdown("**Sub-queries:**")
                            for sq in sqs:
                                st.code(sq, language=None)

                    # Per sub-query detail
                    for sq_data in turn.get("sub_queries", []):
                        sub_q_label = sq_data["sub_query"]
                        with st.expander(f"📌 {sub_q_label[:120]}"):
                            # Retrieval counts
                            counts = sq_data.get("retrieval_counts", {})
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("BM25", counts.get("bm25", 0))
                            c2.metric("Vector", counts.get("vector", 0))
                            c3.metric("Entity", counts.get("entity", 0))
                            c4.metric("Ontology", counts.get("ontology", 0))

                            # Top ranked docs
                            ranked = sq_data.get("ranked_top10", [])
                            if ranked:
                                st.markdown("**📊 Top 排序结果**")
                                for doc in ranked[:5]:
                                    preview = doc.get("text_preview", "")[:150]
                                    st.markdown(
                                        f"`{doc['id']}` &nbsp; score: `{doc['score']}` &nbsp; — {preview}…"
                                    )

                            # Compression stats
                            comp_stats = sq_data.get("compression_stats", [])
                            if comp_stats:
                                orig_total = sum(s["orig_words"] for s in comp_stats)
                                comp_total = sum(s["comp_words"] for s in comp_stats)
                                ratio = comp_total / orig_total if orig_total > 0 else 0
                                st.markdown(
                                    f"**🗜️ 压缩** — {len(comp_stats)} 篇文档: "
                                    f"{orig_total} → {comp_total} 词 （保留 {ratio:.0%}）"
                                )

                            # Evidence chain
                            verif = sq_data.get("verifier_result", {})
                            evidences = verif.get("evidences_chain", [])
                            if evidences:
                                st.markdown("**✅ 证据链**")
                                for ev in evidences:
                                    st.success(ev)
                            else:
                                st.warning("未提取到有效证据")

                            covered = verif.get("sub_query_covered", False)
                            st.markdown(
                                "覆盖状态: " + ("✅ 已覆盖" if covered else "⚠️ 未完全覆盖")
                            )

                    # Reflection
                    refl = turn.get("reflection", {})
                    if refl:
                        st.markdown("**🤔 Reflection**")
                        if refl.get("thought"):
                            st.caption(refl["thought"])
                        if refl.get("answered"):
                            st.success("✅ 本轮已获得答案")
                        elif refl.get("new_query"):
                            st.warning(f"↩️ 继续搜索: {refl['new_query']}")

                    st.divider()


# ──────────────────────────── sidebar ────────────────────────────

with st.sidebar:
    st.title("⚙️ 配置")

    st.subheader("📁 数据集 & 存储")
    dataset_path = st.text_input(
        "Dataset 目录",
        value=st.session_state.get("cfg_dataset", ""),
        placeholder="e.g. D:/data/full",
    )
    storage_path = st.text_input(
        "Index 存储目录",
        value=st.session_state.get("cfg_storage", ""),
        placeholder="e.g. D:/storage/full",
    )

    st.subheader("🔑 API")
    api_key = st.text_input(
        "QWEN_API_KEY",
        value=os.environ.get("QWEN_API_KEY", ""),
        type="password",
    )

    st.subheader("🔍 检索参数")
    max_turns = st.slider("Max Turns", 1, 10, 8)

    st.subheader("⚖️ 检索权重")
    w_bm25   = st.slider("BM25",    0.0, 2.0, 0.25, 0.05)
    w_vector = st.slider("Vector",  0.0, 2.0, 1.00, 0.05)
    w_entity = st.slider("Entity",  0.0, 2.0, 0.10, 0.05)

    st.subheader("🌳 本体树")
    ontology_enabled = st.toggle("启用 Ontology", value=False)
    w_ontology = st.slider(
        "Ontology Weight", 0.0, 2.0, 0.0, 0.05,
        disabled=not ontology_enabled,
    )

    st.subheader("🗜️ 压缩模块")
    comp_enabled   = st.toggle("启用压缩", value=True)
    comp_method    = st.selectbox("方法", ["embedding", "llmlingua"], disabled=not comp_enabled)
    comp_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.3, 0.05, disabled=not comp_enabled)

    st.divider()

    init_btn   = st.button("🚀 加载/重建索引", type="primary", use_container_width=True)
    apply_btn  = st.button("✅ 应用参数变更", use_container_width=True)

    if st.session_state.get("initialized"):
        st.success("引擎已就绪")
    else:
        st.info("请先加载索引")

# ──────────────────────── init / apply ───────────────────────────

if init_btn:
    if not dataset_path or not storage_path or not api_key:
        st.sidebar.error("请填写 Dataset 目录、存储目录和 API Key")
    else:
        with st.spinner("初始化引擎并加载索引（首次可能较慢）…"):
            try:
                cfg = _build_config(
                    dataset_path, storage_path, api_key,
                    max_turns, w_bm25, w_vector, w_entity, w_ontology,
                    ontology_enabled, comp_enabled, comp_method, comp_threshold,
                )
                engine = OmniSearch(cfg)
                corpus = DataLoader(dataset_path).load_corpus()
                _run_async(engine.build_indices(corpus))

                st.session_state.engine      = engine
                st.session_state.initialized = True
                st.session_state.history     = []
                st.session_state.cfg_dataset = dataset_path
                st.session_state.cfg_storage = storage_path
                st.sidebar.success("✅ 索引加载完成")
            except Exception as e:
                st.sidebar.error(f"初始化失败：{e}")

if apply_btn and st.session_state.get("initialized"):
    engine: OmniSearch = st.session_state.engine
    engine.update_search_params(
        max_turns=max_turns,
        weights={"bm25": w_bm25, "vector": w_vector, "entity": w_entity, "ontology": w_ontology},
        ontology_enabled=ontology_enabled,
        compression_enabled=comp_enabled,
    )
    st.sidebar.success("参数已更新")

# ──────────────────────────── main UI ────────────────────────────

st.title("🔍 MRAG — Agentic RAG")

if not st.session_state.get("initialized"):
    st.info("请在左侧填写配置后点击「加载/重建索引」。")
    st.stop()

# Render chat history
for item in st.session_state.get("history", []):
    _display_result(item)

# Chat input
user_query = st.chat_input("输入问题…")
if user_query:
    engine: OmniSearch = st.session_state.engine
    with st.status("🔍 检索中…", expanded=True) as status:
        st.write("正在执行多轮 Agentic 检索…")
        try:
            result = _run_async(engine.search(user_query, trace_mode=True))
            status.update(label="✅ 检索完成", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ 检索失败", state="error")
            st.error(str(e))
            st.stop()

    st.session_state.history.append({"query": user_query, "result": result})
    st.rerun()
