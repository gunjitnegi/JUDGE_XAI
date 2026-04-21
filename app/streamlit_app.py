"""
JUDGE X AI — Streamlit Frontend
Legal Judgment Q&A with RAG + Explainable AI
"""

import os
import sys
import time
import json
import streamlit as st

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.rag.rag_pipeline import RAGPipeline

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="JUDGE X AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — FORCED DARK MODE
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── FORCE dark background on everything ── */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main, .main .block-container,
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] {
    background-color: #0a0e17 !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar dark ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0f1629 0%, #131b30 100%) !important;
    color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header {visibility: hidden !important;}

/* ── Main container ── */
.main .block-container {
    padding: 2rem 3rem !important;
    max-width: 1200px !important;
}

/* ── ALL text and markdown white ── */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
.stMarkdown strong, .stMarkdown em,
[data-testid="stText"], [data-testid="stCaptionContainer"],
.stAlert p {
    color: #e2e8f0 !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: #f1f5f9 !important;
}

/* ── Expanders forced dark ── */
[data-testid="stExpander"],
[data-testid="stExpander"] > details,
[data-testid="stExpander"] > details > summary,
[data-testid="stExpander"] > details > div {
    background-color: #111827 !important;
    color: #e2e8f0 !important;
    border-color: #1e293b !important;
}
[data-testid="stExpander"] > details > summary {
    color: #c7d2fe !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] > details > summary:hover {
    color: #818cf8 !important;
}
details[open] > summary {
    border-bottom: 1px solid #1e293b !important;
}

/* ── Hero title ── */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #a78bfa, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .2rem;
    letter-spacing: -1px;
}
.hero-subtitle {
    color: #94a3b8 !important;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(26, 34, 54, .85) !important;
    backdrop-filter: blur(16px);
    border: 1px solid rgba(99,102,241,.22);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    color: #e2e8f0 !important;
    transition: border-color .25s ease, box-shadow .25s ease;
}
.glass-card:hover {
    border-color: rgba(99,102,241,.4);
    box-shadow: 0 0 24px rgba(99,102,241,.2);
}
.glass-card * { color: #e2e8f0 !important; }

/* ── Answer card ── */
.answer-card {
    background: linear-gradient(135deg, rgba(16,185,129,.12), rgba(99,102,241,.12)) !important;
    border: 1px solid rgba(16,185,129,.3);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin: 1rem 0;
}
.answer-card p {
    color: #f1f5f9 !important;
    line-height: 1.85;
    font-size: .97rem;
}

/* ── XAI header ── */
.xai-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #818cf8 !important;
    margin-bottom: .6rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}

/* ── Source chip ── */
.source-chip {
    display: inline-block;
    background: rgba(99,102,241,.15) !important;
    border: 1px solid rgba(99,102,241,.35);
    border-radius: 8px;
    padding: .35rem .75rem;
    font-size: .82rem;
    color: #a5b4fc !important;
    margin: .2rem .25rem .2rem 0;
    font-weight: 500;
}

/* ── Sim bar ── */
.sim-bar-bg {
    background: rgba(255,255,255,.08) !important;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}
.sim-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width .6s ease;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    padding: .3rem .7rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 600;
}
.status-ready {
    background: rgba(16,185,129,.15) !important;
    color: #34d399 !important;
    border: 1px solid rgba(16,185,129,.35);
}
.status-empty {
    background: rgba(245,158,11,.12) !important;
    color: #fbbf24 !important;
    border: 1px solid rgba(245,158,11,.3);
}

/* -- Keyword pill -- */
.kw-pill {
    display: inline-block;
    background: rgba(59,130,246,.15) !important;
    color: #60a5fa !important;
    border: 1px solid rgba(59,130,246,.3);
    border-radius: 6px;
    padding: .2rem .55rem;
    font-size: .78rem;
    margin: .15rem .2rem .15rem 0;
    font-weight: 500;
}

/* -- Statute badges -- */
.statute-card {
    background: rgba(26, 34, 54, .85) !important;
    border: 1px solid rgba(99,102,241,.18);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: .8rem;
}
.statute-card * { color: #e2e8f0 !important; }
.statute-badge-ipc {
    display: inline-block;
    background: rgba(239,68,68,.15) !important;
    color: #f87171 !important;
    border: 1px solid rgba(239,68,68,.3);
    border-radius: 6px;
    padding: .2rem .6rem;
    font-size: .8rem;
    font-weight: 600;
}
.statute-badge-bns {
    display: inline-block;
    background: rgba(59,130,246,.15) !important;
    color: #60a5fa !important;
    border: 1px solid rgba(59,130,246,.3);
    border-radius: 6px;
    padding: .2rem .6rem;
    font-size: .8rem;
    font-weight: 600;
}
.statute-badge-const {
    display: inline-block;
    background: rgba(168,85,247,.15) !important;
    color: #c084fc !important;
    border: 1px solid rgba(168,85,247,.3);
    border-radius: 6px;
    padding: .2rem .6rem;
    font-size: .8rem;
    font-weight: 600;
}
.statute-badge-unknown {
    display: inline-block;
    background: rgba(100,116,139,.15) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(100,116,139,.3);
    border-radius: 6px;
    padding: .2rem .6rem;
    font-size: .8rem;
    font-weight: 600;
}
.punishment-box {
    background: rgba(239,68,68,.08) !important;
    border: 1px solid rgba(239,68,68,.2);
    border-radius: 8px;
    padding: .5rem .8rem;
    margin-top: .4rem;
    font-size: .85rem;
    color: #fca5a5 !important;
}
.bns-equiv-box {
    background: rgba(59,130,246,.08) !important;
    border: 1px solid rgba(59,130,246,.2);
    border-radius: 8px;
    padding: .4rem .7rem;
    margin-top: .3rem;
    font-size: .82rem;
    color: #93c5fd !important;
}

/* ── Trace step ── */
.trace-step {
    background: rgba(255,255,255,.04) !important;
    border-left: 3px solid #6366f1;
    padding: .8rem 1rem;
    margin-bottom: .5rem;
    border-radius: 0 8px 8px 0;
    font-size: .88rem;
    color: #94a3b8 !important;
}
.trace-step strong { color: #f1f5f9 !important; }
.trace-step em { color: #cbd5e1 !important; }

/* ── Stat card ── */
.stat-card {
    background: rgba(26, 34, 54, .7) !important;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #818cf8 !important;
}
.stat-label {
    font-size: .8rem;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: .5px;
}

/* ── Input field forced dark ── */
.stTextInput > div > div > input,
.stTextInput input {
    background: #1a2236 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    padding: .8rem 1rem !important;
    font-size: .95rem !important;
}
.stTextInput > div > div > input:focus,
.stTextInput input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.25) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #64748b !important;
}

/* ── Primary button ── */
button[kind="primary"],
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: .6rem 1.5rem !important;
}
button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(99,102,241,.3) !important;
}

/* ── Slider forced dark ── */
.stSlider label, .stSlider p { color: #cbd5e1 !important; }

/* ── File uploader dark ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div {
    color: #cbd5e1 !important;
}

/* ── Separator dark ── */
/* ── Summary Cards ── */
.summary-card {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    margin-bottom: 1rem !important;
    backdrop-filter: blur(10px);
}
.summary-section-title {
    color: #6366f1 !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-bottom: 0.5rem !important;
}
.summary-text {
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

hr { border-color: #1e293b !important; }

/* ── JSON viewer dark ── */
.stJson { background: #111827 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
    st.session_state.index_loaded = False
    st.session_state.chunks_count = 0
    st.session_state.history = []
    st.session_state.pdf_name = None

pipeline: RAGPipeline = st.session_state.pipeline

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
INDEX_DIR = r"c:\final_year\JUDGEXAI\data\faiss_index"

with st.sidebar:
    st.markdown("## ⚖️ JUDGE X AI")
    st.markdown("---")

    # Index status
    if st.session_state.index_loaded:
        st.markdown(
            f'<span class="status-badge status-ready">● Index Ready</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"**{st.session_state.chunks_count}** vectors in FAISS index")
        if st.session_state.pdf_name:
            st.caption(f"📄 {st.session_state.pdf_name}")
    else:
        st.markdown(
            '<span class="status-badge status-empty">○ No Index Loaded</span>',
            unsafe_allow_html=True,
        )
        st.caption("Upload a PDF to get started")

    st.markdown("---")

    # PDF Upload
    st.markdown("### 📄 Upload Judgment PDF")
    uploaded_file = st.file_uploader(
        "Upload a legal judgment PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    if uploaded_file is not None:
        if st.button("⚡ Process & Index", type="primary", use_container_width=True, key="process_btn"):
            with st.spinner("Processing PDF..."):
                temp_dir = os.path.join(PROJECT_ROOT, "data", "temp_uploads")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                progress = st.progress(0, text="Extracting & chunking PDF...")
                try:
                    progress.progress(10, text="Extracting text blocks from PDF...")
                    num_chunks = pipeline.ingest_pdf(temp_path, save=True)
                    progress.progress(90, text="Saving FAISS index...")

                    st.session_state.index_loaded = True
                    st.session_state.chunks_count = pipeline.vector_store.index.ntotal
                    st.session_state.pdf_name = uploaded_file.name
                    progress.progress(100, text="Done!")
                    st.success(f"✅ Indexed **{num_chunks}** chunks")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")

                if os.path.exists(temp_path):
                    os.remove(temp_path)

    # Load existing index button (instead of auto-load)
    if not st.session_state.index_loaded and os.path.exists(os.path.join(INDEX_DIR, "legal_index.faiss")):
        st.markdown("---")
        st.caption("A previously saved index was found.")
        if st.button("📂 Load Existing Index", use_container_width=True, key="load_idx_btn"):
            try:
                pipeline.load_index(INDEX_DIR)
                st.session_state.index_loaded = True
                st.session_state.chunks_count = pipeline.vector_store.index.ntotal
                # Read the actual source PDF name from saved metadata
                st.session_state.pdf_name = pipeline.vector_store.source_file or "Unknown document"
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load: {e}")

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top-K)", min_value=1, max_value=10, value=5, key="top_k_slider")

    st.markdown("---")

    # History
    if st.session_state.history:
        st.markdown("### 🕓 Recent Queries")
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            if st.button(f"🔹 {h['question'][:40]}...", key=f"hist_{i}", use_container_width=True):
                st.session_state["query_input"] = h["question"]
                st.rerun()

# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
st.markdown('<div class="hero-title">⚖️ JUDGE X AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">AI-Powered Legal Judgment Analysis · RAG + Explainable AI</div>',
    unsafe_allow_html=True,
)

# Stats row
if st.session_state.index_loaded:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{st.session_state.chunks_count}</div>'
            f'<div class="stat-label">Indexed Chunks</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="stat-card"><div class="stat-value">768</div>'
            '<div class="stat-label">Embedding Dim</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{top_k}</div>'
            f'<div class="stat-label">Top-K Retrieval</div></div>',
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────
# Judgment Summary Section
# ──────────────────────────────────────────────
if st.session_state.index_loaded:
    with st.container():
        st.markdown("### 📋 Judgment Summary")
        
        if pipeline.current_summary:
            summary = pipeline.current_summary
            st.markdown(f"**Case Title:** {summary.get('case_title', 'Unknown')}")
            
            # Statutes chips
            statutes = summary.get("statutes_cited", [])
            if statutes:
                st.markdown('<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">', unsafe_allow_html=True)
                for s in statutes:
                    badge_class = "statute-badge badge-const" if "ART" in s.upper() else "statute-badge badge-ipc" if "IPC" in s.upper() else "statute-badge badge-bns"
                    st.markdown(f'<span class="{badge_class}">{s}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("")
            
            # Section-wise summary cards
            sections = summary.get("sections", {})
            for sec_name, sec_content in sections.items():
                if sec_content and "No content found" not in sec_content:
                    with st.expander(f"🔹 {sec_name.title()}", expanded=(sec_name == "FACTS")):
                        st.markdown(f"""
                        <div class="summary-card">
                            <div class="summary-section-title">{sec_name}</div>
                            <div class="summary-text">{sec_content}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            col_gen, _ = st.columns([1, 2])
            if col_gen.button("🪄 Generate Structured Summary", use_container_width=True):
                with st.spinner("Analyzing judgment structure and generating summary..."):
                    pipeline.generate_summary()
                    st.rerun()

st.markdown("")

# ──────────────────────────────────────────────
# Query input
# ──────────────────────────────────────────────
query = st.text_input(
    "Ask a question about the judgment",
    placeholder="e.g. What IPC sections were applied? What punishment was given?",
    key="query_input",
    label_visibility="collapsed",
)

col_btn, col_space = st.columns([1, 3])
with col_btn:
    ask_clicked = st.button("🔍 Ask JUDGE X", type="primary", use_container_width=True, key="ask_btn")

# ──────────────────────────────────────────────
# Process query
# ──────────────────────────────────────────────
if ask_clicked and query:
    if not st.session_state.index_loaded:
        st.warning("⚠️ Please upload and index a PDF first using the sidebar.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = pipeline.query(query, top_k=top_k)
            except Exception as e:
                st.error(f"❌ Pipeline error: {e}")
                st.stop()

        # Save to history
        st.session_state.history.append({"question": query, "result": result})

        # ── Answer ──
        st.markdown("### 💡 Answer")
        st.markdown(
            f'<div class="answer-card"><p>{result["answer"]}</p></div>',
            unsafe_allow_html=True,
        )

        # ── XAI Section ──
        st.markdown("---")

        # ── 1. Retrieval Trace ──
        with st.expander("🔬 Why this answer? — Retrieval Trace", expanded=False):
            st.markdown(
                '<div class="xai-header">📍 Pipeline Trace</div>',
                unsafe_allow_html=True,
            )
            st.caption("Every step the query went through before the answer was generated.")

            for step in result["xai"]["retrieval_trace"]:
                step_html = f'<div class="trace-step"><strong>Step {step["step"]}: {step["action"]}</strong><br>'
                if "input" in step:
                    step_html += f'Input: <em>{step["input"]}</em><br>'
                    step_html += f'Output: <em>{step["output"][:250]}</em>'
                if "detail" in step:
                    step_html += f'{step["detail"]}'
                step_html += "</div>"
                st.markdown(step_html, unsafe_allow_html=True)

        # ── 2. Source Attribution ──
        with st.expander("📚 Source Attribution — Retrieved Chunks", expanded=True):
            st.markdown(
                '<div class="xai-header">🎯 Ranked Sources</div>',
                unsafe_allow_html=True,
            )

            for r in result["retrieved_chunks"]:
                chunk = r["chunk"]
                meta = chunk["metadata"]
                sim = r["similarity_pct"]
                kw = r.get("keyword_overlap", {})

                if sim >= 70:
                    bar_color = "linear-gradient(90deg, #10b981, #34d399)"
                    score_color = "#34d399"
                elif sim >= 50:
                    bar_color = "linear-gradient(90deg, #f59e0b, #fbbf24)"
                    score_color = "#fbbf24"
                else:
                    bar_color = "linear-gradient(90deg, #ef4444, #f87171)"
                    score_color = "#f87171"

                st.markdown(f"""
                <div class="glass-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:.6rem;">
                        <div>
                            <span class="source-chip">Rank #{r['rank']}</span>
                            <span class="source-chip">📄 Page {meta['page_number']}</span>
                            <span class="source-chip">📂 {meta['section']}</span>
                        </div>
                        <div style="font-size:.95rem; font-weight:700; color:{score_color} !important;">
                            {sim:.1f}%
                        </div>
                    </div>
                    <div class="sim-bar-bg">
                        <div class="sim-bar-fill" style="width:{min(sim,100):.1f}%; background:{bar_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Statutes
                if meta.get("statutes_mentioned"):
                    statute_html = "".join(
                        [f'<span class="source-chip">{s}</span>' for s in meta["statutes_mentioned"]]
                    )
                    st.markdown(f"**Statutes:** {statute_html}", unsafe_allow_html=True)

                # Keyword overlap
                if kw.get("matching_keywords"):
                    kw_html = "".join(
                        [f'<span class="kw-pill">{k}</span>' for k in kw["matching_keywords"]]
                    )
                    st.markdown(
                        f"**Keyword Matches** ({kw.get('overlap_count', 0)}/{len(kw.get('query_keywords', []))}): {kw_html}",
                        unsafe_allow_html=True,
                    )

                # Chunk text
                with st.expander(f"📝 Full chunk text (Rank #{r['rank']})", expanded=False):
                    st.markdown(chunk["text"])

                st.markdown("")

        # -- 3. Statutes Referenced --
        statute_data = result["xai"].get("statute_analysis", [])
        if statute_data:
            with st.expander(f"📜 Statutes Referenced ({len(statute_data)} found)", expanded=True):
                st.markdown(
                    '<div class="xai-header">⚖️ Statute Definitions & Punishments</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Statutes detected in the retrieved judgment chunks, with definitions and BNS cross-references.")

                for sd in statute_data:
                    law = sd["law"]
                    section = sd["section"]
                    title = sd.get("title", "")
                    punishment = sd.get("punishment")
                    simple = sd.get("simple_explanation")
                    bns_eq = sd.get("bns_equivalent")
                    ipc_eq = sd.get("ipc_equivalent")
                    pages = sd.get("pages", [])

                    # Badge class
                    if law == "IPC":
                        badge_class = "statute-badge-ipc"
                        badge_icon = "🔴"
                    elif law == "BNS":
                        badge_class = "statute-badge-bns"
                        badge_icon = "🔵"
                    elif law == "CONSTITUTION":
                        badge_class = "statute-badge-const"
                        badge_icon = "🟣"
                    else:
                        badge_class = "statute-badge-unknown"
                        badge_icon = "⚪"

                    # Page chips
                    page_html = ""
                    if pages:
                        page_chips = "".join([f'<span class="source-chip">Page {p}</span>' for p in pages])
                        page_html = f'<div style="margin-top:.4rem;">{page_chips}</div>'

                    # Title line
                    title_html = f' — <strong>{title}</strong>' if title else ''

                    card_html = f"""
                    <div class="statute-card">
                        <div style="display:flex; align-items:center; gap:.5rem; margin-bottom:.4rem;">
                            <span class="{badge_class}">{badge_icon} {law}</span>
                            <span style="font-size:.95rem; font-weight:600; color:#f1f5f9 !important;">
                                {section}{title_html}
                            </span>
                        </div>
                    """

                    # Simple explanation
                    if simple:
                        card_html += f'<div style="color:#cbd5e1 !important; font-size:.88rem; margin-top:.3rem;">{simple[:300]}</div>'

                    # Punishment
                    if punishment:
                        card_html += f'<div class="punishment-box">⚠️ <strong>Punishment:</strong> {punishment[:250]}</div>'

                    # BNS equivalent
                    if bns_eq:
                        bns_section = bns_eq.get('section', '')
                        card_html += f'<div class="bns-equiv-box">🔄 <strong>BNS Equivalent:</strong> BNS Section {bns_section}</div>'

                    # IPC equivalent (for BNS entries)
                    if ipc_eq:
                        ipc_section = ipc_eq.get('section', '')
                        card_html += f'<div class="bns-equiv-box">🔄 <strong>IPC Equivalent:</strong> {ipc_section}</div>'

                    # Pages
                    card_html += page_html
                    card_html += '</div>'

                    st.markdown(card_html, unsafe_allow_html=True)

        # -- 4. Keyword & Semantic Overlap --
        with st.expander("🔑 Keyword & Semantic Analysis", expanded=False):
            st.markdown(
                '<div class="xai-header">🔤 Query Keyword Coverage</div>',
                unsafe_allow_html=True,
            )

            if result["retrieved_chunks"]:
                first_kw = result["retrieved_chunks"][0].get("keyword_overlap", {})
                query_kws = first_kw.get("query_keywords", [])

                if query_kws:
                    st.markdown("**Your query keywords:**")
                    kw_all_html = "".join([f'<span class="kw-pill">{k}</span>' for k in query_kws])
                    st.markdown(kw_all_html, unsafe_allow_html=True)

                    st.markdown("")
                    st.markdown("**Coverage per chunk:**")
                    for r in result["retrieved_chunks"]:
                        kw = r.get("keyword_overlap", {})
                        ratio = kw.get("overlap_ratio", 0)
                        count = kw.get("overlap_count", 0)
                        total = len(query_kws)
                        st.progress(
                            min(ratio, 1.0),
                            text=f"Rank #{r['rank']}: {count}/{total} keywords matched ({ratio*100:.0f}%)",
                        )

        # -- 5. Raw JSON --
        with st.expander("🛠️ Raw JSON Response", expanded=False):
            safe = {
                "query_info": result["query_info"],
                "answer": result["answer"],
                "xai": result["xai"],
            }
            st.json(safe)

# ──────────────────────────────────────────────
# Empty state
# ──────────────────────────────────────────────
elif not st.session_state.history:
    st.markdown("")
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">⚖️</div>
            <div style="font-size:1.2rem; font-weight:600; color:#f1f5f9 !important; margin-bottom:.5rem;">
                Upload a Legal Judgment & Ask Questions
            </div>
            <div style="color:#94a3b8 !important; font-size:.92rem; max-width:500px; margin:0 auto;">
                Upload a PDF in the sidebar, then ask any question. <br>
                JUDGE X AI will retrieve relevant passages, generate an answer, <br>
                and show you <strong style="color:#818cf8 !important;">exactly why</strong> it chose those passages.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
