"""
app.py — Main Streamlit application for Idea2PRD.

A multi-agent PRD generator with RAG support. Features:
  - Sidebar: API key input, file uploads, sample input loader
  - Main page: Product idea input form
  - Tabs: Final PRD, Retrieved Context, Agent Outputs, Evaluation
  - Export: Download PRD as markdown
"""

import streamlit as st

from config import SAMPLE_INPUTS, EVALUATION_CRITERIA, QUALITY_CHECKLIST
from rag_utils import (
    parse_uploaded_files,
    chunk_documents,
    build_vector_store,
    get_source_summary,
)
from agents import run_full_pipeline


# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Idea2PRD — AI PRD Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Typography ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header Styling ─────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1.05rem;
        opacity: 0.92;
        margin: 0;
        font-weight: 300;
    }

    /* ── Agent Status Cards ─────────────────────────────── */
    .agent-card {
        background: linear-gradient(135deg, #f8f9fe 0%, #f0f2ff 100%);
        border: 1px solid #e0e4f5;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .agent-card h4 {
        color: #4a5568;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }

    /* ── RAG Chunk Display ──────────────────────────────── */
    .rag-chunk {
        background: #fffbeb;
        border: 1px solid #fbbf24;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .rag-chunk .source-tag {
        display: inline-block;
        background: #f59e0b;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* ── Stats Row ──────────────────────────────────────── */
    .stat-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .stat-box .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-box .stat-label {
        font-size: 0.85rem;
        color: #718096;
        margin-top: 0.3rem;
    }

    /* ── Sidebar Styling ────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f36 0%, #252b48 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #cbd5e0;
    }

    /* ── File Uploader Styling ──────────────────────────── */
    [data-testid="stFileUploader"] {
        margin-top: 0.5rem;
    }
    /* Dropzone area */
    [data-testid="stFileUploader"] section {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px dashed rgba(255, 255, 255, 0.3) !important;
        padding: 1rem !important;
    }
    /* Make all text in uploader visible */
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span {
        color: #e2e8f0 !important;
    }
    /* Uploaded file item */
    [data-testid="stUploadedFile"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px !important;
        color: white !important;
        margin-top: 0.5rem !important;
    }
    [data-testid="stUploadedFile"] svg {
        fill: white !important;
        color: white !important;
    }

    /* ── Parsed Document Expander Styling (Sidebar) ─────── */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
        fill: #e2e8f0 !important;
        color: #e2e8f0 !important;
        display: block !important;
    }
    [data-testid="stSidebar"] [data-testid="stText"] {
        color: #cbd5e0 !important;
        font-family: inherit !important;
        white-space: pre-wrap !important;
        line-height: 1.5 !important;
        padding: 0.5rem !important;
        background-color: rgba(0, 0, 0, 0.2) !important;
        border-radius: 4px !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #a0aec0 !important;
    }
    /* ── Sidebar Expand Button (when closed) ────────────── */
    [data-testid="collapsedControl"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease;
        padding: 0.1rem !important;
        margin: 0.5rem !important;
    }
    [data-testid="collapsedControl"]:hover {
        background-color: #f7fafc !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: #4a5568 !important;
        color: #4a5568 !important;
    }

    /* ── Sidebar Close Button (when open) ───────────────── */
    [data-testid="stSidebar"] button[kind="header"] {
        background-color: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 6px !important;
        color: rgba(255, 255, 255, 0.7) !important;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] button[kind="header"]:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    [data-testid="stSidebar"] button[kind="header"] svg {
        display: none !important; /* Hide the Cross (X) */
    }
    [data-testid="stSidebar"] button[kind="header"]::after {
        content: "❮"; /* Good icon instead of Cross */
        font-size: 1.1rem;
        font-weight: 700;
        display: block;
    }
    /* ── Button Styling ─────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.45);
    }

    /* ── Tabs Styling ───────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    /* ── Dividers ───────────────────────────────────────── */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "parsed_docs" not in st.session_state:
    st.session_state.parsed_docs = []
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # ── LLM Provider ──────────────────────────────────────
    provider = st.selectbox(
        "LLM Provider",
        options=["Gemini", "OpenAI", "Groq"],
        index=0,
        help="Select your LLM provider. Gemini offers a generous free tier. Groq provides fast free inference.",
    )
    provider_key = provider.lower()

    # ── API Key ───────────────────────────────────────────
    api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        placeholder=f"Enter your {provider} API key...",
        help=f"Your {provider} API key. It is never stored or logged.",
    )

    st.markdown("---")

    # ── File Upload ───────────────────────────────────────
    st.markdown("## 📎 Upload Documents")
    st.markdown(
        "<small>Upload competitor notes, market research, business rules, "
        "or requirement references to enhance the PRD with RAG.</small>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload supporting documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Process uploaded files when they change
    if uploaded_files:
        # Parse files
        parsed = parse_uploaded_files(uploaded_files)
        if parsed:
            st.session_state.parsed_docs = parsed
            st.success(f"✅ Parsed {len(parsed)} document(s)")

            # Show parsed document info
            for doc in parsed:
                with st.expander(f"📄 {doc.filename}", expanded=False):
                    st.caption(f"Type: {doc.file_type.upper()} | {len(doc.content)} chars")
                    st.text(doc.content[:300] + ("..." if len(doc.content) > 300 else ""))

            # Chunk and build vector store
            chunks = chunk_documents(parsed)
            st.session_state.doc_chunks = chunks
            st.info(f"📦 Created {len(chunks)} text chunks")

            # Build vector store (requires API key)
            if api_key:
                try:
                    with st.spinner("Building vector index..."):
                        vs = build_vector_store(chunks, provider_key, api_key)
                        st.session_state.vector_store = vs
                    st.success("🔍 Vector store ready!")
                except Exception as e:
                    st.error(f"Failed to build vector store: {e}")
                    st.session_state.vector_store = None
            else:
                st.warning("Enter your API key to enable RAG retrieval.")
    else:
        st.session_state.parsed_docs = []
        st.session_state.doc_chunks = []
        st.session_state.vector_store = None

    st.markdown("---")

    # ── Sample Input Loader ───────────────────────────────
    st.markdown("## 🧪 Sample Inputs")
    st.markdown('<p style="color: white; font-size: 14px; margin-bottom: 0.25rem;">Load a sample product idea</p>', unsafe_allow_html=True)
    sample_choice = st.selectbox(
        "Load a sample product idea",
        options=["— Select —"] + [s["name"] for s in SAMPLE_INPUTS],
        index=0,
        label_visibility="collapsed"
    )

    if sample_choice != "— Select —":
        selected = next(s for s in SAMPLE_INPUTS if s["name"] == sample_choice)
        st.session_state["sample_idea"] = selected["idea"]
        st.session_state["sample_users"] = selected["target_users"]
        st.session_state["sample_domain"] = selected["domain"]
        st.session_state["sample_constraints"] = selected["constraints"]
    else:
        # Clear sample data if deselected
        for key in ["sample_idea", "sample_users", "sample_domain", "sample_constraints"]:
            if key in st.session_state:
                del st.session_state[key]


# ═══════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ═══════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📄 Idea2PRD</h1>
    <p>Transform your product idea into a comprehensive Product Requirements Document
    using multi-agent AI with RAG-powered context retrieval.</p>
</div>
""", unsafe_allow_html=True)


# ── Input Form ────────────────────────────────────────────────────────
st.markdown("### 💡 Describe Your Product Idea")

col1, col2 = st.columns([2, 1])

with col1:
    idea = st.text_area(
        "Product Idea *",
        value=st.session_state.get("sample_idea", ""),
        height=140,
        placeholder="Describe your product idea in detail. What problem does it solve? What makes it unique?",
        help="Be as specific as possible. The more detail you provide, the better the PRD.",
    )

    target_users = st.text_input(
        "Target Users *",
        value=st.session_state.get("sample_users", ""),
        placeholder="Who will use this product? (e.g., college students aged 18-25)",
    )

with col2:
    domain = st.text_input(
        "Industry / Domain",
        value=st.session_state.get("sample_domain", ""),
        placeholder="e.g., EdTech, HealthTech, FinTech",
        help="Optional. Helps the agents tailor their analysis.",
    )

    constraints = st.text_area(
        "Known Constraints",
        value=st.session_state.get("sample_constraints", ""),
        height=108,
        placeholder="Budget limits, compliance needs, tech stack preferences...",
        help="Optional. Any known limitations or requirements.",
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ── Generate Button ──────────────────────────────────────────────────
col_btn, col_info = st.columns([1, 2])

with col_btn:
    generate_clicked = st.button(
        "🚀 Generate PRD",
        type="primary",
        use_container_width=True,
        disabled=not (idea and target_users and api_key),
    )

with col_info:
    if not api_key:
        st.warning("⬅ Enter your API key in the sidebar to get started.")
    elif not idea or not target_users:
        st.info("Fill in the product idea and target users to enable generation.")
    else:
        doc_count = len(st.session_state.parsed_docs)
        if doc_count > 0:
            st.success(f"✅ Ready to generate with RAG ({doc_count} document(s) loaded)")
        else:
            st.info("💡 No documents uploaded — PRD will be based on general knowledge.")


# ── Pipeline Execution ───────────────────────────────────────────────
if generate_clicked:
    # Progress tracking
    progress_bar = st.progress(0, text="Starting multi-agent pipeline...")
    status_text = st.empty()

    def update_progress(step_text: str, progress: float):
        progress_bar.progress(progress, text=step_text)
        status_text.markdown(f"**{step_text}**")

    # Run the pipeline
    with st.spinner(""):
        result = run_full_pipeline(
            idea=idea,
            target_users=target_users,
            domain=domain,
            constraints=constraints,
            provider=provider_key,
            api_key=api_key,
            vector_store=st.session_state.vector_store,
            progress_callback=update_progress,
        )

    # Store result
    st.session_state.pipeline_result = result

    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()

    if result.success:
        st.success("✅ PRD generated successfully! Scroll down to view the results.")
    else:
        st.error(f"❌ Generation failed: {result.error}")


# ═══════════════════════════════════════════════════════════════════════
# RESULTS TABS
# ═══════════════════════════════════════════════════════════════════════
result = st.session_state.pipeline_result

if result and result.success:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Stats Row ─────────────────────────────────────────
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">3</div>
            <div class="stat-label">AI Agents Used</div>
        </div>
        """, unsafe_allow_html=True)

    with stat_cols[1]:
        total_chunks = sum(
            len(getattr(out, "rag_result", type("", (), {"chunks": []})()).chunks)
            if hasattr(out, "rag_result") else 0
            for out in [result.ba_output, result.pm_output, result.risk_output]
        )
        # Safer counting
        chunks_used = 0
        for out in [result.ba_output, result.pm_output, result.risk_output]:
            if out.rag_result and out.rag_result.chunks:
                chunks_used += len(out.rag_result.chunks)

        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{chunks_used}</div>
            <div class="stat-label">RAG Chunks Retrieved</div>
        </div>
        """, unsafe_allow_html=True)

    with stat_cols[2]:
        prd_words = len(result.final_prd.split()) if result.final_prd else 0
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{prd_words:,}</div>
            <div class="stat-label">Words in PRD</div>
        </div>
        """, unsafe_allow_html=True)

    with stat_cols[3]:
        doc_count = len(st.session_state.parsed_docs)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{doc_count}</div>
            <div class="stat-label">Source Documents</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")  # spacing

    # ── Tabs ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Final PRD",
        "🔍 Retrieved Context",
        "🤖 Agent Outputs",
        "📊 Evaluation",
    ])

    # ── Tab 1: Final PRD ──────────────────────────────────
    with tab1:
        st.markdown("### 📄 Generated Product Requirements Document")

        # Download button
        st.download_button(
            label="⬇️ Download PRD as Markdown",
            data=result.final_prd,
            file_name="product_requirements_document.md",
            mime="text/markdown",
            use_container_width=False,
        )

        st.markdown("---")

        # Display PRD
        st.markdown(result.final_prd)

    # ── Tab 2: Retrieved Context ──────────────────────────
    with tab2:
        st.markdown("### 🔍 Retrieved Context from Uploaded Documents")

        if not any(
            out.rag_result and out.rag_result.has_context
            for out in [result.ba_output, result.pm_output, result.risk_output]
        ):
            st.info(
                "🚫 No documents were uploaded, so no RAG context was retrieved. "
                "The PRD was generated using only the product idea and general knowledge. "
                "Upload documents in the sidebar to enable RAG-powered context retrieval."
            )
        else:
            # Show context per agent
            for agent_output in [result.ba_output, result.pm_output, result.risk_output]:
                if agent_output.rag_result and agent_output.rag_result.has_context:
                    st.markdown(f"#### {agent_output.agent_name} Agent — Retrieved Chunks")

                    # Source summary
                    source_summary = get_source_summary(agent_output.rag_result.chunks)
                    source_tags = " | ".join(
                        f"**{src}**: {count} chunk(s)" for src, count in source_summary.items()
                    )
                    st.markdown(f"📁 Sources: {source_tags}")

                    # Display each chunk
                    for i, chunk in enumerate(agent_output.rag_result.chunks):
                        st.markdown(
                            f'<div class="rag-chunk">'
                            f'<span class="source-tag">📄 {chunk.source}</span>'
                            f"<br>{chunk.content}"
                            f"<br><small>Relevance score: {chunk.score:.4f}</small>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")

    # ── Tab 3: Agent Outputs ──────────────────────────────
    with tab3:
        st.markdown("### 🤖 Individual Agent Outputs")
        st.markdown(
            "View the raw output from each agent in the multi-agent pipeline. "
            "These outputs were combined to produce the final PRD."
        )

        # Business Analyst
        with st.expander("🔍 Business Analyst Agent", expanded=False):
            st.markdown(
                '<div class="agent-card">'
                "<h4>Role: Extract business context, personas, and pain points</h4>"
                "</div>",
                unsafe_allow_html=True,
            )
            if result.ba_output.rag_result and result.ba_output.rag_result.has_context:
                st.caption(
                    f"📎 Used {len(result.ba_output.rag_result.chunks)} retrieved chunk(s) "
                    "from uploaded documents"
                )
            st.markdown(result.ba_output.output)

        # Product Manager
        with st.expander("📋 Product Manager Agent", expanded=False):
            st.markdown(
                '<div class="agent-card">'
                "<h4>Role: Generate requirements, user stories, and MVP scope</h4>"
                "</div>",
                unsafe_allow_html=True,
            )
            if result.pm_output.rag_result and result.pm_output.rag_result.has_context:
                st.caption(
                    f"📎 Used {len(result.pm_output.rag_result.chunks)} retrieved chunk(s) "
                    "from uploaded documents"
                )
            st.markdown(result.pm_output.output)

        # Risk Reviewer
        with st.expander("⚠️ Risk Reviewer Agent", expanded=False):
            st.markdown(
                '<div class="agent-card">'
                "<h4>Role: Identify risks, gaps, and clarifying questions</h4>"
                "</div>",
                unsafe_allow_html=True,
            )
            if result.risk_output.rag_result and result.risk_output.rag_result.has_context:
                st.caption(
                    f"📎 Used {len(result.risk_output.rag_result.chunks)} retrieved chunk(s) "
                    "from uploaded documents"
                )
            st.markdown(result.risk_output.output)

    # ── Tab 4: Evaluation ─────────────────────────────────
    with tab4:
        st.markdown("### 📊 PRD Quality Evaluation")
        st.markdown(
            "Use this section to manually evaluate the quality of the generated PRD. "
            "This evaluation framework is designed for the LLM course submission."
        )

        # Quality Checklist
        st.markdown("#### ✅ Output Quality Checklist")
        for item in QUALITY_CHECKLIST:
            st.checkbox(item, key=f"check_{item[:20]}")

        st.markdown("---")

        # Evaluation Rubric Table
        st.markdown("#### 📏 Manual Evaluation Rubric")
        st.markdown(
            "Rate the generated PRD on each criterion (1-5). "
            "This table can be included in your course report."
        )

        eval_data = {
            "Criterion": [],
            "Description": [],
            "Scale": [],
            "Your Rating": [],
        }
        for criterion in EVALUATION_CRITERIA:
            eval_data["Criterion"].append(criterion["criterion"])
            eval_data["Description"].append(criterion["description"])
            eval_data["Scale"].append(criterion["scale"])
            eval_data["Your Rating"].append("")

        st.table(eval_data)

        st.markdown("---")

        # Sample Test Cases
        st.markdown("#### 🧪 Sample Test Cases")
        st.markdown(
            "The following sample inputs are available in the sidebar for testing. "
            "Generate PRDs for all three and compare quality."
        )

        for i, sample in enumerate(SAMPLE_INPUTS):
            with st.expander(f"Test Case {i+1}: {sample['name']}", expanded=False):
                st.markdown(f"**Product Idea:** {sample['idea']}")
                st.markdown(f"**Target Users:** {sample['target_users']}")
                st.markdown(f"**Domain:** {sample['domain']}")
                st.markdown(f"**Constraints:** {sample['constraints']}")

        st.markdown("---")

        # Limitations
        st.markdown("#### ⚠️ Known Limitations")
        st.markdown("""
        - **Token limits**: Very long uploaded documents may be truncated during chunking
        - **Single LLM**: All agents use the same underlying LLM (no specialized models)
        - **No memory**: Agents don't retain context between sessions
        - **English only**: Prompts and output are optimized for English text
        - **No real-time collaboration**: Single-user, single-session workflow
        - **RAG quality**: Retrieval effectiveness depends on document quality and relevance
        """)

        st.markdown("#### 🔮 Future Improvements")
        st.markdown("""
        - Add iterative refinement (user feedback → re-generation)
        - Support more file formats (DOCX, HTML, URLs)
        - Add agent-to-agent communication for conflict resolution
        - Implement semantic caching for repeated queries
        - Add PRD comparison mode (before/after RAG)
        - Export to PDF and Google Docs
        - Fine-tune prompts per industry vertical
        """)


# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #718096; font-size: 0.85rem; padding: 1rem;'>"
    "Built with ❤️ by Parmod Kumar for the IIIT LLM Course | "
    "Powered by LangChain, FAISS, and Gemini/OpenAI/Groq"
    "</div>",
    unsafe_allow_html=True,
)
