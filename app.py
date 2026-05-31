import streamlit as st

from fullrag import (
    answer_upload_status_question,
    build_conversational_rag_chain,
    build_vectorstore,
    delete_saved_pdf,
    get_saved_groq_api_key,
    get_saved_pdf_count,
    get_saved_pdf_names,
    get_retrieved_chunks,
    get_session_history,
    load_documents_from_saved_pdfs,
    load_saved_histories,
    save_histories,
    save_uploaded_pdfs,
    stream_rag_answer,
)

# ── Page config ────────────────────────────────────────────────────────────────
def history_to_chat_display(session_history):
    display_messages = []
    for message in session_history.messages:
        if message.type == "human":
            display_messages.append({"role": "user", "content": message.content})
        elif message.type == "ai":
            display_messages.append({"role": "assistant", "content": message.content})
    return display_messages


st.set_page_config(
    page_title="DocMind AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Master CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ════════════════════════════════
   BASE RESET
════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ════════════════════════════════
   PAGE BACKGROUND
════════════════════════════════ */
.stApp {
    background: #080810;
    color: #dcdcee;
}
.stApp::before {
    content: '';
    position: fixed;
    top: -200px;
    left: 50%;
    transform: translateX(-50%);
    width: 800px;
    height: 600px;
    background: radial-gradient(ellipse at center,
        rgba(99, 78, 210, 0.08) 0%,
        transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ════════════════════════════════
   MAIN CONTAINER
════════════════════════════════ */
.block-container {
    padding: 0 2rem 8rem 2rem !important;
    max-width: 820px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}

/* ════════════════════════════════
   SIDEBAR
════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #0c0c18 !important;
    border-right: 1px solid rgba(255,255,255,0.055);
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.4rem 1.1rem 2rem;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #b0b0cc;
}
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
    font-size: 0.84rem !important;
    transition: border-color 0.25s, box-shadow 0.25s;
    padding: 0.45rem 0.75rem !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: rgba(110,90,210,0.65) !important;
    box-shadow: 0 0 0 3px rgba(110,90,210,0.14) !important;
    outline: none !important;
}
[data-testid="stSidebar"] label {
    color: #606080 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 500;
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    color: #888899 !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
    transition: all 0.22s ease;
    padding: 0.45rem 1rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(220, 60, 60, 0.1) !important;
    border-color: rgba(220, 60, 60, 0.28) !important;
    color: #f07070 !important;
    transform: translateY(-1px);
}
[data-testid="stFileUploader"] {
    background: rgba(110,90,210,0.04) !important;
    border: 1.5px dashed rgba(110,90,210,0.28) !important;
    border-radius: 14px !important;
    transition: border-color 0.25s, background 0.25s;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(110,90,210,0.08) !important;
    border-color: rgba(110,90,210,0.55) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small {
    color: #7060c8 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-size: 0.82rem !important;
}

/* ════════════════════════════════
   TOP HEADER BAR
════════════════════════════════ */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1.6rem 0 1.2rem;
    border-bottom: 1px solid rgba(255,255,255,0.055);
    margin-bottom: 1.6rem;
}
.app-header-logo {
    width: 42px; height: 42px;
    background: linear-gradient(145deg, #5a3fd0, #8b68f5);
    border-radius: 13px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 0 20px rgba(100,75,220,0.35);
    flex-shrink: 0;
}
.app-header-info { display: flex; flex-direction: column; gap: 1px; }
.app-header-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #eeeeff;
    margin: 0; letter-spacing: -0.01em;
}
.app-header-sub {
    font-size: 0.76rem;
    color: #48485e;
    margin: 0;
}
.app-header-badge {
    margin-left: auto;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(110,90,210,0.12);
    border: 1px solid rgba(110,90,210,0.25);
    color: #9080e0;
    letter-spacing: 0.04em;
}

/* ════════════════════════════════
   SIDEBAR BRAND
════════════════════════════════ */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.4rem;
}
.sb-brand-logo {
    width: 36px; height: 36px;
    background: linear-gradient(145deg, #5a3fd0, #8b68f5);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    box-shadow: 0 0 14px rgba(100,75,220,0.3);
    flex-shrink: 0;
}
.sb-brand-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: #eeeeff !important;
    line-height: 1.2;
}
.sb-brand-tag {
    font-size: 0.68rem;
    color: #48485e !important;
}
.sb-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 1rem 0;
}
.sb-label {
    font-size: 0.7rem !important;
    font-weight: 600;
    color: #44445c !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
    display: block;
}

/* PDF pill */
.pdf-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(110,90,210,0.07);
    border: 1px solid rgba(110,90,210,0.16);
    border-radius: 9px;
    padding: 7px 10px;
    margin-bottom: 5px;
    font-size: 0.79rem;
    color: #aaaacc;
    transition: background 0.2s;
    overflow: hidden;
}
.pdf-pill:hover { background: rgba(110,90,210,0.13); }
.pdf-pill-icon { font-size: 0.85rem; flex-shrink: 0; }
.pdf-pill-name {
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Indexed badge */
.indexed-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(60,190,110,0.08);
    border: 1px solid rgba(60,190,110,0.2);
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.72rem;
    color: #5cba80;
    margin: 6px 0 8px;
}

/* ════════════════════════════════
   EMPTY STATES
════════════════════════════════ */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 4.5rem 2rem 3rem;
    gap: 10px;
    animation: fadeIn 0.4s ease;
}
.empty-icon { font-size: 2.2rem; opacity: 0.28; margin-bottom: 4px; }
.empty-title { font-size: 0.98rem; font-weight: 500; color: #4a4a68; }
.empty-sub { font-size: 0.8rem; color: #333348; max-width: 320px; line-height: 1.6; }

/* ════════════════════════════════
   CHAT THREAD
════════════════════════════════ */
.chat-thread {
    display: flex;
    flex-direction: column;
    gap: 22px;
    padding: 0.4rem 0 1rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

.msg-row {
    display: flex;
    align-items: flex-end;
    gap: 11px;
    animation: fadeUp 0.32s cubic-bezier(0.22,1,0.36,1);
}
.msg-row.user  { flex-direction: row-reverse; }
.msg-row.bot   { flex-direction: row; }

.avatar {
    width: 33px; height: 33px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem;
    font-weight: 600;
    flex-shrink: 0;
    letter-spacing: -0.01em;
}
.avatar.user {
    background: linear-gradient(145deg, #5a3fd0, #8b68f5);
    color: #fff;
    box-shadow: 0 3px 14px rgba(100,75,220,0.35);
}
.avatar.bot {
    background: #141420;
    border: 1px solid rgba(110,90,210,0.3);
    color: #9078e0;
    font-size: 0.95rem;
}

.bubble-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-width: 76%;
}
.msg-row.user .bubble-wrap { align-items: flex-end; }
.msg-row.bot  .bubble-wrap { align-items: flex-start; }

.msg-label {
    font-size: 0.69rem;
    color: #3e3e58;
    letter-spacing: 0.03em;
    padding: 0 4px;
}

.bubble {
    padding: 11px 16px;
    border-radius: 18px;
    font-size: 0.895rem;
    line-height: 1.68;
    word-break: break-word;
    position: relative;
}
.bubble.user {
    background: linear-gradient(145deg, #4e38c0, #7055e0);
    color: #ede8ff;
    border-bottom-right-radius: 4px;
    box-shadow: 0 5px 22px rgba(100,75,220,0.22);
}
.bubble.bot {
    background: rgba(255,255,255,0.038);
    border: 1px solid rgba(255,255,255,0.07);
    color: #cccce0;
    border-bottom-left-radius: 4px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 4px 22px rgba(0,0,0,0.28);
}
.bubble:hover { transform: translateY(-1px); transition: transform 0.18s ease; }

/* ════════════════════════════════
   TYPING INDICATOR
════════════════════════════════ */
.typing-row {
    display: flex;
    align-items: flex-end;
    gap: 11px;
    animation: fadeIn 0.25s ease;
}
.typing-bubble {
    background: rgba(255,255,255,0.038);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    border-bottom-left-radius: 4px;
    padding: 13px 18px;
    display: flex;
    gap: 5px;
    align-items: center;
}
.dot {
    width: 7px; height: 7px;
    background: #7055e0;
    border-radius: 50%;
    animation: dotBounce 1.3s infinite ease-in-out;
}
.dot:nth-child(2) { animation-delay: 0.18s; background: #8b68f5; }
.dot:nth-child(3) { animation-delay: 0.36s; background: #a98bfa; }
@keyframes dotBounce {
    0%, 80%, 100% { transform: translateY(0) scale(0.65); opacity: 0.45; }
    40%           { transform: translateY(-6px) scale(1);   opacity: 1; }
}

/* ════════════════════════════════
   THREAD DIVIDER
════════════════════════════════ */
.thread-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 1.2rem 0 0.8rem;
}

/* ════════════════════════════════
   INPUT FORM
════════════════════════════════ */
[data-testid="stForm"] {
    background: rgba(255,255,255,0.028) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 18px !important;
    padding: 0.7rem 0.9rem !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), 0 0 0 1px rgba(110,90,210,0.07) !important;
}
[data-testid="stForm"]:focus-within {
    border-color: rgba(110,90,210,0.35) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35),
                0 0 0 1px rgba(110,90,210,0.2),
                0 0 20px rgba(110,90,210,0.06) !important;
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
}
[data-testid="stForm"] textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #dcdcee !important;
    font-size: 0.91rem !important;
    font-family: 'Inter', sans-serif !important;
    caret-color: #8b68f5;
    resize: none !important;
    line-height: 1.6;
}
[data-testid="stForm"] textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stForm"] textarea::placeholder { color: #38385a !important; }

[data-testid="stForm"] [data-testid="baseButton-primary"] {
    background: linear-gradient(145deg, #4e38c0, #7055e0) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #f0ecff !important;
    font-weight: 500 !important;
    font-size: 0.87rem !important;
    letter-spacing: 0.025em;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.22s ease !important;
    box-shadow: 0 4px 18px rgba(100,75,220,0.32) !important;
}
[data-testid="stForm"] [data-testid="baseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 26px rgba(100,75,220,0.48) !important;
    background: linear-gradient(145deg, #5840d0, #7e64f0) !important;
}
[data-testid="stForm"] [data-testid="baseButton-primary"]:active {
    transform: scale(0.97) translateY(0) !important;
    box-shadow: 0 3px 12px rgba(100,75,220,0.3) !important;
}

/* ════════════════════════════════
   STREAMLIT ALERT OVERRIDES
════════════════════════════════ */
.stAlert {
    border-radius: 12px !important;
    font-size: 0.84rem !important;
}
div[data-testid="stNotificationContentWarning"] {
    background: rgba(220,160,40,0.08) !important;
    border: 1px solid rgba(220,160,40,0.2) !important;
    border-radius: 12px;
    color: #c8a040 !important;
}
div[data-testid="stNotificationContentSuccess"] {
    background: rgba(60,190,110,0.08) !important;
    border: 1px solid rgba(60,190,110,0.2) !important;
    border-radius: 12px;
}
div[data-testid="stNotificationContentError"] {
    background: rgba(210,60,60,0.08) !important;
    border: 1px solid rgba(210,60,60,0.22) !important;
    border-radius: 12px;
}

[data-testid="stSpinner"] p { color: #55557a !important; font-size: 0.83rem !important; }

#MainMenu, footer { visibility: hidden; }
header { visibility: visible !important; background: transparent !important; }
button[data-testid="collapsedControl"],
button[kind="header"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(110,90,210,0.22); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(110,90,210,0.42); }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = load_saved_histories()
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "loaded_pdf_names" not in st.session_state:
    st.session_state.loaded_pdf_names = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_api_key" not in st.session_state:
    st.session_state.active_api_key = get_saved_groq_api_key()
if "last_retrieved_chunks" not in st.session_state:
    st.session_state.last_retrieved_chunks = []
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = "default_session"
if not st.session_state.chat_history:
    active_history = get_session_history(
        st.session_state.store,
        st.session_state.active_session_id,
    )
    st.session_state.chat_history = history_to_chat_display(active_history)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-logo">⬡</div>
        <div>
            <div class="sb-brand-name">DocMind AI</div>
            <div class="sb-brand-tag">Groq · LLaMA · RAG</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    st.markdown('<span class="sb-label">Groq API Key</span>', unsafe_allow_html=True)
    api_key = st.text_input(
        "api_key_input",
        value=st.session_state.active_api_key,
        type="password",
        placeholder="gsk_•••••••••••••••••••",
        label_visibility="collapsed",
    )
    if st.button("Use API key", type="primary", use_container_width=True):
        st.session_state.active_api_key = api_key.strip()
        st.session_state.chain = None
        st.session_state.loaded_pdf_names = set()
        st.toast("API key applied.")

    api_key = st.session_state.active_api_key

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown('<span class="sb-label">Session</span>', unsafe_allow_html=True)
    session_id = st.text_input(
        "session_id_input",
        value=st.session_state.active_session_id,
        placeholder="e.g. research-notes",
        label_visibility="collapsed",
    )
    if st.button("Use session", type="primary", use_container_width=True):
        st.session_state.active_session_id = session_id.strip() or "default_session"
        active_history = get_session_history(
            st.session_state.store,
            st.session_state.active_session_id,
        )
        st.session_state.chat_history = history_to_chat_display(active_history)
        st.session_state.last_retrieved_chunks = []
        st.toast(f"Session applied: {st.session_state.active_session_id}")

    session_id = st.session_state.active_session_id

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    st.markdown('<span class="sb-label">Documents</span>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here or browse",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="visible",
    )
    if uploaded_files:
        newly_saved = save_uploaded_pdfs(uploaded_files)
        if newly_saved:
            # FIX: invalidate vectorstore so it rebuilds with the new PDF included
            st.session_state.vectorstore = None
            st.session_state.chain = None
            st.session_state.loaded_pdf_names = set()
            st.toast(f"{len(newly_saved)} PDF(s) uploaded!", icon="✅")

    saved_pdfs = get_saved_pdf_names()
    if saved_pdfs:
        st.markdown(
            f'<div class="indexed-badge">✦ {len(saved_pdfs)} file{"s" if len(saved_pdfs) > 1 else ""} ready</div>',
            unsafe_allow_html=True,
        )
        for pdf_name in saved_pdfs:
            col1, col2 = st.columns([7, 1])
            with col1:
                short = pdf_name if len(pdf_name) <= 24 else pdf_name[:21] + "…"
                st.markdown(
                    f'<div class="pdf-pill">'
                    f'<span class="pdf-pill-icon">📄</span>'
                    f'<span class="pdf-pill-name">{short}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                if st.button("✕", key=f"del_{pdf_name}", help=f"Remove {pdf_name}"):
                    delete_saved_pdf(pdf_name)
                    st.session_state.vectorstore = None
                    st.session_state.chain = None
                    st.session_state.loaded_pdf_names = set()
                    st.rerun()

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    if st.button("⌫  Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_retrieved_chunks = []
        session_history = get_session_history(st.session_state.store, session_id)
        session_history.clear()
        save_histories(st.session_state.store)
        st.rerun()

    st.markdown(
        "<div style='margin-top:auto; padding-top:1.5rem; font-size:0.68rem;"
        " color:#2e2e45; text-align:center;'>DocMind AI · Built with Streamlit</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="app-header-logo">⬡</div>
    <div class="app-header-info">
        <p class="app-header-title">DocMind AI</p>
        <p class="app-header-sub">Ask anything about your uploaded documents</p>
    </div>
    <span class="app-header-badge">RAG · Groq</span>
</div>
""", unsafe_allow_html=True)

# ── Guard: no API key ─────────────────────────────────────────────────────────
if not api_key:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🔑</div>
        <div class="empty-title">API key required</div>
        <div class="empty-sub">Enter your Groq API key in the sidebar to unlock the chat.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Build / rebuild vectorstore only when PDFs change ────────────────────────
current_pdf_names = set(get_saved_pdf_names())
pdfs_changed = current_pdf_names != st.session_state.loaded_pdf_names

if current_pdf_names and pdfs_changed:
    with st.spinner("Indexing your documents…"):
        try:
            documents = load_documents_from_saved_pdfs()
            st.session_state.vectorstore = build_vectorstore(documents)
            st.session_state.chain = build_conversational_rag_chain(
                api_key,
                st.session_state.vectorstore,
                st.session_state.store,
            )
            st.session_state.loaded_pdf_names = current_pdf_names
            st.success(f"✦  {get_saved_pdf_count()} PDF(s) indexed and ready.")
        except Exception as exc:
            st.error(f"Failed to index PDFs: {exc}")
            st.stop()

# ── Guard: no PDFs ────────────────────────────────────────────────────────────
if not current_pdf_names:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📂</div>
        <div class="empty-title">No documents uploaded</div>
        <div class="empty-sub">Upload one or more PDFs from the sidebar to start a conversation.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not st.session_state.chain:
    st.stop()

# ── Chat thread ───────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    html = '<div class="chat-thread">'
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            html += f"""
            <div class="msg-row user">
                <div class="avatar user">U</div>
                <div class="bubble-wrap">
                    <span class="msg-label">You</span>
                    <div class="bubble user">{msg["content"]}</div>
                </div>
            </div>"""
        else:
            html += f"""
            <div class="msg-row bot">
                <div class="avatar bot">⬡</div>
                <div class="bubble-wrap">
                    <span class="msg-label">DocMind</span>
                    <div class="bubble bot">{msg["content"]}</div>
                </div>
            </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.markdown('<hr class="thread-divider">', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">💬</div>
        <div class="empty-title">Ready to answer your questions</div>
        <div class="empty-sub">Type a question below — I'll answer using the content of your uploaded documents.</div>
    </div>
    """, unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
if st.session_state.last_retrieved_chunks:
    with st.expander("Retrieved chunks debug", expanded=False):
        for chunk in st.session_state.last_retrieved_chunks:
            st.markdown(
                f"**Rank {chunk['rank']}** | Source: `{chunk['source']}` | "
                f"Page: `{chunk['page']}` | Score: `{chunk['score']:.4f}`"
            )
            st.code(chunk["content"][:1500])

with st.form("question_form", clear_on_submit=True):
    user_input = st.text_area(
        "question",
        placeholder="Ask anything about your documents…",
        label_visibility="collapsed",
        height=76,
    )
    ask_clicked = st.form_submit_button(
        "Send  ➤",
        type="primary",
        use_container_width=True,
    )

if ask_clicked:
    if not user_input.strip():
        st.warning("Please type a question before sending.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        st.session_state.last_retrieved_chunks = get_retrieved_chunks(
            st.session_state.vectorstore,
            user_input.strip(),
        )

        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div class="typing-row">
            <div class="avatar bot">⬡</div>
            <div class="typing-bubble">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            app_state_answer = answer_upload_status_question(user_input.strip())
            if app_state_answer:
                typing_placeholder.empty()
                answer = app_state_answer
                st.write(f"DocMind: {answer}")
            else:
                typing_placeholder.empty()
                st.markdown("**DocMind:**")
                streamed_chunks = []

                def answer_stream():
                    for chunk in stream_rag_answer(
                        st.session_state.chain,
                        user_input.strip(),
                        session_id,
                    ):
                        streamed_chunks.append(chunk)
                        yield chunk

                st.write_stream(answer_stream())
                answer = "".join(streamed_chunks)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            save_histories(st.session_state.store)
        except Exception as exc:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"⚠️ Something went wrong: {exc}"}
            )
        finally:
            typing_placeholder.empty()

        st.rerun()
