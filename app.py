import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFacePipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – RAG Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0f14; color: #e8e6e0; }
section[data-testid="stSidebar"] { background: #12151c !important; border-right: 1px solid #1e2330; }

.hero-title {
    font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #f0c27f 0%, #e88a4a 50%, #c94f6e 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin-bottom: 0.2rem;
}
.hero-sub { font-size:0.85rem; color:#6b7280; font-weight:300; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:2rem; }

.answer-card {
    background:#161921; border:1px solid #1e2330; border-left:3px solid #e88a4a;
    border-radius:12px; padding:1.5rem; margin:1rem 0; animation:fadeUp 0.4s ease;
}
.source-card {
    background:#12151c; border:1px solid #1e2330; border-left:2px solid #2a3040;
    border-radius:8px; padding:1rem; margin:0.5rem 0; font-size:0.84rem; color:#8b92a0;
}
.answer-label { font-family:'Syne',sans-serif; font-size:0.68rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase; color:#e88a4a; margin-bottom:0.5rem; }
.source-label { font-family:'Syne',sans-serif; font-size:0.65rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase; color:#4a5568; margin-bottom:0.4rem; }
.answer-text { font-size:1.05rem; line-height:1.75; color:#e8e6e0; }

.badge { display:inline-block; padding:0.2rem 0.7rem; border-radius:100px; font-size:0.72rem; font-weight:500; }
.badge-success { background:#14291f; color:#4ade80; border:1px solid #1a3d28; }

.stat-box { background:#161921; border:1px solid #1e2330; border-radius:10px; padding:1rem; text-align:center; }
.stat-number { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:#e88a4a; }
.stat-desc { font-size:0.72rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; }

.filetype-tag { display:inline-block; padding:0.15rem 0.6rem; border-radius:6px; font-size:0.72rem; font-weight:700; font-family:'Syne',sans-serif; letter-spacing:0.08em; text-transform:uppercase; margin-right:0.3rem; }
.tag-pdf  { background:#2a0d0d; color:#f87171; border:1px solid #3d1414; }
.tag-txt  { background:#0d1a0d; color:#86efac; border:1px solid #143d14; }
.tag-docx { background:#0d142a; color:#93c5fd; border:1px solid #14233d; }
.tag-csv  { background:#1a0d2a; color:#c4b5fd; border:1px solid #2a143d; }

.fancy-divider { height:1px; background:linear-gradient(90deg,transparent,#1e2330 30%,#1e2330 70%,transparent); margin:1.5rem 0; }
.history-item { background:#12151c; border:1px solid #1e2330; border-radius:8px; padding:0.65rem 1rem; margin:0.3rem 0; }
.history-q { font-size:0.8rem; color:#9ca3af; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

.stTextInput > div > div > input { background:#161921 !important; border:1px solid #1e2330 !important; border-radius:10px !important; color:#e8e6e0 !important; }
.stTextInput > div > div > input:focus { border-color:#e88a4a !important; box-shadow:0 0 0 1px #e88a4a33 !important; }
.stButton > button { background:linear-gradient(135deg,#e88a4a,#c94f6e) !important; color:white !important; border:none !important; border-radius:10px !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; }
.stButton > button:hover { opacity:0.85 !important; }
[data-testid="stFileUploader"] { background:#161921; border:1px dashed #2a3040; border-radius:12px; padding:0.5rem; }

@keyframes fadeUp { from{opacity:0;transform:translateY(12px);} to{opacity:1;transform:translateY(0);} }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0d0f14; }
::-webkit-scrollbar-thumb { background:#1e2330; border-radius:4px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"qa_chain": None, "doc_stats": None, "chat_history": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Supported types config ────────────────────────────────────────────────────
SUPPORTED_TYPES = {
    "pdf":  {"label": "PDF",  "tag": "tag-pdf",  "emoji": "📄", "desc": "Reports, contracts, research papers"},
    "txt":  {"label": "TXT",  "tag": "tag-txt",  "emoji": "📝", "desc": "Plain text files, logs, notes"},
    "docx": {"label": "DOCX", "tag": "tag-docx", "emoji": "📘", "desc": "Word documents, essays, articles"},
    "csv":  {"label": "CSV",  "tag": "tag-csv",  "emoji": "📊", "desc": "Spreadsheet data, datasets, exports"},
}

QUICK_QUESTIONS = {
    "pdf":  ["Summarize this document", "What are the key risks?",     "What are the main rules?"],
    "txt":  ["Summarize the content",   "What are the main topics?",   "List the key points"],
    "docx": ["Summarize this document", "What decisions are mentioned?","List the action items"],
    "csv":  ["What columns are present?","Summarize the data",          "What trends are visible?"],
}


def get_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower()


# ── Document loader ───────────────────────────────────────────────────────────
def load_documents(file_bytes: bytes, filename: str):
    ext = get_ext(filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
        f.write(file_bytes)
        tmp = f.name
    try:
        loaders = {
            "pdf":  lambda p: PyPDFLoader(p),
            "txt":  lambda p: TextLoader(p, encoding="utf-8"),
            "docx": lambda p: Docx2txtLoader(p),
            "csv":  lambda p: CSVLoader(p, encoding="utf-8"),
        }
        if ext not in loaders:
            raise ValueError(f"Unsupported file type: .{ext}")
        docs = loaders[ext](tmp).load()
    finally:
        os.unlink(tmp)
    return docs


# ── Cached model & embeddings ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    class _Pipe:
        task = "text2text-generation"
        def __init__(self, m, t):
            self.model, self.tokenizer = m, t
        def _gen(self, prompt):
            ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            out = self.model.generate(ids, max_new_tokens=256, do_sample=True, temperature=0.5, top_k=50, top_p=0.95)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
        def __call__(self, inputs, **kw):
            if isinstance(inputs, list):
                return [[{"generated_text": self._gen(p)}] for p in inputs]
            return [{"generated_text": self._gen(inputs)}]

    return HuggingFacePipeline(pipeline=_Pipe(model, tokenizer))


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ── Build QA chain ────────────────────────────────────────────────────────────
def build_qa_chain(file_bytes: bytes, filename: str):
    raw_docs = load_documents(file_bytes, filename)
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(raw_docs)
    vectorstore = FAISS.from_documents(chunks, load_embeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=load_llm(), retriever=retriever, return_source_documents=True)
    ext = get_ext(filename)
    return qa, {"filename": filename, "ext": ext, "pages": len(raw_docs), "chunks": len(chunks)}


# ══ SIDEBAR ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem;">DocMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="font-size:0.68rem;">RAG-Powered Document Q&A</div>', unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Format badges
    st.markdown("**Supported formats**")
    badges = "".join(f'<span class="filetype-tag {v["tag"]}">{v["label"]}</span>' for v in SUPPORTED_TYPES.values())
    st.markdown(badges, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Uploader
    uploaded_file = st.file_uploader(
        "Upload document",
        type=list(SUPPORTED_TYPES.keys()),
        label_visibility="collapsed",
    )

    if uploaded_file:
        ext = get_ext(uploaded_file.name)
        meta = SUPPORTED_TYPES.get(ext, {})
        st.markdown(
            f'<div style="margin:0.5rem 0;font-size:0.82rem;color:#9ca3af;">'
            f'{meta.get("emoji","📄")} <b>{uploaded_file.name}</b> '
            f'<span class="filetype-tag {meta.get("tag","")}">{ext.upper()}</span></div>',
            unsafe_allow_html=True,
        )

        if st.button("⚡  Process Document", use_container_width=True):
            with st.spinner("Reading & indexing…"):
                try:
                    qa, stats = build_qa_chain(uploaded_file.read(), uploaded_file.name)
                    st.session_state.qa_chain = qa
                    st.session_state.doc_stats = stats
                    st.session_state.chat_history = []
                    st.success("Ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Doc stats
    if st.session_state.doc_stats:
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        s = st.session_state.doc_stats
        meta = SUPPORTED_TYPES.get(s["ext"], {})
        st.markdown(
            f'<span class="badge badge-success">● Indexed</span> '
            f'<span class="filetype-tag {meta.get("tag","")}">{s["ext"].upper()}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"<div style='font-size:0.78rem;color:#6b7280;margin:0.4rem 0;'>{s['filename']}</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        label = "Rows" if s["ext"] == "csv" else "Pages"
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{s["pages"]}</div><div class="stat-desc">{label}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{s["chunks"]}</div><div class="stat-desc">Chunks</div></div>', unsafe_allow_html=True)

    # History
    if st.session_state.chat_history:
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Recent Questions**")
        for item in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f'<div class="history-item"><div class="history-q">Q: {item["q"]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;color:#374151;text-align:center;">FLAN-T5 Base · FAISS · LangChain</div>', unsafe_allow_html=True)


# ══ MAIN ═════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">DocMind</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Ask anything about your document</div>', unsafe_allow_html=True)

if not st.session_state.qa_chain:
    # Welcome / empty state with format cards
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Supported File Types")
    cols = st.columns(4)
    for i, (ext, meta) in enumerate(SUPPORTED_TYPES.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="stat-box" style="text-align:left;padding:1.2rem;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">{meta['emoji']}</div>
                <span class="filetype-tag {meta['tag']}">{meta['label']}</span>
                <div style="font-size:0.78rem;color:#6b7280;margin-top:0.6rem;">{meta['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:3rem 2rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:#4b5563;font-weight:700;">
            Upload a PDF, TXT, DOCX, or CSV from the sidebar to get started
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    ext = st.session_state.doc_stats.get("ext", "pdf")
    quick_qs = QUICK_QUESTIONS.get(ext, QUICK_QUESTIONS["pdf"])

    st.markdown("**Quick questions:**")
    q_cols = st.columns(3)
    selected_q = None
    for i, q in enumerate(quick_qs):
        with q_cols[i]:
            if st.button(q, use_container_width=True, key=f"qq_{i}"):
                selected_q = q

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    with st.form("qform", clear_on_submit=True):
        user_q = st.text_input("", placeholder="Type your question here…", label_visibility="collapsed")
        submitted = st.form_submit_button("Ask →")

    final_q = selected_q or (user_q.strip() if submitted and user_q.strip() else None)

    if final_q:
        with st.spinner("Retrieving & generating…"):
            try:
                result = st.session_state.qa_chain({"query": final_q})
                answer = result["result"]
                sources = result.get("source_documents", [])
                st.session_state.chat_history.append({"q": final_q, "a": answer})

                st.markdown(f"""
                <div class="answer-card">
                    <div class="answer-label">Answer</div>
                    <div class="answer-text">{answer}</div>
                </div>
                """, unsafe_allow_html=True)

                if sources:
                    with st.expander(f"📚  View {len(sources)} source chunk(s)"):
                        for i, doc in enumerate(sources):
                            page_meta = doc.metadata.get("page", doc.metadata.get("row", "—"))
                            snippet = doc.page_content[:400] + ("…" if len(doc.page_content) > 400 else "")
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-label">Chunk {i+1} · {page_meta}</div>
                                {snippet}
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # Previous Q&A
    history = st.session_state.chat_history
    prev = history[:-1] if final_q and history else (history if not final_q else [])
    if prev:
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Previous in this session**")
        for item in reversed(prev[-5:]):
            with st.expander(f"Q: {item['q'][:80]}"):
                st.markdown(f"""
                <div class="answer-card" style="margin:0;">
                    <div class="answer-label">Answer</div>
                    <div class="answer-text">{item['a']}</div>
                </div>
                """, unsafe_allow_html=True)
