import os
import sys
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFacePipeline
import tempfile
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – RAG Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d0f14;
    color: #e8e6e0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #12151c !important;
    border-right: 1px solid #1e2330;
}

/* Hero title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f0c27f 0%, #e88a4a 50%, #c94f6e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #6b7280;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Cards */
.answer-card {
    background: #161921;
    border: 1px solid #1e2330;
    border-left: 3px solid #e88a4a;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    animation: fadeUp 0.4s ease;
}

.source-card {
    background: #12151c;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #8b92a0;
    border-left: 2px solid #2a3040;
    animation: fadeUp 0.4s ease;
}

.answer-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #e88a4a;
    margin-bottom: 0.5rem;
}

.source-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 0.4rem;
}

.answer-text {
    font-size: 1.05rem;
    line-height: 1.7;
    color: #e8e6e0;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 500;
    font-family: 'DM Sans', sans-serif;
}

.badge-success { background: #14291f; color: #4ade80; border: 1px solid #1a3d28; }
.badge-warning { background: #2a1f0d; color: #f0a946; border: 1px solid #3d2f14; }
.badge-info    { background: #0d1a2a; color: #60a5fa; border: 1px solid #142233; }

/* Stat boxes */
.stat-box {
    background: #161921;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #e88a4a;
}

.stat-desc {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Chat history item */
.history-item {
    background: #12151c;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin: 0.3rem 0;
    cursor: pointer;
    transition: border-color 0.2s;
}
.history-item:hover { border-color: #e88a4a; }
.history-q {
    font-size: 0.82rem;
    color: #9ca3af;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e2330 30%, #1e2330 70%, transparent);
    margin: 1.5rem 0;
}

/* Input styling override */
.stTextInput > div > div > input {
    background: #161921 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 10px !important;
    color: #e8e6e0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: #e88a4a !important;
    box-shadow: 0 0 0 1px #e88a4a33 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #e88a4a, #c94f6e) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #161921;
    border: 1px dashed #2a3040;
    border-radius: 12px;
    padding: 1rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #1e2330; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
for key in ["qa_chain", "doc_stats", "chat_history", "model_loaded"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "chat_history" else []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False


# ── Helper: custom pipeline ───────────────────────────────────────────────────
class MinimalPipeline:
    def __init__(self, model, tokenizer, max_new_tokens=256, temperature=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.task = "text2text-generation"

    def _generate(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def __call__(self, text_inputs, **kwargs):
        if isinstance(text_inputs, list):
            return [[{"generated_text": self._generate(p)}] for p in text_inputs]
        return [{"generated_text": self._generate(text_inputs)}]


@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = MinimalPipeline(model, tokenizer)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_qa_chain(pdf_bytes, filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = load_model()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain, {"pages": len(documents), "chunks": len(docs), "filename": filename}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem;">DocMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="font-size:0.7rem;">RAG-Powered Document Q&A</div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown("**Upload Document**")
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        if st.button("⚡  Process Document", use_container_width=True):
            with st.spinner("Loading model & indexing document…"):
                qa_chain, stats = build_qa_chain(uploaded_file.read(), uploaded_file.name)
                st.session_state.qa_chain = qa_chain
                st.session_state.doc_stats = stats
                st.session_state.chat_history = []
            st.success("Ready!")

    # Doc stats
    if st.session_state.doc_stats:
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        s = st.session_state.doc_stats
        st.markdown(f'<span class="badge badge-success">● Indexed</span>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8rem;color:#6b7280;margin-top:0.5rem;'>📄 {s['filename']}</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{s["pages"]}</div><div class="stat-desc">Pages</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{s["chunks"]}</div><div class="stat-desc">Chunks</div></div>', unsafe_allow_html=True)

    # Chat history
    if st.session_state.chat_history:
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Recent Questions**")
        for i, item in enumerate(reversed(st.session_state.chat_history[-6:])):
            st.markdown(f'<div class="history-item"><div class="history-q">Q: {item["q"]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#374151;text-align:center;">FLAN-T5 Base · FAISS · LangChain</div>', unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
col_title, _ = st.columns([3, 1])
with col_title:
    st.markdown('<div class="hero-title">DocMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Ask anything about your document</div>', unsafe_allow_html=True)

if not st.session_state.qa_chain:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#374151;">
        <div style="font-size:3rem;margin-bottom:1rem;">📄</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;color:#4b5563;font-weight:700;">
            Upload a PDF to get started
        </div>
        <div style="font-size:0.85rem;margin-top:0.5rem;color:#374151;">
            Use the sidebar to upload and index your document
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️  How it works"):
        st.markdown("""
        1. **Upload** a PDF via the sidebar  
        2. Click **Process Document** — the app splits it into chunks and builds a vector index  
        3. **Ask questions** in natural language  
        4. The model retrieves the most relevant chunks and generates a grounded answer  
        """)

else:
    # Quick-question chips
    st.markdown("**Try a quick question:**")
    q_cols = st.columns(3)
    quick_qs = ["Summarize this document", "What are the key risks?", "What are the main rules?"]
    selected_q = None
    for i, q in enumerate(quick_qs):
        with q_cols[i]:
            if st.button(q, use_container_width=True, key=f"quick_{i}"):
                selected_q = q

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Input
    with st.form("question_form", clear_on_submit=True):
        user_q = st.text_input(
            "Your question",
            placeholder="e.g. What are the eligibility requirements?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask →", use_container_width=False)

    final_q = selected_q or (user_q if submitted else None)

    if final_q:
        with st.spinner("Thinking…"):
            result = st.session_state.qa_chain({"query": final_q})
        answer = result["result"]
        sources = result.get("source_documents", [])

        # Save history
        st.session_state.chat_history.append({"q": final_q, "a": answer})

        # Render answer
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-label">Answer</div>
            <div class="answer-text">{answer}</div>
        </div>
        """, unsafe_allow_html=True)

        if sources:
            with st.expander(f"📚  View {len(sources)} source chunk(s)"):
                for i, doc in enumerate(sources):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-label">Chunk {i+1} · Page {page}</div>
                        {doc.page_content[:400]}{"…" if len(doc.page_content) > 400 else ""}
                    </div>
                    """, unsafe_allow_html=True)

    # Previous Q&A thread
    if st.session_state.chat_history:
        prev = st.session_state.chat_history[:-1] if final_q else st.session_state.chat_history
        if prev:
            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
            st.markdown("**Previous in this session**")
            for item in reversed(prev[-5:]):
                with st.expander(f"Q: {item['q'][:80]}…"):
                    st.markdown(f"""
                    <div class="answer-card" style="margin:0;">
                        <div class="answer-label">Answer</div>
                        <div class="answer-text">{item['a']}</div>
                    </div>
                    """, unsafe_allow_html=True)
