# DocMind – RAG-Powered Document Q&A

A Streamlit app that lets you upload any PDF and ask questions about it using FLAN-T5 + FAISS + LangChain.

---

## 🚀 Deployment Options

### Option A — Streamlit Community Cloud (Free)

1. Push these files to a **GitHub repo** (public or private):
   ```
   app.py
   requirements.txt
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Connect your GitHub repo, set **Main file** to `app.py`
4. Click **Deploy** — done!

> ⚠️ Free tier has ~1 GB RAM. FLAN-T5 Base fits comfortably.

---

### Option B — Run Locally

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Visit: http://localhost:8501

---

### Option C — Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t docmind .
docker run -p 8501:8501 docmind
```

---

## 📁 File Structure

```
your-repo/
├── app.py            ← Main Streamlit application
├── requirements.txt  ← Python dependencies
└── README.md
```

---

## 🧠 How It Works

1. **Upload PDF** → parsed with PyPDF
2. **Chunking** → split into 500-char chunks (100 overlap) via LangChain
3. **Embeddings** → `sentence-transformers/all-MiniLM-L6-v2`
4. **Vector Store** → FAISS (in-memory, per session)
5. **LLM** → `google/flan-t5-base` (Seq2Seq, no GPU needed)
6. **RAG Chain** → LangChain `RetrievalQA` (top-3 chunks)

---

## ⚡ Tips

- First run downloads models (~300 MB) — subsequent runs use cache
- For better answers, upgrade to `google/flan-t5-large` or `flan-t5-xl`
- For GPU deployment, remove `faiss-cpu` and add `faiss-gpu`
