# 🎯 CareerLens AI
### RAG-Powered Resume Analyzer + Interview Prep Agent + Live Job Matcher

> **Built with:** Groq (Llama-3.3-70B) · LlamaIndex · FAISS · HuggingFace Embeddings · Streamlit · Adzuna/Remotive

---

## 🚀 What This Project Does

CareerLens AI is a **production-quality AI engineering project** that combines:

- **RAG (Retrieval-Augmented Generation)** — Your resume + job description are chunked, embedded, and stored in a FAISS vector store. Relevant chunks are retrieved *per-source* (resume-only / JD-only) for each LLM call.
- **LLM Agent** — Groq's Llama-3.3-70B performs structured analysis, generates personalized interview questions, evaluates your answers, and distills your resume into a real job-search query.
- **Agentic Loop** — The interview prep section creates an interactive agent loop: question → your answer → AI feedback → next question.
- **Live Job Matching** — Your resume is turned into a search query and sent to real job board APIs (Adzuna + Remotive) to surface current openings with direct apply links.

### Key Features
| Feature | Description |
|---|---|
| 📊 Resume Analysis | Match score (0-100), strengths, gaps, missing keywords |
| 🔎 Live Job Matches | Real current openings sourced from job boards, with apply links — built from your resume's actual skills |
| 🎤 Interview Questions | 10 personalized questions (Technical + Behavioral + Gap-based) |
| 🤖 Answer Evaluation | Score your answer, give model answer + STAR tips |
| 💬 Career Chat | RAG-powered freeform Q&A about your resume + role |

---

## 🏗️ Architecture

```
User uploads PDF + JD
        │
        ▼
┌─────────────────────┐
│   PDF Parser        │  PyMuPDF → raw text
│   (PyMuPDF)         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   RAG Engine        │  LlamaIndex chunks text
│   (LlamaIndex)      │  BGE-small embeds chunks
│                     │  FAISS stores vectors
│                     │  Source-scoped retrieval (resume-only / JD-only)
└────────┬────────────┘
         │  retrieve relevant chunks
         ▼
┌─────────────────────┐      ┌──────────────────────┐
│   LLM Agent         │─────▶│   Job Search Engine   │
│   (Groq API)        │      │   (Adzuna + Remotive) │
│   Structured JSON    │      │   Live listings + links│
└────────┬────────────┘      └──────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Streamlit UI      │  5-step flow incl. Job Matches
│                     │  Interactive agent loop
└─────────────────────┘
```

---

## ⚙️ Setup (5 minutes)

### 1. Install dependencies
```bash
cd careerlens
pip install -r requirements.txt
```

### 2. Get your FREE Groq API key
- Go to [console.groq.com](https://console.groq.com)
- Sign up (free) → Create API Key
- Copy your key

### 3. (Optional, recommended) Get a FREE Adzuna API key for full job coverage
- Go to [developer.adzuna.com](https://developer.adzuna.com/)
- Sign up free (no credit card) → grab your `app_id` and `app_key`
- Without this, the Job Matches tab still works using Remotive (remote tech jobs only, no key required)

### 4. Set up environment
```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY (and optionally ADZUNA_APP_ID / ADZUNA_APP_KEY)
```
⚠️ **Never commit `.env` to git** — it's already covered by `.gitignore`. If a key was ever shared or committed by mistake, rotate it from the provider's dashboard.

### 5. Run
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
careerlens/
├── app.py                  # Streamlit UI (5-step flow)
├── src/
│   ├── rag_engine.py       # PDF parsing, FAISS index, source-scoped retrieval
│   ├── llm_agent.py        # All Groq API calls + prompt engineering
│   └── job_search.py       # Live job board search (Adzuna + Remotive)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔑 Key Technical Concepts Demonstrated

1. **RAG Pipeline** — Document ingestion → chunking → embedding → vector storage → source-scoped semantic retrieval
2. **Prompt Engineering** — Structured JSON output, few-shot examples, role-based system prompts
3. **Agentic Loop** — Multi-turn conversation with state management, context injection
4. **Vector Search** — FAISS IndexFlatL2, cosine similarity, top-k retrieval, client-side metadata filtering
5. **External API Integration** — Live job board APIs (Adzuna, Remotive) driven by LLM-extracted search intent
6. **Free Stack** — 100% free APIs (Groq free tier + HuggingFace local embeddings + Adzuna/Remotive free tiers)

---

## 🛠️ Recent Fixes

- **Groq model decommissioned**: `llama3-70b-8192` was retired by Groq → updated to `llama-3.3-70b-versatile` everywhere.
- **Resume/JD context leakage**: retrieval previously searched the whole index regardless of source, so "resume chunks" could silently include JD text and vice versa. Retrieval is now scoped per-source.
- **Dependency drift**: `requirements.txt` now matches the actually-installed, working package versions.

---

## 💡 How to Talk About This in Interviews

> *"I built a RAG-based career assistant using LlamaIndex and FAISS for vector storage. The system takes a resume PDF and job description, chunks them using LlamaIndex's node parser, embeds them with BGE-small from HuggingFace, and retrieves source-scoped context for each LLM call. The LLM layer uses Groq's Llama-3.3-70B API with structured prompt engineering to return JSON for analysis, question generation, and answer evaluation. A separate agent step distills the resume into a real job-search query, which I send to live job board APIs to surface current openings with direct apply links. The interview prep module implements an agentic loop where the LLM evaluates candidate answers and provides structured feedback."*

---

## 🔮 Future Improvements (show initiative)

- [ ] Add RAGAS evaluation metrics (faithfulness, answer relevance)
- [ ] Support multiple resume comparison
- [ ] Add voice input/output for interview simulation
- [ ] Deploy to Hugging Face Spaces
- [ ] Add hybrid search (BM25 + semantic)
- [ ] Rank job matches by resume-skill overlap score, not just recency
- [ ] Cache job search results to avoid re-hitting APIs on every rerun

---

## 📚 Resources

- [LlamaIndex Docs](https://docs.llamaindex.ai)
- [Groq API](https://console.groq.com)
- [Groq Model Deprecations](https://console.groq.com/docs/deprecations)
- [FAISS](https://github.com/facebookresearch/faiss)
- [BGE Embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [Adzuna API](https://developer.adzuna.com/)
- [Remotive API](https://remotive.com/api-documentation)
