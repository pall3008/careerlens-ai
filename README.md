# 🎯 CareerLens AI — v2

### Hybrid RAG · NLP Skill Extraction · Voice Interview · MCP Server · Live Jobs

> **Stack:** Groq API · LlamaIndex · ChromaDB · BM25 · BGE embeddings · Whisper · FastMCP · Streamlit · Adzuna/Remotive

---

## What This Does

CareerLens AI is a resume analysis and interview preparation tool built on a hybrid RAG pipeline. Upload your resume PDF, paste a job description, and the system runs through 7 steps:

1. **Upload** — PDF parsed with PyMuPDF, text extracted
2. **Skills Map** — local NLP extracts skills against a 500+ term gazetteer (no API call), compares resume vs JD, shows overlap %
3. **Analysis** — Groq LLM gives match score, strengths, gaps, missing keywords
4. **Resume Audit** — section-by-section health check with auto-rewrite
5. **Live Jobs** — real listings from Adzuna + Remotive matched to your profile
6. **Interview Prep** — 10 personalized questions, type OR speak your answer (Whisper transcribes locally)
7. **Career Chat** — RAG-powered freeform Q&A about your resume and target role

---

## Architecture

```
Resume PDF + Job Description
         │
         ▼
┌─────────────────────────────────────────┐
│  NLP Layer (gazetteer + regex)          │
│  Skill extraction · Gap analysis        │
│  Runs locally — zero API cost           │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  RAG Engine v2                          │
│  PyMuPDF → chunk → BGE-small embed      │
│  ChromaDB (persistent to disk)          │
│  BM25 + semantic hybrid retrieval       │
│  Reciprocal Rank Fusion merge           │
│  Source-scoped (resume ≠ JD chunks)     │
└─────────────┬───────────────────────────┘
              │  relevant chunks
              ▼
┌─────────────────────────────────────────┐    ┌──────────────────────────┐
│  LLM Agent (Groq API)                   │───▶│  Job Search Engine        │
│  Structured JSON prompts                │    │  Adzuna (India + global)  │
│  Analysis · Audit · Questions · Chat    │    │  Remotive (remote, free)  │
└─────────────┬───────────────────────────┘    └──────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  FastMCP Server (mcp_server.py)         │
│  6 tools exposed as MCP protocol        │
│  Claude / Cursor / any MCP client       │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Streamlit UI (7-step flow)             │
│  Voice input via Whisper (local STT)    │
│  Skills Map · Audit · Jobs · Interview  │
└─────────────────────────────────────────┘
```

---

## Key Engineering Decisions

| Decision | Why |
|---|---|
| ChromaDB over FAISS | Persistent to disk — same resume doesn't re-embed on restart. Native metadata filtering for source-scoped retrieval. |
| BM25 + semantic hybrid | Semantic search misses exact skill names ("FastAPI", "Node.js"). BM25 catches exact keyword matches. Combined with RRF gives best of both. |
| Gazetteer before LLM | Skill extraction is deterministic and fast locally. No API cost, no hallucination risk for structured data like skill names. Tech skills are a closed, known set, so a curated vocabulary beats a statistical NER model here — and it ships no model weights. |
| Source-scoped retrieval | Resume and JD chunks were leaking into each other. Metadata filtering ensures resume-only and JD-only retrieval stays clean. |
| Whisper local STT | Zero API cost, full privacy. Runs on CPU. 74MB model, downloads once. |
| FastMCP server | Any MCP-compatible AI assistant (Claude Desktop, Cursor) can call CareerLens tools as plugins. |
| BGE-small embeddings | Runs locally on CPU. No OpenAI embedding API cost. 384-dim vectors, good quality for English text. |

---

## Project Structure

```
careerlens/
├── app.py                  # Streamlit UI — 7-step flow with Skills Map + voice
├── src/
│   ├── rag_engine.py       # ChromaDB + BM25 hybrid RAG + Reciprocal Rank Fusion
│   ├── llm_agent.py        # All Groq API calls — structured JSON prompts
│   ├── nlp_engine.py       # local skill extractor — 500+ skill vocabulary
│   ├── mcp_server.py       # FastMCP server — 6 tools for MCP protocol
│   └── job_search.py       # Live job search — Adzuna + Remotive APIs
├── chroma_db/              # ChromaDB persists vectors here (git-ignored)
├── vectorstore/            # BGE-small model cache (git-ignored)
├── .streamlit/
│   └── config.toml         # Streamlit settings (committed; secrets.toml is not)
├── requirements.txt        # deployment build — local-only extras at the bottom
├── .env.example            # template — copy to .env and fill in
├── .env                    # Your API keys (git-ignored — never commit this)
└── README.md
```

---

## Setup (5 minutes)

### 1. Clone and create environment
```bash
git clone <repo-url>
cd careerlens
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt

# Optional — only needed for voice interview, the MCP server, and RAG eval.
# These are deliberately left out of requirements.txt so the free-tier
# deployment stays under its memory limit.
pip install openai-whisper streamlit-mic-recorder soundfile fastmcp ragas
```

### 2. Get free API keys

**Groq (required)** — [console.groq.com](https://console.groq.com) → free signup → create API key

**Adzuna (optional, recommended)** — [developer.adzuna.com](https://developer.adzuna.com/) → free signup → grab app_id and app_key. Without this, jobs still show via Remotive (remote only).

### 3. Set up .env
```bash
GROQ_API_KEY=your_groq_key_here
ADZUNA_APP_ID=your_adzuna_id
ADZUNA_APP_KEY=your_adzuna_key
ADZUNA_COUNTRY=in
```

### 4. Run the app
```bash
streamlit run app.py --server.fileWatcherType none
```

### 5. (Optional) Run the MCP server
```bash
python src/mcp_server.py
```
Then connect from Claude Desktop or any MCP client.

---

## MCP Tools Available

| Tool | Description |
|---|---|
| `analyze_resume` | Match score, strengths, gaps, missing keywords |
| `audit_resume` | Section-by-section health check + ATS score |
| `generate_questions` | 10 personalized interview questions |
| `evaluate_interview_answer` | Score a spoken or typed answer out of 10 |
| `get_live_jobs` | Live listings from Adzuna + Remotive |
| `extract_resume_skills` | NLP skill extraction — no LLM call |

---

## Tech Stack — 100% Free

| Component | Technology | Cost |
|---|---|---|
| LLM | Groq API (free tier) | Free |
| Embeddings | BGE-small (local) | Free |
| Vector DB | ChromaDB (local) | Free |
| Keyword search | BM25 (pure Python) | Free |
| NLP | Gazetteer + regex (pure Python) | Free |
| Speech-to-text | Whisper base (local) | Free |
| MCP server | FastMCP | Free |
| Job boards | Adzuna free + Remotive | Free |
| UI | Streamlit | Free |

---

## How to Talk About This in Interviews

*"CareerLens AI is a resume analysis tool I built using a hybrid RAG pipeline. I store resume and job description chunks in ChromaDB for persistence, and retrieve them using both BM25 keyword search and semantic vector search combined through Reciprocal Rank Fusion — this ensures I catch exact technical skill names that pure semantic search sometimes misses. Before calling the LLM, I extract skills locally with a curated gazetteer of 500+ tech terms matched on word boundaries — deterministic, millisecond-fast, and zero API cost. I originally reached for spaCy here, then realised a statistical NER model adds nothing when the entity set is closed and known, so I dropped the dependency and the deployment got 200MB lighter. The LLM layer uses Groq with structured JSON prompts for analysis, audit, and interview question generation. For the interview step, users can speak their answer — Whisper transcribes it locally with no API call. I also built a FastMCP server that exposes all features as MCP tools, so any MCP-compatible AI assistant like Claude can use CareerLens as a plugin."*

---

## Known Limitations + Production Notes

- **Free tier rate limits**: Groq free tier limits output tokens/minute. `MAX_TOKENS` is set conservatively at 1024. Production deployment would use a paid tier or token-aware chunking.
- **FAISS → ChromaDB**: v1 used FAISS (in-memory, no persistence). v2 uses ChromaDB which persists to disk — same resume+JD pair never re-embeds.
- **Whisper model**: First voice use downloads the `base` model (74MB). Cached locally after that.
- **Scalability**: ChromaDB's local mode works for single-user demo. Multi-user production would use ChromaDB server mode or Pinecone.

---

## Future Improvements

- [x] Deploy to Streamlit Community Cloud (Hugging Face Spaces no longer offers a free Streamlit SDK — Gradio and Docker Spaces now require a paid plan)
- [ ] Add cover letter generator
- [ ] LinkedIn job scraping
- [ ] Sentence-BERT cross-encoder reranker after hybrid retrieval
- [ ] RAGAS evaluation metrics on the RAG pipeline
- [ ] Docker + docker-compose for one-command setup
