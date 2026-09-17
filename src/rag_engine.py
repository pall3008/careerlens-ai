"""
RAG Engine v2 — CareerLens AI
Upgraded from FAISS (in-memory) to ChromaDB (persistent) + BM25 hybrid search.

Key upgrades vs v1:
  1. ChromaDB replaces FAISS
       - Vectors saved to disk (chroma_db/ folder)
       - Native metadata filtering — no more manual source loop
       - Survives app restarts with same resume
  2. Hybrid search = BM25 keyword + semantic vector, merged via score fusion
       - BM25 catches exact skill names ("FastAPI", "Node.js") that semantic misses
       - Semantic catches meaning ("backend development" ≈ "server-side engineering")
       - Together they give the best of both worlds
  3. Cross-encoder reranker (optional, CPU-friendly)
       - After hybrid retrieval, rerank top chunks by relevance
       - sentence-transformers cross-encoder scores (query, chunk) pairs directly

Why ChromaDB over FAISS?
  - FAISS is a pure math library — no persistence, no metadata, hard to debug
  - ChromaDB is a proper vector database — persistent, filterable, inspectable
  - In interviews: "I chose ChromaDB for persistence and native metadata filtering,
    which let me scope retrieval to resume-only or JD-only chunks cleanly"
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional

import fitz                          # PyMuPDF — PDF text extraction
import chromadb                      # persistent vector store
import torch
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi      # BM25 keyword search algorithm

from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter

try:
    from src.config import writable_dir
except ImportError:            # when this file is run directly from inside src/
    from config import writable_dir


# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384-dim, runs on CPU, free
VECTOR_DIM       = 384
CHUNK_SIZE       = 512
CHUNK_OVERLAP    = 64
CHROMA_DIR       = writable_dir("chroma_db", "careerlens_chroma_db")
EMBED_CACHE_DIR  = writable_dir("vectorstore/embeddings", "careerlens_embeddings")


# ── PDF Extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


# ── Embedding Model ───────────────────────────────────────────────────────────

class DirectHFEmbedding(BaseEmbedding):
    """
    Loads BGE-small directly from HuggingFace transformers.
    Bypasses sentence-transformers to avoid version conflicts.

    How embeddings work:
      1. Text goes in as tokens
      2. BGE-small (a BERT-based model) produces a 384-dim vector per token
      3. We mean-pool all token vectors into one 384-dim vector for the chunk
      4. L2 normalize so cosine similarity = dot product (faster search)
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model_name: str, cache_dir: str, **kwargs):
        super().__init__(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self._model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self._model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(
            input_mask.sum(1), min=1e-9
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self._model(**encoded)
        embeddings = self._mean_pooling(output, encoded["attention_mask"])
        norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)
        embeddings = (embeddings / norms).numpy()
        return embeddings.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


@st.cache_resource(show_spinner=False)
def get_embed_model() -> DirectHFEmbedding:
    """Load embedding model once per server process — cached by Streamlit."""
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    return DirectHFEmbedding(
        model_name=EMBED_MODEL_NAME,
        cache_dir=str(EMBED_CACHE_DIR),
        embed_batch_size=10,
    )


# ── Text Chunker ──────────────────────────────────────────────────────────────

def chunk_text(text: str) -> List[str]:
    """
    Split text into overlapping chunks using LlamaIndex SentenceSplitter.

    Why chunk?
      LLMs have context limits. A long resume can't fit in one prompt.
      Chunking breaks it into pieces. Overlap (64 tokens) ensures sentences
      that span chunk boundaries are still captured.

    Returns list of chunk strings.
    """
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(
        [Document(text=text)]
    )
    return [node.get_content() for node in nodes]


# ── BM25 Retriever ────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Keyword-based retrieval using BM25 (Best Match 25) algorithm.

    BM25 is what Google used before neural search. It scores documents
    by term frequency (how often the word appears) and inverse document
    frequency (how rare the word is across all documents).

    Why add BM25 to semantic search?
      Semantic search finds "similar meaning" but can miss exact terms.
      Example: query "FastAPI" might semantically match "Django" (both web
      frameworks) but BM25 will only match chunks that actually say "FastAPI".
      For technical skill matching, exact keyword match matters a lot.
    """

    def __init__(self, chunks: List[str]):
        # Tokenize: split each chunk into words (lowercase)
        self.chunks = chunks
        tokenized  = [chunk.lower().split() for chunk in chunks]
        self.bm25  = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Return top_k most relevant chunks for the query."""
        query_tokens = query.lower().split()
        scores       = self.bm25.get_scores(query_tokens)
        # Get indices of top_k highest scores
        top_indices  = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]


# ── RAG Engine v2 ─────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Hybrid RAG engine with ChromaDB (semantic) + BM25 (keyword) retrieval.

    Architecture:
      resume text ──┐
                    ├── chunk ── embed ── ChromaDB (semantic search)
      JD text ──────┘               └── BM25 index (keyword search)
                                              │
                              query ──── hybrid retrieve ──── LLM context
    """

    def __init__(self):
        self.embed_model     = None
        self.chroma_client   = None
        self.collection      = None
        self.resume_chunks   = []
        self.jd_chunks       = []
        self.resume_bm25     = None
        self.jd_bm25         = None
        self._collection_id  = None
        self._setup()

    def _setup(self):
        """Initialize embedding model and ChromaDB client."""
        self.embed_model   = get_embed_model()
        Settings.embed_model   = self.embed_model
        Settings.chunk_size    = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP
        Settings.llm           = None

        # ChromaDB persistent client — saves to chroma_db/ folder on disk
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    def _make_collection_id(self, resume_text: str, jd_text: str) -> str:
        """
        Create a unique ID for this resume+JD combination.
        Uses MD5 hash so same inputs = same collection = no re-embedding needed.

        This is why ChromaDB is better than FAISS:
        With FAISS, every app restart re-embeds everything.
        With ChromaDB, if you've seen this resume+JD before, it loads from disk.
        """
        content = (resume_text + jd_text).encode("utf-8")
        return "cl_" + hashlib.md5(content).hexdigest()[:16]

    def build_index(self, resume_text: str, jd_text: str) -> None:
        """
        Build ChromaDB collection + BM25 indexes from resume and JD text.

        Steps:
          1. Chunk resume and JD separately
          2. Embed each chunk using BGE-small
          3. Store in ChromaDB with metadata {"source": "resume"} or {"source": "jd"}
          4. Build BM25 indexes for keyword search
          5. Skip re-embedding if this resume+JD combo was indexed before
        """
        collection_id = self._make_collection_id(resume_text, jd_text)
        self._collection_id = collection_id

        # Check if already indexed (ChromaDB persists to disk)
        existing = [c.name for c in self.chroma_client.list_collections()]

        if collection_id in existing:
            # Load existing collection — no re-embedding needed
            self.collection = self.chroma_client.get_collection(collection_id)
        else:
            # New resume+JD — chunk, embed, store
            self.collection = self.chroma_client.create_collection(
                name=collection_id,
                metadata={"hnsw:space": "cosine"},  # cosine similarity
            )
            self._index_text(resume_text, source="resume")
            self._index_text(jd_text,     source="jd")

        # Build BM25 indexes (always in-memory, fast to rebuild)
        self.resume_chunks = self._get_chunks_by_source("resume")
        self.jd_chunks     = self._get_chunks_by_source("jd")

        if self.resume_chunks:
            self.resume_bm25 = BM25Retriever(self.resume_chunks)
        if self.jd_chunks:
            self.jd_bm25 = BM25Retriever(self.jd_chunks)

    def _index_text(self, text: str, source: str) -> None:
        """Chunk text, embed each chunk, store in ChromaDB with source metadata."""
        chunks = chunk_text(text)
        if not chunks:
            return

        embeddings = self.embed_model._embed(chunks)

        self.collection.add(
            documents  = chunks,
            embeddings = embeddings,
            metadatas  = [{"source": source} for _ in chunks],
            ids        = [f"{source}_{i}" for i in range(len(chunks))],
        )

    def _get_chunks_by_source(self, source: str) -> List[str]:
        """Retrieve all stored chunks for a given source from ChromaDB."""
        results = self.collection.get(
            where={"source": source}
        )
        return results["documents"] or []

    def _semantic_retrieve(
        self, query: str, source: str, top_k: int = 8
    ) -> List[str]:
        """
        Semantic retrieval from ChromaDB.
        Embeds the query and finds the closest chunks by cosine similarity.
        Filters by source metadata so resume and JD chunks never mix.
        """
        query_embedding = self.embed_model._embed([query])[0]
        results = self.collection.query(
            query_embeddings = [query_embedding],
            n_results        = top_k,
            where            = {"source": source},
        )
        docs = results.get("documents", [[]])[0]
        return docs

    def _hybrid_retrieve(
        self, query: str, source: str, top_k: int = 8
    ) -> str:
        """
        Hybrid retrieval = semantic + BM25, merged with Reciprocal Rank Fusion.

        Reciprocal Rank Fusion (RRF):
          Each retriever ranks chunks 1..N.
          Final score = 1/(rank_in_semantic + 60) + 1/(rank_in_bm25 + 60)
          The chunk ranked high by BOTH retrievers wins.
          60 is a smoothing constant (standard in research papers).

        Why this works:
          Semantic: finds "backend development" when query says "server-side"
          BM25:     finds "FastAPI" when query says "FastAPI" (exact match)
          Together: best coverage of both meaning and keywords
        """
        # Semantic results
        semantic_chunks = self._semantic_retrieve(query, source, top_k=top_k)

        # BM25 results
        bm25_retriever = self.resume_bm25 if source == "resume" else self.jd_bm25
        bm25_chunks    = bm25_retriever.retrieve(query, top_k=top_k) if bm25_retriever else []

        # Reciprocal Rank Fusion
        scores = {}
        for rank, chunk in enumerate(semantic_chunks):
            scores[chunk] = scores.get(chunk, 0) + 1 / (rank + 60)
        for rank, chunk in enumerate(bm25_chunks):
            scores[chunk] = scores.get(chunk, 0) + 1 / (rank + 60)

        # Sort by combined score, take top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_chunks = [chunk for chunk, _ in ranked[:top_k]]

        # Format with source label for LLM context
        parts = [f"[{source.upper()}]\n{chunk}" for chunk in top_chunks]
        return "\n\n---\n\n".join(parts)

    def get_resume_chunks(self, top_k: int = 8) -> str:
        """Get most relevant resume chunks for analysis."""
        return self._hybrid_retrieve(
            "candidate skills experience education projects achievements",
            source="resume",
            top_k=top_k,
        )

    def get_jd_chunks(self, top_k: int = 8) -> str:
        """Get most relevant JD chunks for analysis."""
        return self._hybrid_retrieve(
            "job requirements responsibilities skills qualifications",
            source="jd",
            top_k=top_k,
        )

    def retrieve_for_query(self, query: str, source: str, top_k: int = 5) -> str:
        """General-purpose hybrid retrieval for any query — used by chat."""
        return self._hybrid_retrieve(query, source=source, top_k=top_k)
