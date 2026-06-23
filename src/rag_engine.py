"""
RAG Engine - Core of CareerLens AI
Handles: PDF parsing → chunking → embedding → vector store → retrieval
"""

import os
import fitz  # PyMuPDF
import faiss
from pathlib import Path
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from typing import List


# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIM       = 384
CHUNK_SIZE       = 512
CHUNK_OVERLAP    = 64
VECTORSTORE_DIR  = Path("vectorstore")


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


class DirectHFEmbedding(BaseEmbedding):
    """
    Directly uses HuggingFace transformers — bypasses sentence-transformers
    entirely to avoid the safe_serialization / prompts compatibility crash.
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
        # L2 normalise
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
    """
    Load the model once per server process via @st.cache_resource.
    Uses DirectHFEmbedding which calls transformers directly — no
    sentence-transformers involved, so no version conflicts.
    """
    cache_dir = str((VECTORSTORE_DIR / "embeddings").resolve())
    os.makedirs(cache_dir, exist_ok=True)
    return DirectHFEmbedding(
        model_name=EMBED_MODEL_NAME,
        cache_dir=cache_dir,
        embed_batch_size=10,
    )


# ── RAG Engine ────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Builds a FAISS vector index from resume + JD text and retrieves relevant chunks.
    """

    def __init__(self):
        self.index       = None
        self.embed_model = None
        self._setup_settings()

    def _setup_settings(self):
        self.embed_model       = get_embed_model()
        Settings.embed_model   = self.embed_model
        Settings.chunk_size    = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP
        Settings.llm           = None   # Groq handles all LLM calls directly

    def build_index(self, resume_text: str, jd_text: str) -> None:
        """Build a FAISS vector store from resume + job description."""
        documents = [
            Document(text=resume_text, metadata={"source": "resume"}),
            Document(text=jd_text,     metadata={"source": "job_description"}),
        ]

        faiss_index  = faiss.IndexFlatL2(VECTOR_DIM)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_ctx,
            show_progress=False,
        )

    def retrieve(self, query: str, top_k: int = 5, source: str = None) -> str:
        """
        Retrieve top-k relevant chunks for a query.
        Filters by source ('resume' or 'job_description') client-side
        since FAISS flat-index has no native metadata filtering.
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index() first.")

        fetch_k   = max(top_k * 4, 20) if source else top_k
        retriever = self.index.as_retriever(similarity_top_k=fetch_k)
        nodes     = retriever.retrieve(query)

        if source:
            nodes = [n for n in nodes if n.metadata.get("source") == source]
        nodes = nodes[:top_k]

        parts = []
        for node in nodes:
            src = node.metadata.get("source", "unknown")
            parts.append(f"[{src.upper()}]\n{node.get_content()}")

        return "\n\n---\n\n".join(parts)

    def get_resume_chunks(self, top_k: int = 8) -> str:
        return self.retrieve(
            "candidate skills experience education projects achievements",
            top_k=top_k,
            source="resume",
        )

    def get_jd_chunks(self, top_k: int = 8) -> str:
        return self.retrieve(
            "job requirements responsibilities skills qualifications",
            top_k=top_k,
            source="job_description",
        )
