"""PDF text extraction + semantic-embedding wrapper."""
from pathlib import Path
from typing import List
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = None


def get_model() -> SentenceTransformer:
    """Lazy-load — downloads ~90MB weights on first use."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2", token=False)
    return _MODEL


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_text(text: str) -> List[str]:
    """Split into semantic chunks: bullet lines + sentences."""
    chunks = []
    for raw in text.splitlines():
        line = raw.strip(" •●-*\t")
        if len(line) < 15:
            continue
        # Long lines get split by period too
        if len(line) > 200:
            for s in line.split(". "):
                s = s.strip()
                if len(s) > 15:
                    chunks.append(s)
        else:
            chunks.append(line)
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return L2-normalized embeddings (cosine sim == dot product)."""
    if not texts:
        return np.zeros((0, 384))
    model = get_model()
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return emb


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity (both already normalized)."""
    return a @ b.T