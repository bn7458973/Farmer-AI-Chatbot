import os
import csv
import difflib
from typing import List, Optional, Tuple

KB_FILE = os.path.join(os.path.dirname(__file__), "knowledge_base.txt")

# ── Load knowledge base ──
def load_knowledge_base() -> List[str]:
    if not os.path.exists(KB_FILE):
        return [
            "DAP fertilizer improves root growth in crops.",
            "NPK fertilizers affect leaf and fruit development.",
            "Apply fertilizer based on soil test results for best yield.",
            "Watering in the morning reduces evaporation and stress.",
            "Crop rotation improves soil health and reduces pests.",
            "Rice needs 5-10 cm standing water during growing season.",
            "Wheat requires 4-5 irrigations for optimal yield.",
            "Drip irrigation saves 30-50% water compared to flood irrigation.",
            "Integrated Pest Management uses natural methods first.",
            "Mulching reduces water loss and controls weeds."
        ]

    with open(KB_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Guardrail: ignore accidental shell/setup lines that can pollute retrieval quality.
    noisy_prefixes = (
        "printf ",
        "cat .env",
        "pkill ",
        "PORT=",
        "python app.py",
        "source .venv",
        "pip install ",
    )
    cleaned_lines = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            cleaned_lines.append(raw)
            continue
        if "OPENAI_API_KEY" in line or "GEMINI_API_KEY" in line:
            continue
        if line.startswith(noisy_prefixes):
            continue
        if line.startswith("# create or overwrite .env") or line.startswith("# confirm file") or line.startswith("# stop any running app instances") or line.startswith("# start app"):
            continue
        cleaned_lines.append(raw)
    content = "\n".join(cleaned_lines)

    chunks, current_chunk = [], ""
    for line in content.split('\n'):
        if line.startswith('#'):
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            current_chunk += "\n" + line
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    final_chunks = []
    for chunk in chunks:
        if len(chunk) > 300:
            sentences = chunk.replace('.\n', '.|').replace('. ', '.|').split('|')
            sub = ""
            for sent in sentences:
                if len(sub) + len(sent) > 300:
                    if sub.strip():
                        final_chunks.append(sub.strip())
                    sub = sent
                else:
                    sub += sent
            if sub.strip():
                final_chunks.append(sub.strip())
        else:
            final_chunks.append(chunk)

    return [c for c in final_chunks if c.strip()]


documents = load_knowledge_base()
USE_SEMANTIC_RAG = os.getenv("USE_SEMANTIC_RAG", "0").strip().lower() in {"1", "true", "yes", "on"}

# ── Try to load sentence-transformers for semantic search ──
_embedder = None
_doc_embeddings = None

def _load_embedder():
    global _embedder, _doc_embeddings
    if _embedder is not None:
        return True
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        print("[RAG] Loading semantic embedder (first time may take ~30s)...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, free, local
        _doc_embeddings = _embedder.encode(documents, convert_to_numpy=True)
        print(f"[RAG] Semantic embedder ready. {len(documents)} chunks indexed.")
        return True
    except Exception as e:
        print(f"[RAG] Semantic search unavailable ({e}), using keyword fallback.")
        return False


def _semantic_retrieve(query: str, top_k: int = 3) -> str:
    import numpy as np
    q_emb = _embedder.encode([query], convert_to_numpy=True)[0]
    # cosine similarity
    norms = np.linalg.norm(_doc_embeddings, axis=1) * np.linalg.norm(q_emb)
    norms = np.where(norms == 0, 1e-9, norms)
    scores = (_doc_embeddings @ q_emb) / norms
    top_idx = scores.argsort()[::-1][:top_k]
    return "\n\n".join(documents[i] for i in top_idx)


def _keyword_retrieve(query: str, top_k: int = 3) -> str:
    query_words = set(query.lower().split())
    scored = []
    for doc in documents:
        doc_lower = doc.lower()
        score = sum(3 if w in doc_lower else 0 for w in query_words if len(w) > 2)
        score += sum(1 for w in query_words if w in doc_lower)
        if score > 0:
            scored.append((score, doc))
    if scored:
        scored.sort(reverse=True, key=lambda x: x[0])
        return "\n\n".join(d for _, d in scored[:top_k])
    return "General agriculture advice: Practice crop rotation, use balanced fertilizers, and conserve water."


def retrieve_context(query: str, top_k: int = 3) -> str:
    try:
        if not query or not documents:
            return "General agriculture advice: Practice crop rotation, use balanced fertilizers, and conserve water."
        # Keyword search is the default because it is much faster on first response.
        # Semantic search can be enabled explicitly with USE_SEMANTIC_RAG=1.
        if USE_SEMANTIC_RAG and _load_embedder():
            return _semantic_retrieve(query, top_k)
        return _keyword_retrieve(query, top_k)
    except Exception as e:
        print(f"[RAG] retrieve_context error: {e}")
        return _keyword_retrieve(query, top_k)


def exact_match_answer(query: str, csv_path: Optional[str] = None, threshold: float = 0.9) -> Tuple[Optional[str], float]:
    try:
        if not csv_path:
            csv_path = os.path.join(os.path.dirname(__file__), 'farmer_qa_dataset.csv')
        if not os.path.exists(csv_path):
            return None, 0.0

        q_norm = ' '.join(query.lower().split())
        best_score, best_answer = 0.0, None

        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                q = row.get('question') or row.get('Question') or row.get('q') or ''
                a = row.get('answer') or row.get('Answer') or row.get('a') or ''
                if not q:
                    continue
                q_cmp = ' '.join(q.lower().split())
                if q_cmp == q_norm:
                    return a.strip(), 1.0
                score = difflib.SequenceMatcher(None, q_cmp, q_norm).ratio()
                if score > best_score:
                    best_score, best_answer = score, a.strip()

        return (best_answer, best_score) if best_score >= threshold else (None, best_score)
    except Exception as e:
        print(f"[RAG] exact_match error: {e}")
        return None, 0.0


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "how to grow rice"
    print(f"Query: {q}\nContext:\n{retrieve_context(q)}")
