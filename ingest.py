# ingest.py
import os
import json
from pathlib import Path
import openai
import faiss
import numpy as np
import fitz  # PyMuPDF

openai.api_key = os.environ.get("OPENAI_API_KEY")

def extract_text_from_pdf(path: str):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i+1, "text": text})
    return pages

def chunk_text(text, chunk_size=1000, overlap=200):
    tokens = text.split()
    i = 0
    chunks = []
    while i < len(tokens):
        chunk = " ".join(tokens[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def embed_texts(texts):
    # Use OpenAI embeddings
    resp = openai.Embedding.create(model="text-embedding-3-small", input=texts)
    return [r["embedding"] for r in resp["data"]]

def build_faiss_index(docs_dir="sample_docs", index_path="faiss.index", meta_path="meta.json"):
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory '{docs_dir}' does not exist")

    pdfs = list(docs_path.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in '{docs_dir}'. Add documents then re-run.")

    metas = []
    embeddings = []
    for p in pdfs:
        pages = extract_text_from_pdf(str(p))
        for page in pages:
            chunks = chunk_text(page["text"], chunk_size=800, overlap=150)
            for j, chunk in enumerate(chunks):
                meta = {"source": p.name, "page": page["page"], "chunk_id": j}
                metas.append({"meta": meta, "text": chunk})
                embeddings.append(embed_texts([chunk])[0])

    d = len(embeddings[0])
    xb = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    faiss.write_index(index, index_path)

    with open(meta_path, "w") as f:
        json.dump(metas, f)
    print(f"Built index with {len(metas)} chunks from {len(pdfs)} PDFs.")

if __name__ == "__main__":
    build_faiss_index()
