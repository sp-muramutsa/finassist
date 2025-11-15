# FinAssist LLM App


This repository contains a minimal, runnable prototype of the FinAssist LLM app that:
- Ingests PDF documents and builds a FAISS vector index using OpenAI embeddings
- Exposes an MCP-like server that provides search/fetch tools
- Provides a chat API which orchestrates OpenAI + MCP to answer financial-document questions with provenance
- Includes a minimal static frontend (index.html) to interact with the chat API


---


## File layout


- README.md
- requirements.txt
- .env.example
- ingest.py
- mcp_server.py
- chat_agent.py
- api.py
- static/index.html
- sample_docs/ (place PDFs here)


---


### FILE: README.md
```markdown
# FinAssist - Local Prototype


This repo is a minimal local prototype of the FinAssist app described earlier. It uses OpenAI embeddings & chat API, a local FAISS index for vector search, and a lightweight MCP-style server.


## Prereqs
- Python 3.10+
- pip
- (Optional) Tesseract if you need OCR for scanned PDFs


## Setup
1. Clone this repo.
2. Copy `.env.example` to `.env` and add your keys:
- OPENAI_API_KEY
- MCP_API_KEY (choose a dev key, e.g. "devkey")
- MCP_URL (defaults used by scripts)


3. Create and activate a virtualenv, then install:


```bash
python -m venv .venv
source .venv/bin/activate # windows: .venv\Scripts\activate
pip install -r requirements.txt
```


4. Put PDFs you want indexed into `sample_docs/`.


5. Build the FAISS index (this calls OpenAI embeddings):


```bash
export OPENAI_API_KEY="sk-..."
python ingest.py
```


This writes `faiss.index` and `meta.json` in the repo root.


6. Start the MCP server:


```bash
---