# FinAssist – Agentic Finance AI (FastAPI + Pydantic + Docker + K8s)


This project is a small, end-to-end “agentic” finance assistant. It ingests PDFs into a FAISS vector index using OpenAI embeddings, serves MCP-style retrieval tools, and exposes a chat endpoint that orchestrates retrieval + LLM reasoning with citations.


## Highlights (match your resume claims)
- **FastAPI + Pydantic** for typed schemas, validation, and clean API contracts.
- **Agentic flow**: retrieval over financial docs via FAISS + OpenAI embeddings, then LLM reasoning with citations.
- **Docker/Kubernetes ready**: containerized service with sample Deployment/Service manifests.
- **Configurable** via `.env` and typed settings; health check, CORS enabled.


## Architecture at a glance
- `ingest.py` – builds `faiss.index` + `meta.json` from PDFs in `sample_docs/`.
- `mcp_server.py` – MCP-style search/fetch router backed by FAISS + OpenAI embeddings.
- `chat_agent.py` – orchestrates MCP search + OpenAI chat completion with citations.
- `api.py` – FastAPI entrypoint exposing `/chat`, `/health`, and mounting MCP routes.
- `models.py` – Pydantic schemas for requests/responses and search hits.
- `k8s/` – example Deployment/Service/PVC manifests for Kubernetes.
- `Dockerfile` – container image for local or cluster deploys.


## Prerequisites
- Python 3.10+
- OpenAI API key
- `pip` and (optional) Docker + kubectl for deployment


## Quickstart (local)
1) **Install deps**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) **Configure env**
```bash
cp .env.example .env
# edit .env with your OPENAI_API_KEY (and adjust MCP_API_KEY if desired)
```

3) **Add documents & ingest**
```bash
mkdir -p sample_docs
# drop your finance PDFs into sample_docs/
python ingest.py
```

4) **Run the API**
```bash
uvicorn api:app --reload --port 8000
```

5) **Test**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the revenue recognition policy"}'
```


## Docker
Build and run locally:
```bash
docker build -t finassist:local .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e MCP_API_KEY=devkey \
  -v $(pwd)/faiss.index:/app/faiss.index \
  -v $(pwd)/meta.json:/app/meta.json \
  finassist:local
```


## Kubernetes (sample manifests)
1) Create secrets for your keys:
```bash
kubectl create secret generic finassist-secrets \
  --from-literal=OPENAI_API_KEY=your-key \
  --from-literal=MCP_API_KEY=devkey
```
2) Apply storage + deploy + service:
```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```
Mount `faiss.index` and `meta.json` into the PVC (or bake them into the image) before rolling out.


## Endpoints
- `GET /health` – readiness/liveness check.
- `POST /chat` – body: `ChatRequest` (question, top_k, temperature, model?); returns `ChatResponse` with answer + citations.
- `POST /mcp/search` – body: `SearchRequest`; returns ranked FAISS hits.
- `POST /mcp/fetch` – body: `ChunkMeta`; returns the full chunk text.


## Interview-ready talking points
- **Data validation:** Pydantic schemas (`models.py`) guard requests/responses and MCP search hits.
- **Agentic flow:** retrieval (FAISS + embeddings) -> LLM reasoning with explicit evidence prompts and citations.
- **Deployment:** Dockerfile for containerization; Kubernetes manifests (Deployment/Service/PVC) plus secrets for keys; readiness/liveness probes via `/health`.
- **Extensibility:** swap vector DB, change models via env (`OPENAI_MODEL`), add tools by extending MCP router.


## Notes
- Ingestion must run before serving so `faiss.index` and `meta.json` exist.
- Default model is `gpt-4o-mini`; override with `OPENAI_MODEL` env or `ChatRequest.model`.



