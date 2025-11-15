# chat_agent.py
import os, requests, openai, json
openai.api_key = os.environ["OPENAI_API_KEY"]
MCP_URL = os.environ.get("MCP_URL", "http://localhost:8001")
MCP_KEY = os.environ.get("MCP_API_KEY", "devkey")

def call_mcp_search(query, top_k=3):
    r = requests.post(MCP_URL + "/mcp/search",
                      headers={"Authorization": f"Bearer {MCP_KEY}"},
                      json={"query": query, "top_k": top_k})
    r.raise_for_status()
    return r.json()["results"]

def build_system_prompt():
    return (
        "You are FinAssist, a financial-document assistant. When asked a question, "
        "first request evidence by calling the MCP search tool with the user's query; "
        "if results are returned, fetch the most relevant passages and cite them (source + page). "
        "Always include a short summary answer plus citations and, when appropriate, suggest follow-ups."
    )

def respond(user_query):
    # 1) call MCP search
    hits = call_mcp_search(user_query, top_k=4)
    # 2) prepare context chunks
    context_snippets = []
    for h in hits:
        context_snippets.append(f"Source: {h['meta']['meta']['source']} page {h['meta']['meta']['page']} score:{h['score']}\n{h['meta']['text'][:800]}")

    # 3) call OpenAI with evidence and instruction to cite
    messages = [
        {"role":"system", "content": build_system_prompt()},
        {"role":"user", "content": user_query},
        {"role":"assistant", "content": "Use the following evidence to answer and cite sources:\n" + "\n---\n".join(context_snippets)}
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini", # example - pick model available
        messages=messages,
        temperature=0.0,
        max_tokens=600
    )
    return resp["choices"][0]["message"]["content"]

if __name__ == "__main__":
    q = "Summarize the revenue recognition policy referenced in the attached 2023 financial report."
    print(respond(q))
