from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MayoChat (Prototype)</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 40px; max-width: 720px; }
    input { width: 100%; padding: 12px; font-size: 16px; }
    button { margin-top: 10px; padding: 10px 14px; font-size: 16px; cursor: pointer; }
    .box { margin-top: 18px; padding: 14px; border: 1px solid #ddd; border-radius: 10px; }
    .muted { color: #666; font-size: 14px; }
    pre { white-space: pre-wrap; word-wrap: break-word; }
  </style>
</head>
<body>
  <h1>MayoChat Prototype</h1>
  <p class="muted">Type a question and press Enter.</p>

  <input id="q" placeholder="Ask a bowel prep question…" />
  <button id="ask">Ask</button>

  <div class="box">
    <div class="muted">Response</div>
    <pre id="out">—</pre>
  </div>

  <script>
    const input = document.getElementById("q");
    const out = document.getElementById("out");
    const btn = document.getElementById("ask");

    async function ask() {
      const query = input.value.trim();
      if (!query) return;
      out.textContent = "Thinking…";

      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query })
        });

        const data = await res.json();
        out.textContent = JSON.stringify(data, null, 2);
      } catch (e) {
        out.textContent = "Error: " + e;
      }
    }

    btn.addEventListener("click", ask);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") ask();
    });
  </script>
</body>
</html>
"""

@app.post("/chat")
def chat(req: ChatRequest):
    # Placeholder response for now (we’ll wire RAG/Vertex later)
    return {
        "judgement": 1,
        "evidence": [],
        "argumentation": "Stub response. RAG not connected yet.",
        "query": f"You asked: {req.query}"
    }