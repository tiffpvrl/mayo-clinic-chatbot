"""
FastAPI entrypoint for the MayoChat prototype.

Handles the web UI and /chat API endpoint, receives patient_id + query,
runs the RAG pipeline, and returns the generated answer.
"""

import vertexai

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.retrieval.rag import retrieve_for_query
from src.llm.generate_response import generate_response

# Initialize Vertex AI
vertexai.init(
    project="industrial-net-487818-h9",
    location="global"
)

app = FastAPI()

class ChatRequest(BaseModel):
    patient_id: str
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
    input { width: 100%; padding: 12px; font-size: 16px; margin-bottom: 10px; }
    button { margin-top: 10px; padding: 10px 14px; font-size: 16px; cursor: pointer; }
    .box { margin-top: 18px; padding: 14px; border: 1px solid #ddd; border-radius: 10px; }
    .muted { color: #666; font-size: 14px; }
    pre { white-space: pre-wrap; word-wrap: break-word; }
  </style>
</head>
<body>
  <h1>MayoChat Prototype</h1>
  <p class="muted">Enter a patient ID and a question.</p>

  <input id="patientId" placeholder="Enter patient ID..." />
  <input id="q" placeholder="Ask a bowel prep question..." />
  <button id="ask" type="button">Ask</button>

  <div class="box">
    <div class="muted">Answer</div>
    <pre id="out">-</pre>
  </div>

  <div class="box">
    <div class="muted">Debug JSON</div>
    <pre id="debug">-</pre>
  </div>

  <script>
    const patientInput = document.getElementById("patientId");
    const input = document.getElementById("q");
    const out = document.getElementById("out");
    const debugBox = document.getElementById("debug");
    const btn = document.getElementById("ask");

    async function ask() {
      const patientId = patientInput.value.trim();
      const query = input.value.trim();

      if (!patientId) {
        out.textContent = "Please enter a patient ID.";
        return;
      }

      if (!query) {
        out.textContent = "Please enter a question.";
        return;
      }

      out.textContent = "Thinking...";
      debugBox.textContent = "-";

      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            patient_id: patientId,
            query: query
          })
        });

        const text = await res.text();

        let data;
        try {
          data = JSON.parse(text);
        } catch (err) {
          out.textContent = "Backend error: " + text;
          return;
        }

        out.textContent = data.answer || data.error || "No answer returned.";
        debugBox.textContent = JSON.stringify(data.debug || data, null, 2);

      } catch (e) {
        out.textContent = "Error: " + e;
      }
    }

    btn.onclick = ask;

    input.addEventListener("keydown", function(e) {
      if (e.key === "Enter") {
        ask();
      }
    });

    patientInput.addEventListener("keydown", function(e) {
      if (e.key === "Enter") {
        ask();
      }
    });
  </script>
</body>
</html>
"""

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        patient_record, hits, context = retrieve_for_query(req.query, req.patient_id)

        if patient_record is None:
          return {"error": "Patient ID not found."}

        answer = generate_response(req.query, context)

        sources = [
            {
                "id": h.get("id"),
                "metadata": h.get("metadata", {}),
                "snippet": (h.get("document") or "")[:300]
            }
            for h in hits
        ]

        return {
            "query": req.query,
            "answer": answer,
            "debug": {
                "num_chunks": len(hits),
                "sources": sources,
                "context_preview": context[:500]
            }
        }

    except Exception as e:
        return {
            "query": req.query,
            "answer": "Something went wrong.",
            "debug": {"error": str(e)}
        }