"""
FastAPI entrypoint for MayoChat

Handles:
- lightweight chat-style web UI
- /chat API endpoint
- patient_id + query submission
- RAG retrieval + response generation
"""

import vertexai
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.retrieval.rag import retrieve_for_query
from src.llm.generate_response import generate_response
from src.patient_data.bigquery_client import get_patient_record


# Initialize Vertex AI
vertexai.init(
    project="industrial-net-487818-h9",
    location="global",
)

app = FastAPI()


class ChatRequest(BaseModel):
    patient_id: str
    query: str


@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>MayoChat Prototype</title>
  <style>
    :root{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --border: #d9e1ea;
      --text: #1f2937;
      --muted: #6b7280;
      --brand: #0f4c81;
      --brand-soft: #eaf2f9;
      --user: #dff1ff;
      --bot: #f3f4f6;
      --danger: #b42318;
      --shadow: 0 8px 24px rgba(16, 24, 40, 0.08);
      --radius: 18px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f8fbff 0%, #f4f6fa 100%);
      color: var(--text);
    }

    .page {
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 28px 16px;
    }

    .shell {
      width: 100%;
      max-width: 800px;
      display: block;
    }

    .chat-card,
    .side-card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }

    .chat-card {
      min-height: 78vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .chat-header {
      padding: 20px 22px 16px 22px;
      border-bottom: 1px solid #eef2f7;
      background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
    }

    .title-row {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 6px;
    }

    .logo {
      width: 38px;
      height: 38px;
      border-radius: 12px;
      background: var(--brand);
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 14px;
    }

    h1 {
      margin: 0;
      font-size: 1.2rem;
      font-weight: 700;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 0.95rem;
    }

    .chat-window {
      flex: 1;
      padding: 18px;
      overflow-y: auto;
      background:
        radial-gradient(circle at top left, rgba(15,76,129,0.03), transparent 30%),
        radial-gradient(circle at bottom right, rgba(15,76,129,0.02), transparent 25%);
    }

    .messages {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .message-row {
      display: flex;
`      width: 100%;
    }

    .message-row.user {
      justify-content: flex-end;
      padding-left: 5%;
    }

    .message-row.bot {
      justify-content: flex-start;
      padding-right: 5%;
    }

    .bubble {
      max-width: 100%;
      padding: 12px 14px;
      border-radius: 18px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.97rem;
      border: 1px solid transparent;
    }

    .bubble.user {
      background: var(--user);
      border-color: #c8e7ff;
      border-bottom-right-radius: 6px;
    }

    .bubble.bot {
      background: var(--bot);
      border-color: #e5e7eb;
      border-bottom-left-radius: 6px;
    }

    .meta {
      font-size: 0.76rem;
      color: var(--muted);
      margin-bottom: 4px;
      padding: 0 4px;
    }

    .composer {
      border-top: 1px solid #eef2f7;
      padding: 14px;
      background: #fcfdff;
    }

    .composer-grid {
      display: grid;
      grid-template-columns: 170px 1fr auto;
      gap: 10px;
      align-items: start;
    }

    label {
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      color: #475467;
      margin-bottom: 6px;
    }

    input, textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 13px;
      font: inherit;
      color: var(--text);
      background: white;
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    input:focus, textarea:focus {
      border-color: #8ab4d8;
      box-shadow: 0 0 0 4px rgba(15,76,129,0.08);
    }

    textarea {
      min-height: 100px;
      max-height: 300px;
      resize: vertical;
    }

    .send-btn {
      height: 50px;
      padding: 0 18px;
      border: none;
      border-radius: 14px;
      background: var(--brand);
      color: white;
      font-weight: 700;
      font-size: 0.95rem;
      cursor: pointer;
      transition: transform 0.05s ease, opacity 0.15s ease;
    }

    .send-btn:hover { opacity: 0.95; }
    .send-btn:active { transform: translateY(1px); }
    .send-btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .side-card {
      padding: 18px;
      height: fit-content;
      position: sticky;
      top: 28px;
    }

    .side-card h2 {
      margin: 0 0 10px 0;
      font-size: 1rem;
    }

    .side-card p {
      margin: 0 0 12px 0;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }

    .debug-toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 12px 0 14px 0;
      font-size: 0.92rem;
    }

    .debug-box {
      display: none;
      margin-top: 10px;
      padding: 12px;
      border-radius: 14px;
      background: #f8fafc;
      border: 1px solid #e6ebf2;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.78rem;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 420px;
      overflow-y: auto;
    }

    .status {
      margin-top: 8px;
      font-size: 0.85rem;
      color: var(--muted);
      min-height: 20px;
    }

    .error {
      color: var(--danger);
      font-weight: 600;
    }

    .verify-btn {
      display: inline-block;
      margin: 10px 8px 0 0;
      padding: 9px 22px;
      border-radius: 999px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      border: none;
      transition: opacity 0.15s ease;
    }
    .verify-btn:hover { opacity: 0.85; }
    .verify-btn.yes { background: var(--brand); color: white; }
    .verify-btn.no  { background: #fee2e2; color: #b42318; }
    .verify-btn:disabled { opacity: 0.4; cursor: not-allowed; }

    .helper-chip {
      display: inline-block;
      margin: 6px 8px 0 0;
      padding: 8px 10px;
      border-radius: 999px;
      background: var(--brand-soft);
      color: var(--brand);
      font-size: 0.82rem;
      cursor: pointer;
      border: 1px solid #d4e3f1;
    }

    @media (max-width: 900px) {
      .shell {
        grid-template-columns: 1fr;
      }

      .side-card {
        position: static;
      }

      .composer-grid {
        grid-template-columns: 1fr;
      }

      .bubble {
        max-width: 90%;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="shell">
      <section class="chat-card">
        <div class="chat-header">
          <div class="title-row">
            <div class="logo">MC</div>
            <div>
              <h1>MayoChat Prototype</h1>
              <p class="subtitle">Patient Q&A for colonoscopy preparation</p>
            </div>
          </div>
        </div>

        <div class="chat-window" id="chatWindow">
          <div class="messages" id="messages">
            <div class="message-row bot">
              <div>
                <div class="meta">MayoChat</div>
                <div class="bubble bot">
Hello! Please enter your Patient ID in the field below, then ask a question about your colonoscopy preparation.
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="composer">
          <form id="chatForm">
            <div class="composer-grid">
              <div>
                <label for="patientId">Patient ID</label>
                <input id="patientId" placeholder="Not yet set" readonly style="background:#f3f4f6;color:#6b7280;cursor:default;" />
              </div>

              <div>
                <label for="query">Message</label>
                <textarea
                  id="query"
                  placeholder="Type your question here..."
                  required
                ></textarea>
              </div>

              <div>
                <button class="send-btn" id="sendBtn" type="submit">Send</button>
              </div>
            </div>
            <div class="status" id="status"></div>
          </form>
        </div>
      </section>
    </div>
  </div>

  <script>
    const form = document.getElementById("chatForm");
    const patientIdInput = document.getElementById("patientId");
    const queryInput = document.getElementById("query");
    const sendBtn = document.getElementById("sendBtn");
    const messages = document.getElementById("messages");
    const chatWindow = document.getElementById("chatWindow");
    const statusEl = document.getElementById("status");

    let patientIdSet = false;
    let patientVerified = false;
    let patientName = "";

    function scrollToBottom() {
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function addMessage(role, text) {
      const row = document.createElement("div");
      row.className = `message-row ${role}`;

      const wrapper = document.createElement("div");

      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = role === "user" ? "You" : "MayoChat";

      const bubble = document.createElement("div");
      bubble.className = `bubble ${role}`;
      bubble.textContent = text;

      wrapper.appendChild(meta);
      wrapper.appendChild(bubble);
      row.appendChild(wrapper);
      messages.appendChild(row);
      scrollToBottom();

      return bubble;
    }

    function addVerifyPrompt() {
      const row = document.createElement("div");
      row.className = "message-row bot";
      const wrapper = document.createElement("div");
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = "MayoChat";
      const bubble = document.createElement("div");
      bubble.className = "bubble bot";
      bubble.textContent = "Does this information look correct?";

      const yesBtn = document.createElement("button");
      yesBtn.className = "verify-btn yes";
      yesBtn.textContent = "✓  Yes, looks correct";

      const noBtn = document.createElement("button");
      noBtn.className = "verify-btn no";
      noBtn.textContent = "✗  No, something is wrong";

      function disableBtns() {
        yesBtn.disabled = true;
        noBtn.disabled = true;
      }

      yesBtn.addEventListener("click", () => {
        disableBtns();
        addMessage("bot", `Thanks for verifying, ${patientName}! How can I help you with your colonoscopy prep today? You can ask questions like:\n• Can I take my regular medications?\n• What can I eat before the procedure?\n• What should I do if I feel nauseous during prep?\n• How long does the procedure take?`);
        patientVerified = true;
        queryInput.disabled = false;
        sendBtn.disabled = false;
        queryInput.focus();
      });

      noBtn.addEventListener("click", () => {
        disableBtns();
        addMessage("bot", "We're sorry about the confusion. Please contact your doctor or care team to update your records before proceeding.");
      });

      bubble.appendChild(document.createElement("br"));
      bubble.appendChild(yesBtn);
      bubble.appendChild(noBtn);
      wrapper.appendChild(meta);
      wrapper.appendChild(bubble);
      row.appendChild(wrapper);
      messages.appendChild(row);
      scrollToBottom();
    }

    function setLoadingState(isLoading) {
      sendBtn.disabled = isLoading;
      queryInput.disabled = isLoading;
      statusEl.textContent = isLoading ? "Thinking..." : "";
    }

    document.querySelectorAll(".helper-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        queryInput.value = chip.dataset.prompt;
        queryInput.focus();
      });
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      const input = queryInput.value.trim();
      if (!input) return;
      if (patientIdSet && !patientVerified) return;

      addMessage("user", input);
      queryInput.value = "";
      queryInput.style.height = "52px";

      // First message sets the patient ID — validate against BigQuery
      if (!patientIdSet) {
        setLoadingState(true);
        try {
          const res = await fetch("/validate-patient", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ patient_id: input })
          });
          const data = await res.json();
          console.log("[validate-patient] Response:", data);
          if (!data.valid) {
            console.log("[validate-patient] Not valid. Error:", data.error);
            addMessage("bot", `Patient ID "${input}" was not found. Please check your ID and try again.`);
            return;
          }
          console.log("[validate-patient] Valid. Building summary...");
          patientIdInput.value = input;
          patientIdSet = true;
          const s = data.summary;
          console.log("[validate-patient] Summary object:", s);
          patientName = s.patient_name || s.patient_id;
          const info = [
            `Patient ID:            ${s.patient_id}`,
            `Name:                  ${s.patient_name}`,
            `Sex at Birth:          ${s.sex_at_birth}`,
            `Gender Identity:       ${s.gender_identity}`,
            `Comorbidities:         ${s.comorbidity_descriptions}`,
            `Current Medications:   ${s.current_medications}`,
            `Bowel Prep Start:      ${s.bowel_prep_start}`,
            `Bowel Prep End:        ${s.bowel_prep_end}`,
            `Colonoscopy Date/Time: ${s.colonoscopy_datetime}`,
            `Indication:            ${s.colonoscopy_indication}`,
            `Chief Complaint:       ${s.chief_complaint}`,
            `Prep Agent:            ${s.prep_agent}`,
          ].join("\\n");
          try {
            addMessage("bot", `Here is the record on file:\n\n${info}`);
            console.log("[validate-patient] EHR message added.");
            queryInput.disabled = true;
            sendBtn.disabled = true;
            addVerifyPrompt();
            console.log("[validate-patient] Verify prompt added.");
          } catch (renderErr) {
            console.error("[validate-patient] Render error:", renderErr);
            addMessage("bot", "Patient verified but could not display record. Please try again.");
          }
        } catch (err) {
          addMessage("bot", "Could not verify patient ID. Please try again.");
        } finally {
          setLoadingState(false);
        }
        return;
      }

      const patient_id = patientIdInput.value.trim();
      const botBubble = addMessage("bot", "...");
      setLoadingState(true);

      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ patient_id, query: input })
        });

        const data = await res.json();

        if (data.error) {
          botBubble.textContent = data.error;
          botBubble.parentElement.parentElement.querySelector(".meta").textContent = "System";
          statusEl.innerHTML = '<span class="error">Request completed with an error.</span>';
        } else {
          botBubble.textContent = data.answer || "No answer returned.";
        }
      } catch (err) {
        botBubble.textContent = "Something went wrong while contacting the server.";
        botBubble.parentElement.parentElement.querySelector(".meta").textContent = "System";
        statusEl.innerHTML = '<span class="error">Network or server error.</span>';
      } finally {
        setLoadingState(false);
        scrollToBottom();
      }
    });

    queryInput.addEventListener("input", () => {
      queryInput.style.height = "auto";
      queryInput.style.height = Math.min(queryInput.scrollHeight, 180) + "px";
    });

    queryInput.addEventListener("keydown", function(e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();   // prevents newline
        form.requestSubmit(); // submits the form
      }
    });
  </script>
</body>
</html>
    """


class ValidateRequest(BaseModel):
    patient_id: str


@app.post("/validate-patient")
def validate_patient(req: ValidateRequest):
    try:
        print(f"[validate-patient] Looking up patient_id: {req.patient_id}")
        record = get_patient_record(req.patient_id)
        print(f"[validate-patient] Record found: {record is not None}")
        if record is None:
            return {"valid": False}
        def fmt_dt(val):
            if val is None:
                return "N/A"
            return str(val).replace("T", " ").split(".")[0]
        summary = {
            "patient_id":               record.get("patient_id", "N/A"),
            "patient_name":             record.get("patient_name", "N/A"),
            "sex_at_birth":             record.get("sex_at_birth", "N/A"),
            "gender_identity":          record.get("gender_identity", "N/A"),
            "comorbidity_descriptions": record.get("comorbidity_descriptions", "N/A"),
            "current_medications":      record.get("current_medications", "N/A"),
            "bowel_prep_start":         fmt_dt(record.get("bowel_prep_start_datetime")),
            "bowel_prep_end":           fmt_dt(record.get("bowel_prep_end_datetime")),
            "colonoscopy_datetime":     fmt_dt(record.get("colonoscopy_datetime")),
            "colonoscopy_indication":   record.get("colonoscopy_indication", "N/A"),
            "chief_complaint":          record.get("chief_complaint", "N/A"),
            "prep_agent":               record.get("prep_agent", "N/A"),
        }
        print(f"[validate-patient] Returning summary: {summary}")
        return {"valid": True, "summary": summary}
    except Exception as e:
        print(f"[validate-patient] ERROR: {e}")
        return {"valid": False, "error": str(e)}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        result = retrieve_for_query(req.query, req.patient_id)

        if result.patient_record is None:
            return {"error": "Patient ID not found."}

        print("\n===== RAG CONTEXT =====")
        print(result.combined_context)
        print("=======================\n")

        answer = generate_response(req.query, result.combined_context)

        sources = [
            {
                "id": h.get("id"),
                "metadata": h.get("metadata", {}),
                "snippet": (h.get("document") or "")[:300],
            }
            for h in result.clinical_hits
        ]

        return {
            "query": req.query,
            "answer": answer,
            "debug": {
                "num_chunks": len(result.clinical_hits),
                "sources": sources,
                "context_preview": result.combined_context[:500],
            },
        }

    except Exception as e:
        return {
            "query": req.query,
            "answer": "Something went wrong.",
            "debug": {"error": str(e)},
        }