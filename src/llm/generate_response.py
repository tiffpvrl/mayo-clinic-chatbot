"""
LLM generation layer.

Takes the user query and combined context (patient data + retrieved documents)
and generates a grounded response using the Vertex AI model.
"""

from pathlib import Path

from vertexai.generative_models import GenerativeModel
from src.config import LLM_MODEL


def _load_fewshot_examples() -> str:
    """
    Load the few-shot examples stored alongside this file.
    If the file is missing, return an empty string so generation still works.
    """
    examples_path = Path(__file__).with_name("chatbot_fewshot_examples.md")

    try:
        return examples_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


FEWSHOT_EXAMPLES = _load_fewshot_examples()


SYSTEM_RULES = """
You are MayoChat, a helpful chatbot that answers patient questions about colonoscopy preparation.

Use the patient-specific context and knowledge-base context together.
Answer like you are speaking directly to the patient.
Be clear, calm, and grounded in the provided evidence.

The patient context may include:

RISK ASSESSMENT
- Risk tier for inadequate bowel prep: Low | Medium | High

You must use this risk tier to adjust your response style.

LOW RISK:
- Provide standard instructions
- Keep tone concise and reassuring
- Do not overemphasize risks

MEDIUM RISK:
- Emphasize the importance of following instructions carefully
- Add light reinforcement and reminders
- Clarify steps more explicitly

HIGH RISK:
- Be more directive and explicit
- Emphasize consequences of inadequate prep, such as missed findings or repeat procedure
- Encourage strict adherence to instructions
- Suggest contacting the care team when relevant and supported by the context

Critical rule:
- Risk tier affects tone, emphasis, and level of detail only
- Do not change medical facts
- Do not introduce new clinical recommendations not supported by the provided context
- Do not override retrieved evidence

If the context truly does not help answer the question, say:
"I could not find that information in the provided instructions."
"""


def generate_response(query: str, context: str) -> str:
    prompt = f"""
{SYSTEM_RULES}

Few-shot examples for tone and structure:
{FEWSHOT_EXAMPLES}

Provided context:
{context}

Patient question:
{query}

Write the final patient-facing answer only.

Before answering, follow this decision process internally:
1. Decide whether the question is answerable from the provided context.
2. If it is not answerable, say you could not find that information in the provided instructions and recommend contacting the care team if appropriate.
3. If the question is administrative, redirect the patient to their care team.
4. If the question involves severe symptoms, possible complications, or needs medical judgment, escalate to the care team.
5. If the context contains outside-hospital instructions, do not present them as the patient's required policy.
6. If the context is clearly from research or trials, answer cautiously and do not substitute it for the patient’s own instructions from their care team.

Answer:
"""

    response = GenerativeModel(LLM_MODEL).generate_content(prompt)
    text = response.text.strip()

    if not text:
        return "I could not find that information in the provided instructions."

    return text
