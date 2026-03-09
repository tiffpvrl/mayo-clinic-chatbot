"""
LLM generation layer.

Takes the user query and combined context (patient data + retrieved documents)
and generates a grounded response using the Vertex AI model.
"""

from vertexai.generative_models import GenerativeModel
from src.config import LLM_MODEL

model = GenerativeModel(LLM_MODEL)

def generate_response(query: str, context: str) -> str:
    prompt = f"""
You are MayoChat, a helpful chatbot that answers patient questions about colonoscopy preparation.

Answer the question directly and clearly.
Use the patient-specific context and knowledge-base context together.
Answer like you are speaking directly to the patient.
If the context truly does not help answer the question, say:
"I could not find that information in the provided instructions."

Context:
{context}

User question:
{query}

Answer:
"""
    response = model.generate_content(prompt)
    return response.text.strip()