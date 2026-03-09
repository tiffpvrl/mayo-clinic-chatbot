from vertexai.generative_models import GenerativeModel
from src.config import LLM_MODEL

model = GenerativeModel(LLM_MODEL)

def generate_response(query: str, context: str) -> str:
    prompt = f"""
You are MayoChat, a helpful chatbot that answers patient questions about colonoscopy preparation.

Answer the question directly and clearly in 1-3 sentences.
Use only the provided context.
Answer like you are speaking directly to the patient.
If the answer is not in the context, say:
"I could not find that information in the provided instructions."

Context:
{context}

User question:
{query}

Answer:
"""
    response = model.generate_content(prompt)
    return response.text.strip()