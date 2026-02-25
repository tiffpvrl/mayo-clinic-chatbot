from sentence_transformers import SentenceTransformer
from openai import OpenAI

class Embedding:
    def __init__(self, model_type: str):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI()  # reads OPENAI_API_KEY from env
            self.openai_model = "text-embedding-3-large"
        elif model_type == "pubmed_bert": # medical literature
            self.model = SentenceTransformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        elif model_type == "bio_bert": 
            self.model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.1')
        elif model_type == "mini_lm": # smaller embedding
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.model_type == "openai":
            response = self.client.embeddings.create(
                model=self.openai_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        else:
            return self.model.encode(texts, show_progress_bar=True).tolist()





