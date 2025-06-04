from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Dummy data
texts = [
    "Japan is famous for cherry blossoms and cutting-edge tech.",
    "India is home to the world’s largest democracy and spicy food.",
    "China has the Great Wall that’s visible from space (almost!).",
    "Thailand is known for its beautiful beaches and delicious street food.",
    "South Korea loves K-pop and high-speed internet.",
    "France is the birthplace of fine wine and stylish fashion.",
    "Germany is famous for its cars and hearty sausages.",
    "Italy is where pizza and art were born.",
    "Spain dances to the rhythm of flamenco and sunshine.",
    "Sweden is known for IKEA and midnight sun summers."
]

# Create embeddings
embeddings = model.encode(texts)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

data = texts

# FastAPI
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/search")
def search(req: QueryRequest):
    query_embedding = model.encode([req.query])
    distances, indices = index.search(np.array(query_embedding), req.top_k)
    results = [data[i] for i in indices[0]]
    return {"query": req.query, "results": results}

# in terminal, enter 'uvicorn main:app --reload'
# http://127.0.0.1:8000/docs