import os
import requests
import numpy as np
import faiss
import pickle
from embeddings import get_embedding


# Saves the ind
def save_index(index, chunks, path="faiss_index"):
    faiss.write_index(index, f"{path}.index")
    with open(f"{path}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("Index Saved")

def load_index(path="faiss_index"):
    index = faiss.read_index(f"{path}.index")
    with open(f"{path}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print("Complete")
    return index, chunks

class AcademicAssistantRAG:
    def __init__(self, embedding_model, chunks, openrouter_api_key, model="", use_cache=True, cache_path="faiss_index"):
        self.embedding_model = embedding_model
        self.chunks = chunks
        self.model = model
        self.api_key = openrouter_api_key
        self.cache_path = cache_path

        if use_cache and os.path.exists(f"{cache_path}.index") and os.path.exists(f"{cache_path}_chunks.pkl"):
            print("Loading FAISS index from cache...")
            self.index, saved_chunks = load_index(cache_path)

            # Detect new chunks
            new_chunks = [c for c in self.chunks if c not in saved_chunks]
            if new_chunks:
                print(f"Found {len(new_chunks)} new chunks. Adding to index...")
                self._add_new_chunks(new_chunks)
                self.chunks.extend(new_chunks)
                save_index(self.index, self.chunks, cache_path)
            else:
                self.chunks = saved_chunks
        else:
            print("Building new FAISS index...")
            self.index = self._build_faiss_index()
            save_index(self.index, self.chunks, cache_path)

    def _build_faiss_index(self):
        sample_vec = get_embedding(self.chunks[0], self.embedding_model)
        dim = sample_vec.shape[0]

        index = faiss.IndexFlatIP(dim)  # cosine similarity
        vectors = np.array([get_embedding(chunk, self.embedding_model) for chunk in self.chunks])
        faiss.normalize_L2(vectors)
        index.add(vectors)
        return index

    def _add_new_chunks(self, new_chunks):
        vectors = np.array([get_embedding(chunk, self.embedding_model) for chunk in new_chunks])
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def retrieve_chunks(self, query, top_k=8):
        query_vec = get_embedding(query, self.embedding_model).reshape(1, -1)
        faiss.normalize_L2(query_vec)
        distances, indices = self.index.search(query_vec, top_k)
        return [self.chunks[i] for i in indices[0]]

    def generate_answer(self, question, context, temperature=0.3):
        prompt = f"""You are an academic assistant. Use the context below to answer the question.
Answer factually and clearly, avoid unrelated details.

Context:
{context}

Question: {question}
Answer:"""

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"OpenRouter Error {response.status_code}: {response.text}")

    def rebuild_index(self):
        """Force rebuild the FAISS index from current chunks"""
        print("Rebuilding FAISS index from scratch...")
        self.index = self._build_faiss_index()
        save_index(self.index, self.chunks, self.cache_path)
