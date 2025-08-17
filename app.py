import os
from embeddings import chunk_text, embedding_model
from rag_pipeline import AcademicAssistantRAG
import fitz

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_documents(data_dir="data"):
    docs = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if fname.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                docs.append(f.read())
        elif fname.endswith(".pdf"):
            docs.append(extract_text_from_pdf(fpath))
    return docs

# Step 1: Load documents
docs = load_documents()
chunks = []
for doc in docs:
    chunks.extend(chunk_text(doc))

# Step 2: Initialize with OpenRouter + caching
rag = AcademicAssistantRAG(
    embedding_model=embedding_model,
    chunks=chunks,
    openrouter_api_key="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",  
    model="deepseek/deepseek-chat-v3-0324:free",
    use_cache=True
)



