from sentence_transformers import SentenceTransformer
import nltk

# Download NLTK data
nltk.download('punkt')

# Load pretrained sentence embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits text into overlapping chunks by word count.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_embedding(text, model=embedding_model):
    """
    Generates embedding vector for text.
    """
    return model.encode(text)
