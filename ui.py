import streamlit as st
from app import rag, extract_text_from_pdf
from embeddings import chunk_text, get_embedding
import os
import numpy as np
import faiss

st.set_page_config(page_title="Academic Assistant", page_icon="📚")

st.title("📚 Academic Assistant (RAG-Based)")

# --- Session State ---
if "answer" not in st.session_state:
    st.session_state.answer = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

# --- File Upload & Incremental Indexing ---
uploaded_files = st.file_uploader(
    "Upload new documents (PDF)", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = f"data/{uploaded_file.name}"
        if not os.path.exists(save_path):  # avoid reprocessing if file already exists
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded {uploaded_file.name}")

            # Process the new file only
            if uploaded_file.name.endswith(".pdf"):
                text = extract_text_from_pdf(save_path)
            else:
                with open(save_path, "r", encoding="utf-8") as f:
                    text = f.read()

            new_chunks = chunk_text(text)
            st.info(f"📑 {uploaded_file.name} split into {len(new_chunks)} chunks. Embedding now...")

            # Embed and add new chunks incrementally
            vectors = np.array([get_embedding(chunk, rag.embedding_model) for chunk in new_chunks])
            faiss.normalize_L2(vectors)
            rag.index.add(vectors)

            # Append new chunks to memory
            rag.chunks.extend(new_chunks)

            st.success(f"{uploaded_file.name} added to index!")
        else:
            st.warning(f"{uploaded_file.name} already exists. Skipping re-indexing.")

st.divider()

# --- Question Input ---
query = st.text_input("Ask your question:")

if st.button("Submit Question") and query:
    with st.spinner("Retrieving relevant chunks and generating answer..."):
        retrieved_chunks = rag.retrieve_chunks(query)
        context = " ".join(retrieved_chunks)
        answer = rag.generate_answer(query, context)

        st.session_state.answer = answer
        st.session_state.chunks = retrieved_chunks


# --- Display Results ---
if st.session_state.answer:
    st.subheader("Answer")
    st.write(st.session_state.answer)

    st.subheader("Supporting Evidence")
    for i, chunk in enumerate(st.session_state.chunks, 1):
        st.markdown(f"**Context {i}:** {chunk}")
