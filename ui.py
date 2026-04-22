import streamlit as st
import os

from src.search import RAGSearch
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

st.title("🤖 Chat with your Documents")

# ✅ Session state
if "processed" not in st.session_state:
    st.session_state.processed = False

if "rag" not in st.session_state:
    st.session_state.rag = None

# 📄 Upload file
uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt"])

# 📥 Handle upload
if uploaded_file is not None and not st.session_state.processed:

    # ❌ Delete old files
    for file in os.listdir("data"):
        file_path = os.path.join("data", file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # ✅ Save new file
    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully!")

    # 🔄 Rebuild FAISS
    with st.spinner("Processing document..."):
        docs = load_all_documents("data")
        store = FaissVectorStore("faiss_store")
        store.build_from_documents(docs)

    # 🔥 Reset RAG
    st.session_state.rag = None
    st.session_state.processed = True

    st.success("Document processed! Now ask questions.")

# 🤖 Initialize RAG
if st.session_state.rag is None:
    st.session_state.rag = RAGSearch()

# 💬 User query
query = st.text_input("Ask a question")

if st.button("Get Answer"):
    if query:
        with st.spinner("Thinking..."):
            answer = st.session_state.rag.search_and_summarize(query)
        st.write(answer)
