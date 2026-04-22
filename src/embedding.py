from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        if not documents:
            print("[ERROR] No documents received for chunking.")
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)

        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")

        # DEBUG
        if len(chunks) > 0:
            print("[DEBUG] First chunk preview:", chunks[0].page_content[:100])
        else:
            print("[ERROR] No chunks created!")

        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        if not chunks:
            print("[ERROR] No chunks to embed.")
            return np.array([])

        texts = [chunk.page_content for chunk in chunks if chunk.page_content.strip() != ""]

        if not texts:
            print("[ERROR] All chunks are empty.")
            return np.array([])

        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(texts, show_progress_bar=True)

        if len(embeddings) == 0:
            print("[ERROR] Embeddings generation failed.")
            return np.array([])

        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings
