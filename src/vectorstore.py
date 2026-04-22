import faiss
import numpy as np

class FaissVectorStore:
    def __init__(self):
        self.index = None
        self.texts = []

    def build(self, embeddings, texts):
        if len(embeddings) == 0:
            raise ValueError("No embeddings found")

        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.texts = texts

    def search(self, query_embedding, top_k=3):
        D, I = self.index.search(query_embedding, top_k)

        results = []
        for idx in I[0]:
            results.append(self.texts[idx])

        return results
