import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "documents"
INDEX_DIR = BASE_DIR / "vector_store"


def build_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    documents = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(text)

    if not documents:
        raise ValueError("No documents found in the documents folder.")

    embeddings = model.encode(documents)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss_index.index"))

    with (INDEX_DIR / "documents.pkl").open("wb") as handle:
        pickle.dump(documents, handle)

    print("Vector Database Created Successfully")


if __name__ == "__main__":
    build_index()
