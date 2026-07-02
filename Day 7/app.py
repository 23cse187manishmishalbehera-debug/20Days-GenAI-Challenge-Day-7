from pathlib import Path
import pickle

import faiss
import numpy as np
from flask import Flask, render_template_string, request
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "vector_store" / "faiss_index.index"
DOCS_PATH = BASE_DIR / "vector_store" / "documents.pkl"

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(str(INDEX_PATH))

with DOCS_PATH.open("rb") as handle:
    documents = pickle.load(handle)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Semantic Search</title>
  <style>
    body {
      margin: 0;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #0f172a, #2563eb);
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .card {
      background: rgba(15, 23, 42, 0.88);
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 20px;
      padding: 32px;
      width: min(640px, 100%);
      box-shadow: 0 20px 45px rgba(0, 0, 0, 0.25);
      backdrop-filter: blur(10px);
    }
    h1 {
      margin-top: 0;
      margin-bottom: 8px;
      font-size: 2rem;
    }
    p {
      color: #cbd5e1;
      line-height: 1.5;
    }
    .topic-grid {
      display: grid;
      gap: 16px;
      margin: 22px 0;
    }
    .topic-card {
      background: rgba(148, 163, 184, 0.08);
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 16px;
      padding: 18px;
      line-height: 1.6;
      text-align: center;
      font-weight: 600;
    }
    .topic-card h3 {
      margin: 0;
      color: #f8fafc;
      font-size: 1.05rem;
    }
    form {
      display: flex;
      gap: 10px;
      margin: 20px 0 24px;
      flex-wrap: wrap;
    }
    input[type="text"] {
      flex: 1;
      min-width: 220px;
      padding: 12px 14px;
      border: 1px solid #64748b;
      border-radius: 10px;
      font-size: 1rem;
      outline: none;
    }
    button {
      background: linear-gradient(135deg, #38bdf8, #2563eb);
      color: white;
      border: none;
      border-radius: 10px;
      padding: 12px 18px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
    }
    .result-box {
      margin-top: 10px;
      padding: 16px;
      border-radius: 12px;
      background: #111827;
      border: 1px solid #38bdf8;
      color: #f8fafc;
      line-height: 1.6;
    }
    .hint {
      margin-top: 12px;
      font-style: italic;
      color: #94a3b8;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Semantic Search</h1>
    <p>Ask a question and the app will retrieve the most relevant document from your FAISS index.</p>
    <form method="POST">
      <input type="text" name="query" placeholder="Ask a question like: How do machines learn?" required>
      <button type="submit">Search</button>
    </form>
    {% if result %}
      <h3>Most Relevant Result</h3>
      <div class="result-box">{{ result }}</div>
    {% else %}
      <p class="hint">Try searching for something like “How do machines learn?”</p>
    {% endif %}
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            embedding = model.encode([query])
            embedding = np.array(embedding).astype("float32")
            _, indexes = index.search(embedding, 1)
            result = documents[indexes[0][0]]
    return render_template_string(HTML, result=result)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
