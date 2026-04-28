import fitz
import re
import uuid
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

url= os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def detect_lang(text: str) -> str:
    try:
        return detect(text[:1000])
    except:
        return "unknown"

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)

        if chunk.strip():
            chunks.append(chunk)

        i += (chunk_size - overlap)

    return chunks

def create_normalised_chunks(chunks, language):
    result = []

    for idx, chunk in enumerate(chunks):
        result.append({
            "chunk_id": str(uuid.uuid4()),
            "channel_id": 123,  
            "source_type": "script_upload",
            "language_code": language,
            "text": chunk,
            "chunk_index": idx,
            "is_canonical": True, 
            "metadata": {}
        })

    return result


def generate_embeddings(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts)

    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb.tolist() 
    print(chunks[0]["embedding"][:5])
    return chunks


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query, chunks, top_k=3):
    query_emb = model.encode(query)

    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_emb, chunk["embedding"])
        scored.append((score, chunk["text"]))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]


def process_pdf(file_input):
    text = ""

    if isinstance(file_input, bytes):
        doc = fitz.open(stream=file_input, filetype="pdf")
    else:
        doc = fitz.open(file_input)

    with doc:
        for page in doc:
            text += page.get_text()

    text = clean_text(text)
    language = detect_lang(text)

    raw_chunks = chunk_text(text)

    normalised_chunks = create_normalised_chunks(
        chunks=raw_chunks,
        language=language
    )

    embedded_chunks = generate_embeddings(normalised_chunks)

    return embedded_chunks



# if __name__ == "__main__":
#     chunks = process_pdf("sample_script.pdf")
#     supabase.table('channel_memory').insert(chunks).execute()
#     print("saved in db")
    
#     results = search("What is Rahul doing?", chunks)

#     print("\n--- SEARCH RESULTS ---\n")
#     for score, text in results:
#         print(f"Score: {score:.4f}")
#         print(text[:200])
#         print("------")