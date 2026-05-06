import re
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer
from supabase import create_client
import os
from dotenv import load_dotenv
import hashlib
import pymupdf4llm

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

url= os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

def make_chunk_id(userId, idx, chunk):
    return f"{userId}_{idx}_{hashlib.md5(chunk.encode()).hexdigest()[:10]}"


def clean_text(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'#+ ', '', text)  
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

def create_normalised_chunks(chunks, language,userId):
    result = []

    for idx, chunk in enumerate(chunks):
        result.append({
            "chunk_id": make_chunk_id(userId, idx, chunk),
            "channel_id": 123,  
            "source_type": "script_upload",
            "language_code": language,
            "text": chunk,
            "chunk_index": idx,
            "is_canonical": True, 
            "metadata": {},
            "userId" : userId
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


def extract_pdf_text(file_input):
    text = pymupdf4llm.to_markdown(file_input)
    return text


def process_pdf(file_input, userId):
    text = extract_pdf_text(file_input)

    text = clean_text(text)
    language = detect_lang(text)

    raw_chunks = chunk_text(text)

    normalised_chunks = create_normalised_chunks(
        chunks=raw_chunks,
        language=language,
        userId=userId
    )

    embedded_chunks = generate_embeddings(normalised_chunks)

    return embedded_chunks

