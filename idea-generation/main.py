# import re
# import time
# import requests
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('all-MiniLM-L6-v2')

# TOPICS = [
#     "AI automation",
#     "productivity hacks",
# ]

# HEADERS = {"User-Agent": "Mozilla/5.0 (content-idea-bot/1.0)"}
# REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"


# def chunk_text(text, chunk_size=300, overlap=50):
#         words = text.split()
#         chunks = []
#         i = 0
#         while i < len(words):
#             chunk_words = words[i:i + chunk_size]
#             chunk = " ".join(chunk_words)
#             if chunk.strip():
#                 chunks.append(chunk)
#             i += (chunk_size - overlap)
#         return chunks


# def generate_embeddings(chunks):
#     texts = [c["text"] for c in chunks]
#     embeddings = model.encode(texts)

#     for i, emb in enumerate(embeddings):
#         chunks[i]["embedding"] = emb.tolist() 
#     print(chunks[0]["embedding"][:5])
#     return chunks



# def clean_text(sentence):

#     remove_words = {"to","with","then","a","an","the","is","are","was","were","be","been","being",
#     "in","on","at","for","of","by","from","up","down","out","over","under",
#     "and","or","but","if","because","as","until","while","although",
#     "i","me","my","you","your","he","him","his","she","her","it","we","they","them",
#     "this","that","these","those",
#     "am","do","does","did","doing","have","has","had","having",
#     "can","could","will","would","shall","should","may","might","must",
#     "not","no","nor","so","too","very",
#     "just","now","also","even","only","again","once"
#     }

#     sentence = re.sub(r'[^\w\s]', '', sentence)

#     sentence = sentence.lower()

#     return " ".join(
#         word for word in sentence.split()
#         if word not in remove_words
#     )

# def fetch_titles(topic, limit=50, time_range="month"):
#     titles = []
#     after = None

#     while len(titles) < limit:
#         params = {"q": topic, "sort": "top", "t": time_range, "limit": 100, "type": "link"}
#         if after:
#             params["after"] = after

#         resp = requests.get(REDDIT_SEARCH_URL,params=params, headers=HEADERS, timeout=15)
#         resp.raise_for_status()

#         data     = resp.json().get("data", {})
#         children = data.get("children", [])
#         after    = data.get("after")

#         for child in children:
#             title = (child.get("data") or {}).get("title", "").strip()
#             if title:
#                 titles.append(re.sub(r"\s+", " ", title))

#         if not after or not children:
#             break

#         time.sleep(1.5)
#     return titles[:limit]


# if __name__ == "__main__":
#     for topic in TOPICS:
#         for i, title in enumerate(fetch_titles(topic), 1):
#             cleaned_text = clean_text(title)
#             chunked_text = chunk_text(cleaned_text)
#             embeddings = generate_embeddings(chunked_text)
#             print(embeddings)