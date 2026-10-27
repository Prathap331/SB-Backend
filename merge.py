from qdrant_client import QdrantClient

client = QdrantClient(url="http://37.27.101.243:6333", timeout=60)


cols = client.get_collections().collections

for c in cols:
    print(c.name, client.count(c.name, exact=True).count)