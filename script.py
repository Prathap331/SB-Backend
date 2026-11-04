
import requests
import os
import time
import json
from pathlib import Path

API_KEY = os.getenv("SERPAPI_API_KEY")

def fetch_images(query, limit=2, folder="images"):

    os.makedirs(folder, exist_ok=True)

    response = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google_images",
            "q": query,
            "api_key": API_KEY,
            "ijn": 0
        },
        timeout=30
    )

    response.raise_for_status()

    images = response.json().get("images_results", [])

    downloaded = 0
    results = []

    for image in images:

        if downloaded >= limit:
            break

        image_url = image.get("original")

        if not image_url:
            continue

        width = image.get("original_width", 0)
        height = image.get("original_height", 0)

        if width <= height:
            continue

        try:
            img = requests.get(
                image_url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30
            )

            img.raise_for_status()

            if not img.headers.get("Content-Type", "").startswith("image/"):
                continue

            extension = Path(img.url.split("?")[0]).suffix.lower()

            if extension not in [".jpg", ".jpeg", ".png", ".webp"]:
                extension = ".jpg"

            filepath = os.path.join(
                folder,
                f"image_{downloaded + 1}{extension}"
            )

            with open(filepath, "wb") as f:
                f.write(img.content)

            downloaded += 1

            results.append({
                "image": filepath,
                "original_url": image_url,
                "thumbnail_url": image.get("thumbnail"),
                "title": image.get("title"),
                "source": image.get("source"),
                "width": width,
                "height": height
            })

            print(f"\n[+] Downloaded: {filepath}")
            print(f"URL: {image_url}")
            print(f"Dimensions: {width}x{height}")

            time.sleep(3)

        except requests.RequestException as e:
            print(f"[-] Failed: {e}")
            time.sleep(3)

    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nDownloaded {downloaded}/{limit} landscape images")


fetch_images(
    query="Quit India protest Bombay crowd",
    limit=2,
    folder="indian_history"
)























# from qdrant_client import QdrantClient

# client = QdrantClient(url="http://37.27.101.243:6333", timeout=60)


# cols = client.get_collections().collections

# for c in cols:
#     print(c.name, client.count(c.name, exact=True).count)
