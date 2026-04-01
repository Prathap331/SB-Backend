"""
diagnose.py — Tests Wikipedia API, NewsAPI, and Jina Reader connectivity.
Run: python diagnose.py
"""
import os, httpx
from dotenv import load_dotenv
load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
TEST_TOPIC  = "Black holes"

SEP = "─" * 55

# ── 1. Wikipedia API ──────────────────────────────────────────
print(f"\n{'='*55}")
print("TEST 1: Wikipedia API")
print(SEP)
try:
    resp = httpx.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action":"query","list":"search","srsearch":TEST_TOPIC,
                "srlimit":2,"format":"json","utf8":1},
        timeout=15,
    )
    print(f"  Status:        {resp.status_code}")
    print(f"  Content-Type:  {resp.headers.get('content-type','')}")
    print(f"  Response body: {resp.text[:300]}")
    data = resp.json()
    results = data.get("query",{}).get("search",[])
    print(f"  ✓ Parsed OK — {len(results)} results: {[r['title'] for r in results]}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# ── 2. Wikipedia — full extract ───────────────────────────────
print(f"\n{SEP}")
print("TEST 1b: Wikipedia full extract (Black hole)")
print(SEP)
try:
    resp = httpx.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action":"query","titles":"Black hole","prop":"extracts",
                "explaintext":True,"exsectionformat":"plain",
                "exintro":True,"format":"json","utf8":1},
        timeout=15,
    )
    print(f"  Status:        {resp.status_code}")
    print(f"  Body preview:  {resp.text[:300]}")
    data = resp.json()
    pages = data.get("query",{}).get("pages",{})
    for pid, page in pages.items():
        extract = page.get("extract","")
        print(f"  ✓ Extract: {len(extract.split())} words — '{extract[:120]}...'")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# ── 3. NewsAPI ────────────────────────────────────────────────
print(f"\n{SEP}")
print("TEST 2: NewsAPI.org")
print(SEP)
if not NEWSAPI_KEY:
    print("  ✗ NEWSAPI_KEY not set in .env")
else:
    try:
        resp = httpx.get(
            "https://newsapi.org/v2/everything",
            params={"q":TEST_TOPIC,"language":"en","pageSize":3,
                    "apiKey":NEWSAPI_KEY},
            timeout=15,
        )
        print(f"  Status:  {resp.status_code}")
        data = resp.json()
        print(f"  Status field: {data.get('status')}")
        print(f"  Message:      {data.get('message','(none)')}")
        print(f"  totalResults: {data.get('totalResults', 0)}")
        articles = data.get("articles", [])
        for a in articles[:2]:
            src  = a.get("source",{}).get("name","?")
            url  = a.get("url","")
            body = (a.get("content","") or a.get("description",""))[:100]
            print(f"  ✓ [{src}] {url[:60]} — '{body}'")
        if not articles:
            print(f"  Full response: {resp.text[:500]}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")

# ── 4. Jina Reader ────────────────────────────────────────────
print(f"\n{SEP}")
print("TEST 3: Jina AI Reader (r.jina.ai)")
print(SEP)
test_url = "https://en.wikipedia.org/wiki/Black_hole"
try:
    resp = httpx.get(
        f"https://r.jina.ai/{test_url}",
        headers={"Accept":"text/plain","User-Agent":"Mozilla/5.0"},
        timeout=20,
    )
    print(f"  Status:  {resp.status_code}")
    text = resp.text.strip()
    words = len(text.split())
    print(f"  Words:   {words}")
    print(f"  Preview: {text[:200]}")
    if words > 200:
        print(f"  ✓ Jina working")
    else:
        print(f"  ✗ Too short — Jina may be blocked or rate limited")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# ── 5. Raw httpx (direct Wikipedia) ──────────────────────────
print(f"\n{SEP}")
print("TEST 4: Raw httpx (direct URL fetch)")
print(SEP)
try:
    resp = httpx.get(
        "https://en.wikipedia.org/wiki/Black_hole",
        headers={"User-Agent":"Mozilla/5.0","Accept":"text/html"},
        timeout=15, follow_redirects=True,
    )
    print(f"  Status:  {resp.status_code}")
    print(f"  Words:   {len(resp.text.split())}")
    print(f"  ✓ Direct httpx working" if resp.status_code == 200 else "  ✗ Got non-200")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

print(f"\n{'='*55}")
print("Diagnostics complete.")
print('='*55)