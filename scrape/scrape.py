from bs4 import BeautifulSoup
from supabase import create_client
from sentence_transformers import SentenceTransformer
import numpy as np
from pytrends.request import TrendReq
from google import genai
import os
import re
import requests
# from google.genai import types as genai_types
# import time

gnews_key = os.getenv("GnewsApi")
newsdata_api_key = os.getenv("Newsdata_api_key")
google_api_key = os.getenv("GOOGLE_API_KEY")

print(google_api_key)

url= os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

pytrends = TrendReq(hl='en-US', tz=360)

supabase = create_client(url, key)

client = genai.Client(api_key=google_api_key)

model = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


bbc = "https://www.bbc.com/"

async def ai_itellengence(article):
    print("ai ready")
    prompt = f"""System: You are an expert Viral Content Strategist. Your goal is to identify high-viral-potential news and turn it into engaging social media content.

    User:
    I have a news headline.

    [DATA]
    Title: {article.get("tittle")}
    [/DATA]

    STEP 0: VELOCITY SCORING
    Estimate a "Velocity Score" from 0–100 based on:
    - How fast this topic could trend
    - Public interest potential
    - Emotional impact
    - Shareability

    Rules:
    - 80–100 → Highly viral / trending
    - 50–79 → Moderate interest
    - Below 50 → Low impact / likely filler

    STEP 1: QUALITY FILTER
    - If score < 50 → respond ONLY with "SKIP"
    - Otherwise continue

    STEP 2: OUTPUT FORMAT

    1. VELOCITY SCORE:
    Give the score you estimated (just a number)

    2. THE HOOK:
    Create 3 scroll-stopping opening lines:
    - One Negative/Warning hook
    - One Curiosity hook
    - One Direct Value hook

    3. THE VIRAL ANGLE:
    - Identify emotion (Awe, Anger, Anxiety, Amusement)
    - Explain why people will share this

    4. THE SCRIPT:
    - 0-5s: Hook
    - 5-15s: Main reveal
    - 15-25s: Surprising fact
    - 25-30s: Engagement question

    5. KEYWORDS & TAGS:
    Provide 5 trending hashtags

    Tone: Fast-paced, engaging, authoritative.
    """
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )

        data = {
           "tittle" : article.get("tittle"),
           "summary":response.text
        }

        supabase.table("content_radar").upsert(data,on_conflict="tittle").execute()
        print(response.text)
        print("Saved: ai_itellengence")
        return {"response": response.text}
    except Exception as e:
        print(e)



def clean_keywords(text):
    stop_words = {
    "the","is","a","an","in","on","at","to","and","but","says",
    "did","not","was","were","of","for","with","by","as","from","this","that",
    "it", "its", "they", "them", "their", "who", "which", "what", "where", "when",
    "how", "why", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "too", "very", "can", "will", "just", "should",
    "now", "about", "after", "before", "during", "under", "over", "between", 
    "into", "through", "breaking", "latest", "report", "update", "watch",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "up", "down", "out", "off", 
    "over", "under", "again", "further", "then", "once"
    "today", "yesterday", "tomorrow", "daily", "weekly", "month", "year", "years", 
    "time", "days", "hours", "minutes", "new", "old", "first", "last", "next", 
    "recent", "past",
    "report", "reports", "reported", "breaking", "update", "latest", "exclusive", 
    "official", "source", "sources", "according", "confirmed", "told", "claims", 
    "claimed", "details", "video", "watch", "live", "shared", "posted",
    "it", "its", "they", "them", "their", "who", "whom", "which", "what", "where", 
    "when", "how", "why", "all", "any", "both", "each", "few", "more", "most", 
    "other", "some", "such", "no", "nor", "too", "very", "can", "will", "just", 
    "should", "now"
       
   }

    words = re.findall(r"[a-zA-Z]+", text.lower())

    keywords = [w for w in words if w not in stop_words]

    return keywords[:5]


# LAST_CALL = 0
# MIN_DELAY = 30

# def rate_limit():
#     global LAST_CALL
#     now = time.time()
#     elapsed = now - LAST_CALL

#     if elapsed < MIN_DELAY:
#         sleep_time = MIN_DELAY - elapsed + random.uniform(1, 3)
#         print(f"Sleeping {round(sleep_time, 2)}s to avoid rate limit...")
#         time.sleep(sleep_time)

#     LAST_CALL = time.time()


# def fetch_trends(pytrends, kw_list, retries=4):
#     for attempt in range(retries):
#         try:
#             rate_limit()

#             pytrends.build_payload(
#                 kw_list[:5],
#                 cat=0,
#                 timeframe='today 12-m',
#                 geo='IN',
#                 gprop=''
#             )

#             df = pytrends.interest_over_time()
#             return df

#         except Exception as e:
#             if "429" in str(e):
#                 wait = (2 ** attempt) * 5 + random.uniform(1, 3)
#                 print(f"429 error. Retry {attempt+1} in {round(wait,2)}s...")
#                 time.sleep(wait)
#             else:
#                 print("Unexpected error:", e)
#                 return None

#     return None


async def calculate_cos(article):

    # text = article.get("tittle", "")
    # kw_list = clean_keywords(text)

    # if len(kw_list) == 0:
    #     return

    # df = fetch_trends(pytrends, kw_list)

    # print(df)

    try:
        # latest = df.iloc[-1].drop("isPartial", errors="ignore")
        # values = latest.values.tolist()

        # if not values:
        #     return

        # score = max(values) 
        await ai_itellengence(article=article)
        print("ai answer saved")

    except Exception as e:
        print("Processing error:", e)

async def getting_and_scroing_articles(article):
    await calculate_cos(article=article)
    # res = supabase.table("news").select("*").execute()
    # articles = res.data
    # for article in articles:


def cosine_similarity(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def scrape(url, section_container, inner_section, element, id):

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    m = get_model()

    if id:
        return

    content = soup.find(section_container, class_=inner_section)

    headlines = content.find_all(element)

    res = supabase.table("news").select("Vectors").execute()

    existing_vectors = []

    for row in res.data:
        vec = row.get("Vectors")
        if vec:
            existing_vectors.append(vec)

    print(f"Loaded {len(existing_vectors)} existing embeddings")


    for title in headlines:

        text = title.get_text(strip=True)

        if not text:
            continue

        embedding = m.encode(text).tolist()

        is_duplicate = False

        for vec in existing_vectors:
            try:
                score = cosine_similarity(embedding, vec)

                if score > 0.85:
                    print("Duplicate skipped:", text)
                    is_duplicate = True
                    break
            except:
                continue

        if is_duplicate:
            continue

        article = {
            "tittle": text,
            "Vectors": embedding
        }

        try:
            supabase.table("news").upsert(article, on_conflict="tittle").execute()
            print("Saved:", text)
            existing_vectors.append(embedding)
            await getting_and_scroing_articles(article)
        except Exception as e:
            print("Failed:", text)
            print(e)

        print("-" * 80)



async def get_data_via_api():

    articles = []
    res = requests.get(f"https://newsdata.io/api/1/latest?apikey={newsdata_api_key}")
    data = res.json()

    m = get_model()

    for item in data.get("results", []):

        if not isinstance(item, dict):
            continue

        title = item.get("tittle")   

        if not title:
            continue

        embedding = m.encode(title).tolist()

        articles.append({
            "tittle": title,
            "Vectors": embedding
        })

    res2 =  requests.get(
        f"https://gnews.io/api/v4/search?q=example&lang=en&country=in&max=10&apikey={gnews_key}"
    )
    data2 = res2.json()

    for item in data2.get("articles", []):

        if not isinstance(item, dict):
            continue

        title = item.get("title")  

        if not title:
            continue

        embedding = m.encode(title).tolist()

        articles.append({
            "tittle": title,
            "Vectors": embedding   
        })
    res_db = supabase.table("news").select("Vectors").execute()

    existing_vectors = [
        row["Vectors"]
        for row in res_db.data
        if row.get("Vectors")
    ]

    THRESHOLD = 0.85

    for article in articles:

        try:
            embedding = article["Vectors"]

            is_duplicate = False

            for vec in existing_vectors:
                try:
                    score = cosine_similarity(embedding, vec)

                    if score > THRESHOLD:
                        print("Duplicate skipped:", article["tittle"])
                        is_duplicate = True
                        break
                except:
                    continue

            if is_duplicate:
                continue

            supabase.table("news").upsert(article, on_conflict="tittle").execute()

            print("Saved:", article["tittle"])

            existing_vectors.append(embedding)
            await getting_and_scroing_articles(article)
            
        except Exception as e:
            print("Failed:", article["tittle"])
            print(e)