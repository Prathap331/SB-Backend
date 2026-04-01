#main3
# Add these imports to main.py
from fastapi import Depends, HTTPException, status
from auth_dependencies import get_current_user # Import our dependency
from postgrest.exceptions import APIError # For handling DB errors
from gotrue.types import User # For type hinting the user object

import os
import google.generativeai as genai
import httpx
import asyncio
import time
import re
from bs4 import BeautifulSoup
from ddgs import DDGS
from readability import Document
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
import nltk
from nltk.tokenize import sent_tokenize

import json

from fastapi import Request, Header # Import Request and Header
import hmac
import hashlib
import razorpay

# --- Setup and Configuration (Using your specified models) ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=api_key)

# Using the models from your working code



pro_model = genai.GenerativeModel('models/gemini-2.5-flash-preview-09-2025')

flash_model = genai.GenerativeModel('models/gemini-flash-latest')

content_genmodel=genai.GenerativeModel('models/gemini-2.5-flash-preview-09-2025')


embedding_model = 'models/text-embedding-004'

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials not found in .env file")
supabase: Client = create_client(supabase_url, supabase_key)
print("Clients for Google AI and Supabase initialized successfully.")


# --- Helper Functions (Your existing, working code) ---
def chunk_text(text: str, chunk_size: int = 250, chunk_overlap: int = 50) -> list[str]:
    # (This function is unchanged)
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            overlap_words = current_chunk.split()[-chunk_overlap:]
            current_chunk = " ".join(overlap_words) + " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def add_scraped_data_to_db(article_title: str, article_text: str, article_url: str):
    # (This function is unchanged)
    print(f"BACKGROUND TASK: Starting to upload '{article_title[:30]}...'")
    try:
        raw_chunks = chunk_text(article_text)
        chunks = [chunk for chunk in raw_chunks if chunk and not chunk.isspace()]
        if not chunks:
            print("BACKGROUND TASK: No valid chunks to process.")
            return
        embedding_result = genai.embed_content(model=embedding_model, content=chunks, task_type="retrieval_document")
        embeddings = embedding_result['embedding']
        documents_to_insert = [{"content": chunk, "embedding": embeddings[i], "source_title": article_title, "source_url": article_url} for i, chunk in enumerate(chunks)]
        supabase.table('documents').insert(documents_to_insert).execute()
        print(f"BACKGROUND TASK: Successfully uploaded {len(documents_to_insert)} chunks.")
    except Exception as e:
        print(f"BACKGROUND TASK: Failed to add data to DB. Error: {e}")

async def scrape_url(client: httpx.AsyncClient, url: str, scraped_urls: set):
    # (This function is unchanged)
    if url in scraped_urls:
        return None
    print(f"Scraping: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = await client.get(url, headers=headers, timeout=10, follow_redirects=True)
        response.raise_for_status()
        scraped_urls.add(url)
        doc = Document(response.text)
        title = doc.title()
        article_html = doc.summary()
        soup = BeautifulSoup(article_html, 'html.parser')
        article_text = soup.get_text(separator='\n', strip=True)
        return {"url": url, "title": title, "text": article_text}
    except Exception as e:
        print(f"An error occurred while processing {url}: {e}")
        return None

async def deep_search_and_scrape(keywords: list[str], scraped_urls: set) -> list[dict]:
    # (This function is unchanged)
    print("--- DEEP WEB SCRAPE: Starting full search... ---")
    urls_to_scrape = set()
    with DDGS(timeout=20) as ddgs:
        for keyword in keywords:
            search_results = list(ddgs.text(keyword, region='wt-wt', max_results=3))
            if search_results:
                urls_to_scrape.add(search_results[0]['href'])
    async with httpx.AsyncClient() as client:
        tasks = [scrape_url(client, url, scraped_urls) for url in urls_to_scrape]
        results = await asyncio.gather(*tasks)
        return [res for res in results if res and res.get("text")]

async def get_latest_news_context(topic: str, scraped_urls: set) -> list[dict]:
    # (This function is unchanged)
    print("--- LIGHT WEB SCRAPE: Starting lightweight news search... ---")
    try:
        keyword = f"{topic} latest news today"
        urls_to_scrape = set()
        with DDGS(timeout=10) as ddgs:
            search_results = list(ddgs.text(keyword, region='wt-wt', max_results=2))
            for result in search_results:
                urls_to_scrape.add(result['href'])
        async with httpx.AsyncClient() as client:
            tasks = [scrape_url(client, url, scraped_urls) for url in urls_to_scrape]
            results = await asyncio.gather(*tasks)
            return [res for res in results if res and res.get("text")]
    except Exception as e:
        print(f"--- WEB TASK: Error during news scraping: {e} ---")
        return []

async def get_db_context(topic: str) -> list[dict]:
    # (This function is unchanged)
    print("--- DB TASK: Starting HyDE database search... ---")
    try:
        hyde_prompt = f"""
        Write a short, factual, encyclopedia-style paragraph that provides a direct answer to the following topic.
        This will be used to find similar documents, so be concise and include key terms.
        
        Topic: "{topic}"
        """
        hyde_response = await flash_model.generate_content_async(hyde_prompt)
        query_embedding = genai.embed_content(model=embedding_model, content=hyde_response.text, task_type="retrieval_query")['embedding']
        db_results = supabase.rpc('match_documents', {'query_embedding': query_embedding, 'match_threshold': 0.65, 'match_count': 5}).execute()
        return db_results.data
    except Exception as e:
        print(f"--- DB TASK: Error during database search: {e} ---")
        return []

# --- FastAPI App ---
app = FastAPI()
class PromptRequest(BaseModel):
    topic: str

@app.get("/")
async def read_root(): return {"status": "Welcome"}

@app.post("/process-topic")
async def process_topic(request: PromptRequest, background_tasks: BackgroundTasks):
    total_start_time = time.time()
    #user_id = current_user.id
    print(f"Received topic from user : {request.topic}")
    '''
    # --- NEW: Credit Check Logic ---
    IDEA_COST = 1 # Define the cost for this specific action
    try:
        profile_response = supabase.table('profiles').select('credits_remaining, user_tier').eq('id', user_id).single().execute()
        profile = profile_response.data
        if not profile:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found.")

        credits = profile.get('credits_remaining', 0)
        user_tier = profile.get('user_tier', 'free')

        # Check if the user has ENOUGH credits for THIS action
        if user_tier != 'admin' and credits < IDEA_COST:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. This action requires {IDEA_COST} credit(s). You have {credits}."
            )

        print(f"User {user_id} (Tier: {user_tier}) has {credits} credits. Action requires {IDEA_COST}.")
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error checking profile: {e.message}")
    except HTTPException as e: # Re-raise HTTP exceptions from credit check
        raise e
    except Exception as e:
         print(f"Unexpected Error checking credits: {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error checking profile.")
    # --------------------------------'''
    
    try:
        db_task = asyncio.create_task(get_db_context(request.topic))
        
        await asyncio.sleep(11) # Your working sleep time

        db_results = []
        new_articles = []
        scraped_urls = set()
        # --- NEW: Initialize these here to ensure they exist for the final return ---
        base_keywords = []
        source_of_context = ""
        # --------------------------------------------------------------------------

        if db_task.done():
            db_results = db_task.result()
            print(f"--- DB task finished early. Found {len(db_results)} documents. ---")

        if len(db_results) >= 3:
            source_of_context = "DATABASE_WITH_NEWS"
            new_articles = await get_latest_news_context(request.topic, scraped_urls)
        else:
            source_of_context = "DEEP_SCRAPE"
            print("--- DB MISS or SLOW: Initiating DEEP web scrape. ---")
            keyword_prompt =  f"""
            Your ONLY task is to generate 3 diverse search engine keyword phrases for the topic: '{request.topic}'.
            Follow these rules STRICTLY:
            1. Return ONLY the 3 phrases.
            2. DO NOT add numbers, markdown, explanations, or any introductory text.
            3. Each phrase must be on a new line.
            EXAMPLE INPUT: Is coding dead?
            EXAMPLE OUTPUT:
            future of programming jobs automation
            AI replacing software developers
            demand for software engineers 2025
            """
            response = await flash_model.generate_content_async(keyword_prompt)
            # This parsing logic is from your older, working version
            raw_text = response.text
            keywords_in_quotes = re.findall(r'"(.*?)"', raw_text)
            if keywords_in_quotes:
                base_keywords = keywords_in_quotes
            else:
                base_keywords = [kw.strip() for kw in raw_text.strip().split('\n') if kw.strip()]
            
            targeted_keywords = [kw for kw in base_keywords] + [f"{kw} site:reddit.com" for kw in base_keywords]
            new_articles = await deep_search_and_scrape(targeted_keywords, scraped_urls)

        if not db_task.done():
            print("--- Waiting for DB task to complete... ---")
            db_results = await db_task
            print(f"--- DB task finished. Found {len(db_results)} documents. ---")

        db_context, web_context = "", ""
        source_urls = []
        
        if db_results:
            db_context = "\n\n".join([item['content'] for item in db_results])
            source_urls.extend(list(set([item['source_url'] for item in db_results if item['source_url']])))

        if new_articles:
            web_context = "\n\n".join([f"Source: {art['title']}\n{art['text']}" for art in new_articles])
            source_urls.extend([art['url'] for art in new_articles])
            for article in new_articles:
                background_tasks.add_task(add_scraped_data_to_db, article['title'], article['text'], article['url'])

        if not db_context and not web_context:
            return {"error": "Could not find any information."}

        

        # --- THIS IS THE UPGRADED PROMPT ---
        final_prompt = f"""
        You are an expert YouTube title strategist and scriptwriter.
        Your mission is to generate 4 distinct, attention-grabbing video titles for the topic: "{request.topic}", AND a corresponding description for each.

        Use the provided research material to inform your output:
        - Use the 'FOUNDATIONAL KNOWLEDGE' for deep context, facts, and historical background.
        - Use the 'LATEST NEWS' to find a fresh, timely, or surprising angle, especially considering the current date is October 15, 2025.

        RULES FOR YOUR OUTPUT:
        1.  For each of the 4 ideas, provide a 'TITLE' and a 'DESCRIPTION'.
        2.  Each 'DESCRIPTION' MUST be between 90 and 110 words.
        3.  Separate each complete idea (title + description) with '---'.
        4.  DO NOT add any introductory sentences, explanations, or any text other than the titles and descriptions in the specified format.

        EXAMPLE OUTPUT FORMAT:
        TITLE: This Is Why Everyone Is Suddenly Talking About [Topic]
        DESCRIPTION: In this video, we uncover the shocking truth behind [Topic]. For years, experts have believed one thing, but new data from October 2025 reveals a completely different story. We'll break down the historical context, analyze the latest reports, and explain exactly why this topic is about to become the biggest conversation on the internet. You'll learn about the key players, the secret history, and what this means for the future. Don't miss this deep dive into one of the most misunderstood subjects of our time, it will change everything you thought you knew.
        ---
        TITLE: The Hidden Truth Behind [Related Concept]
        DESCRIPTION: Everyone thinks they understand [Related Concept], but they're wrong. We've dug through the archives and analyzed the latest breaking news to bring you the untold story. This video explores the forgotten origins, the powerful figures who shaped its narrative, and the surprising new developments that are challenging everything we know. We connect the dots from the foundational knowledge to the fresh web updates to give you a complete picture you won't find anywhere else. Get ready to have your mind blown by the real story behind [Related Concept].
        ---
        
        RESEARCH FOR TOPIC: "{request.topic}"
        ---
        FOUNDATIONAL KNOWLEDGE (from our database):
        {db_context}
        ---
        LATEST NEWS UPDATES (from the web):
        {web_context}
        ---
        """
        step3_start_time = time.time()
        final_response = await pro_model.generate_content_async(final_prompt)
        step3_end_time = time.time()
        print(f"--- PROFILING: Step 3 (Final Idea Gen) took {step3_end_time - step3_start_time:.2f} seconds ---")

        # --- THIS IS THE NEW, SMARTER PARSING LOGIC ---
        response_text = final_response.text
        
        final_ideas = []
        final_descriptions = []
        
        # Split the entire response into blocks, one for each idea
        idea_blocks = response_text.strip().split('---')
        
        for block in idea_blocks:
            title = ""
            description = ""
            lines = block.strip().split('\n')
            
            for line in lines:
                if line.startswith('TITLE:'):
                    # Extract text after "TITLE:"
                    title = line.replace('TITLE:', '', 1).strip()
                elif line.startswith('DESCRIPTION:'):
                    # Extract text after "DESCRIPTION:"
                    description = line.replace('DESCRIPTION:', '', 1).strip()
            
            # Only add the pair if both title and description were found
            if title and description:
                final_ideas.append(title)
                final_descriptions.append(description)

        print(f"Final generated ideas: {final_ideas}")
        print(f"Final generated descriptions: {len(final_descriptions)} descriptions found.")
        total_end_time = time.time()
        print(f"--- PROFILING: Total request time was {total_end_time - total_start_time:.2f} seconds ---")
        


        '''
        # --- <<< CREDIT DECREMENT LOGIC >>> ---
        # Ensure 'user_tier' and 'credits' were fetched successfully earlier in the function
        if 'user_tier' in locals() and user_tier != 'admin':
            try:
                # Use the 'credits' variable fetched at the start of the function
                # IDEA_COST should also be defined at the start (e.g., IDEA_COST = 1)
                new_credit_balance = credits - IDEA_COST

                # Ensure balance doesn't go below zero (optional safety check)
                if new_credit_balance < 0:
                    new_credit_balance = 0
                    print(f"WARN: User {user_id} credit balance would go below zero. Setting to 0.")

                # Update the database
                update_result = supabase.table('profiles').update(
                    {'credits_remaining': new_credit_balance}
                ).eq('id', user_id).execute()

                # Optional: Check if the update was successful
                if len(update_result.data) > 0:
                    print(f"Decremented {IDEA_COST} credit(s) for user {user_id}. New balance: {new_credit_balance}")
                else:
                    # This might happen if the user was deleted between check and update,
                    # or if RLS rules prevent the update (less likely with service_role key)
                    print(f"WARN: Credit decrement query executed for user {user_id} but returned no updated rows.")

            except APIError as e:
                # Log the error but crucially, DO NOT raise an HTTPException here.
                # The user already got their result, failing the credit decrement
                # is an internal issue we should log, not fail the user's request for.
                print(f"ERROR: Failed to decrement credits for user {user_id}. DB Error: {e.message}")
            except Exception as e:
                # Catch any other unexpected errors during the update
                print(f"ERROR: An unexpected error occurred during credit decrement for user {user_id}: {e}")
        elif 'user_tier' in locals() and user_tier == 'admin':
             print(f"User {user_id} is admin. No credits decremented.")
        else:
            # This case indicates an issue earlier in the function
            print(f"WARN: Could not decrement credits for user {user_id}. User tier or initial credit count not available.")
        # --- <<< END CREDIT DECREMENT LOGIC >>> ---
        '''






        # --- THIS IS THE UPDATED RETURN STATEMENT ---
        return {
            "source_of_context": source_of_context,
            "ideas": final_ideas,
            "descriptions": final_descriptions, # The new descriptions list
            "generated_keywords": base_keywords,
            "source_urls": list(set(source_urls)),
            "scraped_text_context": f"DB CONTEXT:\n{db_context}\n\nWEB CONTEXT:\n{web_context}"
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": "An error occurred in the processing pipeline."}
    





# ------------------------------------

# --- Define Script Structure Options ---
STRUCTURE_GUIDANCE = {
    "problem_solution": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (~10%)
    - Problem / Conflict (~15%)
    - Evidence & Data (~20%)
    - Real-world Examples (~25%)
    - Potential Solutions / Insights (~25%)
    - Call to Action (~5%)
    """,
        "storytelling": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (Introduce Ordinary World) (~10%)
    - Call to Adventure / Inciting Incident (~10%)
    - Trials & Tribulations (Rising Action, using examples/data) (~50%)
    - Climax / Resolution (~20%)
    - Reflection & Takeaway (Call to Action) (~10%)
    """,
        "listicle": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (State the list topic & number) (~10%)
    - Item 1 (Explanation, examples, pros/cons) (~15-20%)
    - Item 2 (...) (~15-20%)
    - Item 3 (...) (~15-20%)
    - Item X (...) (~15-20%) - *Adjust percentages based on number of items*
    - (Optional) Bonus Item / Honorable Mentions (~10%)
    - Conclusion & Call to Action (Summarize, final thought) (~10%)
    """,
        "chronological": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (Introduce topic & relevance) (~10%)
    - Early Beginnings / Origins (~20%)
    - Key Developments / Turning Points (~40%) - *This is the main body*
    - Later Stages / Modern Impact (~20%)
    - Conclusion & Reflection (Call to Action) (~10%)
    """,
        "myth_debunking": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (Introduce common misconception) (~10%)
    - Myth 1 & Fact 1 (State myth, then debunk with evidence) (~25%)
    - Myth 2 & Fact 2 (...) (~25%)
    - Myth 3 & Fact 3 (...) (~25%) - *Adjust percentages based on number of myths*
    - Conclusion & Call to Action (Summarize truths, encourage critical thinking) (~15%)
""",
    "tech_review": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (Show product, state review goal) (~10%)
    - Design & Build Quality (Look, feel) (~15%)
    - Key Features & Specs (What it promises, tech details) (~20%)
    - Performance & User Experience (Real-world testing, how it feels to use, battery, camera examples etc.) (~30%)
    - Pros & Cons (Balanced summary of good and bad) (~10%)
    - Verdict & Recommendation (Who is it for? Worth the price? Call to Action) (~15%)
    """
}
# ------------------------------------

# --- FastAPI App ---
#app = FastAPI()

class PromptRequest(BaseModel):
    topic: str

# --- UPDATED: Add duration_minutes ---
class ScriptRequest(BaseModel):
    topic: str
    emotional_tone: str | None = "engaging"
    creator_type: str | None = "educator"
    audience_description: str | None = "a general audience interested in learning"
    accent: str | None = "neutral"
    duration_minutes: int | None = 10 # NEW: Add video duration in minutes, default 10
    script_structure: str | None = "problem_solution" # NEW FIELD
# ------------------------------------

# --- REWRITTEN: The /generate-script endpoint with dynamic duration ---
@app.post("/generate-script")
async def generate_script(request: ScriptRequest, background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)):
    total_start_time = time.time()
    user_id = current_user.id
    print(f"SCRIPT GENERATION from user ({user_id}): '{request.topic}'")

    # --- NEW: Credit Check Logic ---
    IDEA_COST = 3 # Define the cost for this specific action
    try:
        profile_response = supabase.table('profiles').select('credits_remaining, user_tier').eq('id', user_id).single().execute()
        profile = profile_response.data
        if not profile:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found.")

        credits = profile.get('credits_remaining', 0)
        user_tier = profile.get('user_tier', 'free')

        # Check if the user has ENOUGH credits for THIS action
        if user_tier != 'admin' and credits < IDEA_COST:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. This action requires {IDEA_COST} credit(s). You have {credits}."
            )

        print(f"User {user_id} (Tier: {user_tier}) has {credits} credits. Action requires {IDEA_COST}.")
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error checking profile: {e.message}")
    except HTTPException as e: # Re-raise HTTP exceptions from credit check
        raise e
    except Exception as e:
         print(f"Unexpected Error checking credits: {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error checking profile.")
    # --------------------------------
    # --------------------------------
    #print(f"SCRIPT GENERATION: Received request for topic: '{request.topic}'")
    print(f"Personalization - Duration: {request.duration_minutes} min, Tone: {request.emotional_tone}, Type: {request.creator_type}, Audience: {request.audience_description}, Accent: {request.accent}")

    try:
        # --- Step 1: Gather Context (Unchanged) ---
        db_task = asyncio.create_task(get_db_context(request.topic))
        await asyncio.sleep(11) # Give DB head start

        db_results = []
        new_articles = []
        scraped_urls = set()
        base_keywords = []

        if db_task.done():
            db_results = db_task.result()
            print(f"--- DB task finished early. Found {len(db_results)} documents. ---")

        if len(db_results) >= 3:
            print("--- DB HIT: Performing LIGHT web scrape for latest news. ---")
            new_articles = await get_latest_news_context(request.topic, scraped_urls)
        else:
            print("--- DB MISS or SLOW: Initiating DEEP web scrape. ---")
            # (Deep scrape logic remains the same)
            keyword_prompt =  f"""
            Your ONLY task is to generate 3 diverse search engine keyword phrases for the topic: '{request.topic}'.
            Follow these rules STRICTLY:
            1. Return ONLY the 3 phrases.
            2. DO NOT add numbers, markdown, explanations, or any introductory text.
            3. Each phrase must be on a new line.
            EXAMPLE INPUT: Is coding dead?
            EXAMPLE OUTPUT:
            future of programming jobs automation
            AI replacing software developers
            demand for software engineers 2025
            """
            response = await flash_model.generate_content_async(keyword_prompt)
            raw_text = response.text
            keywords_in_quotes = re.findall(r'"(.*?)"', raw_text)
            if keywords_in_quotes: base_keywords = keywords_in_quotes
            else: base_keywords = [kw.strip() for kw in raw_text.strip().split('\n') if kw.strip()]
            targeted_keywords = [kw for kw in base_keywords] + [f"{kw} site:reddit.com" for kw in base_keywords]
            new_articles = await deep_search_and_scrape(targeted_keywords, scraped_urls)

        if not db_task.done():
            print("--- Waiting for DB task to complete... ---")
            db_results = await db_task
            print(f"--- DB task finished. Found {len(db_results)} documents. ---")

        # --- Step 2: Merge Context (Unchanged) ---
        db_context, web_context = "", ""
        if db_results:
            db_context = "\n\n".join([item['content'] for item in db_results])
        if new_articles:
            web_context = "\n\n".join([f"Source: {art['title']}\n{art['text']}" for art in new_articles])
            for article in new_articles:
                background_tasks.add_task(add_scraped_data_to_db, article['title'], article['text'], article['url'])

        if not db_context and not web_context:
            return {"error": "Could not find any research material to write the script."}

        # --- Step 3: Calculate Word Count & Create Personalized Prompt ---
        print("SCRIPT GENERATION: Generating personalized script...")
        
        # --- NEW: Calculate target word count ---
        WORDS_PER_MINUTE = 130
        target_duration = request.duration_minutes if request.duration_minutes else 10 # Use default if not provided
        target_word_count = target_duration * WORDS_PER_MINUTE
        print(f"Targeting {target_duration} minutes / approx. {target_word_count} words.")
        
        
        # --- NEW: Select the requested structure guidance ---
        requested_structure = request.script_structure if request.script_structure else "problem_solution"
        structure_guidance_text = STRUCTURE_GUIDANCE.get(requested_structure, STRUCTURE_GUIDANCE["problem_solution"]) # Fallback to default
        print(f"Using script structure: {requested_structure}")
        
        # --------------------------------------
        
        # --- UPDATED PROMPT with dynamic values ---
        script_prompt = f"""
        You are a professional YouTube scriptwriter who creates natural, engaging, and conversational scripts that feel like a real YouTuber speaking directly to the camera.

        **Creator Profile:**
        * **Creator Type:** {request.creator_type}
        * **Target Audience:** {request.audience_description}
        * **Desired Emotional Tone:** {request.emotional_tone}
        * **Accent/Dialect:** {request.accent} (use phrasing natural for this accent)

        **Your Task:**
        Generate a complete YouTube video script of approximately **{target_duration} minutes** (~{target_word_count} words) based on the **main topic** below, using the provided **research context**.

        **Script Style & Flow:**
        - Output only the spoken dialogue — what the YouTuber would actually say aloud.
        - **Do NOT include** section titles, notes, stage directions, or metadata.
        - Speak directly to the viewer — friendly, confident, slightly spontaneous, and off-the-cuff.
        - Use **short and medium-length sentences**, natural pauses (…) or dashes, and occasional repetition for emphasis.
        - Include interjections, rhetorical questions, playful digressions, humor, and brief asides (“Wait, actually…”, “Can you believe that…?”, “By the way…”).
        - Include personal anecdotes or opinions (“I remember…”, “When I tried this…”).
        - Use **visual and emotional imagery** to make scenes vivid (“Imagine this…”, “Picture it like…”).
        - Hook viewers emotionally in the first 15–30 seconds.
        - Alternate between facts, insights, reactions, and short reflections to keep pacing dynamic.
        - Treat the script as a conversation with the audience — inclusive language like “you guys”, “we all”, “my friends”.
        - Build suspense naturally with rhetorical questions, mini cliffhangers, or curiosity hooks.
        - Use relatable analogies or humor when explaining complex topics.
        - Occasionally reference the creator’s regional or cultural context for relatability.
        - Maintain natural pacing as if recording live — mix excitement, storytelling, and factual explanation.
        - Stay close to **{target_word_count} words** (±50).

        
        {structure_guidance_text} 

        **Main Topic/Idea:** "{request.topic}"

        **Research Context:**
        FOUNDATIONAL KNOWLEDGE (from database): {db_context}
        LATEST NEWS (from web): {web_context}

        **Additional Notes:**
        - Make the opening a curiosity-driven hook that emotionally pulls the viewer in within 15–30 seconds.
        - Use storytelling techniques: tension, suspense, surprise, and moral dilemmas when relevant.
        - Make historical or technical details feel immersive and personal, not like a lecture.
        - Emphasize the narrative arc: build curiosity, climax, and reflection for the audience.
        - Ensure adaptability: script should feel natural regardless of topic, duration, or target audience.
        """

        
        # ------------------------------------------

        script_response = await content_genmodel.generate_content_async(script_prompt)
        
        total_end_time = time.time()
        print(f"--- PROFILING: Script generation took {total_end_time - total_start_time:.2f} seconds ---")



        # --- NEW: Prompt for Script Analysis ---
        ANALYSIS_PROMPT_TEMPLATE = """
        You are an expert script analyzer. Analyze the provided YouTube script based on the following criteria:

        1.  **Real-world Examples:** Count how many distinct real-world examples, case studies, or specific stories are mentioned.
        2.  **Research Facts/Stats:** Count how many distinct research findings, statistics, or specific data points are cited or explained.
        3.  **Proverbs/Sayings:** Count how many common proverbs, idioms, or well-known sayings are used.
        4.  **Emotional Depth:** Assess the overall emotional depth and engagement level of the script. Rate it as Low, Medium, or High.

        **Your Task:**
        Read the script below and return ONLY a JSON object with the results. Do not add any explanation or other text.

        **EXAMPLE OUTPUT FORMAT:**
        {{
        "examples_count": 3,
        "research_facts_count": 5,
        "proverbs_count": 1,
        "emotional_depth": "Medium"
        }}

        --- SCRIPT TO ANALYZE ---
        {script_text}
        --- END SCRIPT ---
        """


        # --- NEW: Step 6 - Analyze the Generated Script ---
        print("SCRIPT ANALYSIS: Analyzing generated script...")
        analysis_start_time = time.time()
        analysis_prompt_filled = ANALYSIS_PROMPT_TEMPLATE.format(script_text=script_response.text)
        
        analysis_response = await flash_model.generate_content_async(analysis_prompt_filled)
        analysis_end_time = time.time()
        print(f"--- PROFILING: Script analysis took {analysis_end_time - analysis_start_time:.2f} seconds ---")
        
        # --- NEW: Parse the analysis results (with error handling) ---
        analysis_results = {
            "examples_count": 0,
            "research_facts_count": 0,
            "proverbs_count": 0,
            "emotional_depth": "Unknown"
        }
        try:
            # Attempt to parse the JSON response from the analysis model
            analysis_data = json.loads(analysis_response.text)
            analysis_results["examples_count"] = analysis_data.get("examples_count", 0)
            analysis_results["research_facts_count"] = analysis_data.get("research_facts_count", 0)
            analysis_results["proverbs_count"] = analysis_data.get("proverbs_count", 0)
            analysis_results["emotional_depth"] = analysis_data.get("emotional_depth", "Unknown")
            print(f"Script Analysis Results: {analysis_results}")
        except json.JSONDecodeError:
            print("SCRIPT ANALYSIS: Failed to parse analysis JSON response from AI.")
        except Exception as e:
             print(f"SCRIPT ANALYSIS: Error during analysis parsing: {e}")
        # -----------------------------------------------------------

        total_end_time = time.time()
        print(f"--- PROFILING: Total /generate-script analysis request time was {total_end_time - total_start_time:.2f} seconds ---")
        
        
        generated_word_count = len(script_response.text.split())
        print(f"Generated script word count: approx. {generated_word_count}")



        # --- <<< CREDIT DECREMENT LOGIC >>> ---
        # Ensure 'user_tier' and 'credits' were fetched successfully earlier in the function
        if 'user_tier' in locals() and user_tier != 'admin':
            try:
                # Use the 'credits' variable fetched at the start of the function
                # IDEA_COST should also be defined at the start (e.g., IDEA_COST = 1)
                new_credit_balance = credits - IDEA_COST

                # Ensure balance doesn't go below zero (optional safety check)
                if new_credit_balance < 0:
                    new_credit_balance = 0
                    print(f"WARN: User {user_id} credit balance would go below zero. Setting to 0.")

                # Update the database
                update_result = supabase.table('profiles').update(
                    {'credits_remaining': new_credit_balance}
                ).eq('id', user_id).execute()

                # Optional: Check if the update was successful
                if len(update_result.data) > 0:
                    print(f"Decremented {IDEA_COST} credit(s) for user {user_id}. New balance: {new_credit_balance}")
                else:
                    # This might happen if the user was deleted between check and update,
                    # or if RLS rules prevent the update (less likely with service_role key)
                    print(f"WARN: Credit decrement query executed for user {user_id} but returned no updated rows.")

            except APIError as e:
                # Log the error but crucially, DO NOT raise an HTTPException here.
                # The user already got their result, failing the credit decrement
                # is an internal issue we should log, not fail the user's request for.
                print(f"ERROR: Failed to decrement credits for user {user_id}. DB Error: {e.message}")
            except Exception as e:
                # Catch any other unexpected errors during the update
                print(f"ERROR: An unexpected error occurred during credit decrement for user {user_id}: {e}")
        elif 'user_tier' in locals() and user_tier == 'admin':
             print(f"User {user_id} is admin. No credits decremented.")
        else:
            # This case indicates an issue earlier in the function
            print(f"WARN: Could not decrement credits for user {user_id}. User tier or initial credit count not available.")
        # --- <<< END CREDIT DECREMENT LOGIC >>> ---








        # --- FINAL RETURN STATEMENT with all the data ---
        return {
            "script":script_response.text ,
            "estimated_word_count": generated_word_count,
            "source_urls": list(scraped_urls), # Use the correct list
            "analysis": analysis_results # Add the analysis results
        }

    except Exception as e:
        print(f"SCRIPT GENERATION: An error occurred: {e}")
        return {"error": "An error occurred during the script generation pipeline."}




class CreateOrderRequest(BaseModel):
    amount: int # Amount in paisa (e.g., 50000 for ₹500.00)
    currency: str = "INR"
    receipt: str | None = None # Optional unique receipt ID from your system
    target_tier: str # 'basic' or 'pro' - needed to calculate amount

# Endpoint to create a Razorpay order
@app.post("/payments/create-order")
async def create_razorpay_order(
    request_data: CreateOrderRequest,
    current_user: User = Depends(get_current_user)
):
    if not razorpay_client:
        raise HTTPException(status_code=503, detail="Payment service unavailable.")

    user_id = current_user.id
    amount = request_data.amount # Amount should be sent from frontend based on selected tier
    currency = request_data.currency

    # Basic validation (add more as needed)
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Invalid amount.")
    if request_data.target_tier not in ['basic', 'pro']:
         raise HTTPException(status_code=400, detail="Invalid target tier specified.")

    order_data = {
        "amount": amount,
        "currency": currency,
        #"receipt": request_data.receipt or f"receipt_{user_id}_{int(time.time())}", 
        "receipt": request_data.receipt or f"rec_{int(time.time())}", # Shorter default receipt
        # Generate a receipt if none provided
        "notes": { # Store extra info like user ID and target tier
            "user_id": str(user_id),
            "target_tier": request_data.target_tier
        }
    }

    try:
        order = razorpay_client.order.create(data=order_data)
        print(f"Created Razorpay order {order['id']} for user {user_id}")
        # Return the order ID and key ID to the frontend
        return {"order_id": order['id'], "key_id": RAZORPAY_KEY_ID, "amount": amount, "currency": currency}
    except Exception as e:
        print(f"Error creating Razorpay order: {e}")
        raise HTTPException(status_code=500, detail="Could not create payment order.")
    


RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")


# Endpoint for Razorpay Webhook
@app.post("/payments/webhook")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str | None = Header(None) # Get signature from header
):
    if not RAZORPAY_WEBHOOK_SECRET or not razorpay_client:
        print("Webhook received but service is not configured.")
        return {"status": "Webhook ignored"} # Don't raise error, just ignore

    body = await request.body() # Get raw body

    # --- 1. Verify Signature ---
    try:
        razorpay_client.utility.verify_webhook_signature(
            body.decode('utf-8'), # Decode body bytes to string
            x_razorpay_signature,
            RAZORPAY_WEBHOOK_SECRET
        )
        print("Webhook signature verified successfully.")
    except razorpay.errors.SignatureVerificationError as e:
        print(f"Webhook signature verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")
    except Exception as e:
        print(f"Error during webhook signature verification: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing error.")

    # --- 2. Process the Event ---
    try:
        event_data = json.loads(body)
        event_type = event_data.get('event')

        print(f"Received webhook event: {event_type}")

        if event_type == 'payment.captured' or event_type == 'order.paid':
            payment_entity = event_data['payload']['payment']['entity']
            order_entity = event_data['payload']['order']['entity'] # Get order entity

            order_id = order_entity['id']
            payment_id = payment_entity['id']
            amount_paid = order_entity['amount_paid'] # Use amount from order entity
            notes = order_entity.get('notes', {})
            user_id = notes.get('user_id')
            target_tier = notes.get('target_tier')

            print(f"Processing successful payment: {payment_id} for order {order_id}, user {user_id}, tier {target_tier}")

            if not user_id or not target_tier:
                print(f"ERROR: Missing user_id or target_tier in order notes for order {order_id}.")
                return {"status": "error", "message": "Missing required order notes."}

            # --- 3. Update User Profile in Supabase ---
            # Define credits based on tier (example values)
            credits_to_add = 0
            if target_tier == 'basic':
                credits_to_add = 50 # Example: Basic tier gets 50 credits
            elif target_tier == 'pro':
                credits_to_add = 200 # Example: Pro tier gets 200 credits

            try:
                # Fetch current credits first (important for concurrency)
                profile_response = supabase.table('profiles').select('credits_remaining').eq('id', user_id).single().execute()
                current_credits = 0
                if profile_response.data:
                    current_credits = profile_response.data.get('credits_remaining', 0)

                new_credits = current_credits + credits_to_add

                update_result = supabase.table('profiles').update({
                    'user_tier': target_tier,
                    'credits_remaining': new_credits
                }).eq('id', user_id).execute()

                if update_result.data:
                    print(f"Successfully updated user {user_id} to tier '{target_tier}' with {new_credits} credits.")
                else:
                    print(f"WARN: Failed to update profile for user {user_id} after payment {payment_id}.")
                    # Consider adding to a retry queue or alerting system

            except APIError as e:
                print(f"ERROR: Supabase APIError updating profile for user {user_id}: {e}")
                # Add to retry queue or alert
            except Exception as e:
                 print(f"ERROR: Unexpected error updating profile for user {user_id}: {e}")
                 # Add to retry queue or alert

        elif event_type == 'payment.failed':
            payment_entity = event_data['payload']['payment']['entity']
            order_id = payment_entity.get('order_id')
            print(f"Payment failed for order {order_id}. Reason: {payment_entity.get('error_description')}")
            # Optionally update your system (e.g., mark order as failed)

        else:
            print(f"Ignoring unhandled webhook event type: {event_type}")

        # Respond to Razorpay quickly
        return {"status": "Webhook processed successfully"}

    except json.JSONDecodeError:
        print("Webhook error: Could not decode JSON body.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    except Exception as e:
        print(f"Webhook error: Unexpected error processing event: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing webhook.")

