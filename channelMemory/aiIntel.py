from supabase import create_client
from dotenv import load_dotenv
import os
from google import genai
import asyncio

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=google_api_key)

print(google_api_key)

url= os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

async def get_chunks_from_db():
    try:
        response = supabase.table("channel_memory").select("chunk_id , text").execute()
        data = response.data  
        for item in data:
            chunk = item["text"]
            chunk_id = item["chunk_id"]
            await get_intelligence(chunks=chunk,chunk_id=chunk_id)
            print(chunk)
            print("------")  
    except Exception as e:
        print(e)        
        
async def get_intelligence(chunks,chunk_id):
   
    try :
        prompt = f"""
        Analyze the following script fragments from a single creator. Your goal is to create a "Channel Profile" that allows another AI to perfectly mimic this creator's style.
        Script Fragments:
        f{chunks}
        Analyze and provide a structured report on the following 5 axes:
        Formality & Tone: On a scale of 1-10 (1=Street slang/Casual, 10=Academic/Professional), where does this sit? Describe the "vibe" (e.g., sarcastic, urgent, empathetic).
        Sentence Rhythm: Does the creator use short, punchy fragments, or long, complex narrations? Do they use rhetorical questions often?
        Vocabulary & "Slang": Identify at least 3-5 specific words, technical terms, or "filler" phrases the creator uses repeatedly.
        Structural Signatures: How do they open a scene or transition between ideas? (e.g., "Cut to," "So here's the thing," "Anyway...").
        Energy Level: Is the delivery "High Hype," "Deadpan," or "Steady Instructional"?
        Format the output as a clean summary that can be used as a 'System Instruction' for future generations
        """

        response = await client.aio.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        summary = response.text
        supabase.table("Channel Profile").insert({
            "chunk_id": chunk_id,
            "Summary": summary
        }).execute()       
        print("saved ai response in db")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    asyncio.run(get_chunks_from_db())