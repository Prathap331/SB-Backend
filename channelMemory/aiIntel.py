from supabase import create_client
from dotenv import load_dotenv
import os
from openai import OpenAI
import asyncio

load_dotenv()

url= os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


async def get_intelligence(chunks, userId):
    try:
        prompt = f"""
        Analyze the following script fragments from a single creator. Your goal is to create a "Channel Profile" that allows another AI to perfectly mimic this creator's style.

        Script Fragments:
        {chunks}

        Analyze and provide a structured report on the following 5 axes:

        Formality & Tone:
        On a scale of 1-10 (1=Street slang/Casual, 10=Academic/Professional), where does this sit?
        Describe the "vibe" (e.g., sarcastic, urgent, empathetic).

        Sentence Rhythm:
        Does the creator use short, punchy fragments, or long, complex narrations?
        Do they use rhetorical questions often?

        Vocabulary & "Slang":
        Identify at least 3-5 specific words, technical terms, or "filler" phrases the creator uses repeatedly.

        Structural Signatures:
        How do they open a scene or transition between ideas?
        (e.g., "Cut to," "So here's the thing," "Anyway...")

        Energy Level:
        Is the delivery "High Hype," "Deadpan," or "Steady Instructional"?

        Format the output as a clean summary that can be used as a 'System Instruction' for future generations.
        """

        response = await asyncio.to_thread(
            deepseek_client.chat.completions.create,
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "Return only the structured channel profile."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        summary = response.choices[0].message.content

        existing = (
            supabase
            .table("user_channel_memory_input")
            .select("userId")
            .eq("userId", userId)
            .execute()
        )

        if not existing.data:
            supabase.table("user_channel_memory_input").insert({
                "userId": userId,
                "Summary": summary
            }).execute()
            print("Channel profile created")
        else:
            supabase.table("user_channel_memory_input").update({  
                "Summary": summary
            }).eq("userId", userId).execute()
            print("Channel profile updated")

    except Exception as e:
        print(e)

async def get_chunks_from_db():
    response = supabase.table("user_channel_memory").select("text, userId").execute()
    data = response.data  

    all_chunks = [item["text"] for item in data]

    combined_text = "\n\n".join(all_chunks)

    userId = data[0]["userId"] if data else None

    await get_intelligence(combined_text, userId)