
import math
from supabase import create_client
import requests
from fastapi import FastAPI
import httpx
from fish_audio_sdk import Session, TTSRequest, Prosody 
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import contextvars
from deep_translator import GoogleTranslator
import os
from pydantic import BaseModel
import re

GENERATED_AUDIO_BUCKET = "generated-audio"


FISH_AUDIO_API_KEY = os.getenv("FISH_AUDIO_API_KEY")
FISH_AUDIO_TTS_URL = "https://api.fish.audio/v1/tts"
supabase_url_env = os.getenv("SUPABASE_URL")
supabase_key_env = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url_env, supabase_key_env)

fish_session = Session(FISH_AUDIO_API_KEY)


TTS_SPEECH_SPEED = float(os.getenv("TTS_SPEECH_SPEED", "0.95"))

VOICE_CREDITS_PER_MINUTE = 5

OPENAI_CALL_TIMEOUT = float(os.getenv("OPENAI_CALL_TIMEOUT", "45"))
async def _openai_create_with_timeout(call_fn, timeout: float = OPENAI_CALL_TIMEOUT):
    return await asyncio.wait_for(_run_io(call_fn), timeout=timeout)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0, max_retries=1)


_script_keywords_cache: contextvars.ContextVar = contextvars.ContextVar(
    "_script_keywords_cache", default=None
)


def _start_token_tracking() -> None:
    _request_token_log.set([])
    _script_keywords_cache.set({})


SUPPORTED_LANGUAGES = {
    "afrikaans": "af",
    "albanian": "sq",
    "amharic": "am",
    "arabic": "ar",
    "armenian": "hy",
    "assamese": "as",
    "aymara": "ay",
    "azerbaijani": "az",
    "bambara": "bm",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bhojpuri": "bho",
    "bosnian": "bs",
    "bulgarian": "bg",
    "catalan": "ca",
    "cebuano": "ceb",
    "chichewa": "ny",
    "chinese simplified": "zh-CN",
    "chinese traditional": "zh-TW",
    "corsican": "co",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dhivehi": "dv",
    "dogri": "doi",
    "dutch": "nl",
    "english": "en",
    "esperanto": "eo",
    "estonian": "et",
    "ewe": "ee",
    "filipino": "tl",
    "finnish": "fi",
    "french": "fr",
    "frisian": "fy",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "guarani": "gn",
    "gujarati": "gu",
    "haitian creole": "ht",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "iw",
    "hindi": "hi",
    "hmong": "hmn",
    "hungarian": "hu",
    "icelandic": "is",
    "igbo": "ig",
    "ilocano": "ilo",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "kannada": "kn",
    "kazakh": "kk",
    "khmer": "km",
    "kinyarwanda": "rw",
    "konkani": "gom",
    "korean": "ko",
    "krio": "kri",
    "kurdish kurmanji": "ku",
    "kurdish sorani": "ckb",
    "kyrgyz": "ky",
    "lao": "lo",
    "latin": "la",
    "latvian": "lv",
    "lingala": "ln",
    "lithuanian": "lt",
    "luganda": "lg",
    "luxembourgish": "lb",
    "macedonian": "mk",
    "maithili": "mai",
    "malagasy": "mg",
    "malay": "ms",
    "malayalam": "ml",
    "maltese": "mt",
    "maori": "mi",
    "marathi": "mr",
    "meiteilon/manipuri": "mni-Mtei",
    "mizo": "lus",
    "mongolian": "mn",
    "myanmar": "my",
    "nepali": "ne",
    "norwegian": "no",
    "odia": "or",
    "oromo": "om",
    "pashto": "ps",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "quechua": "qu",
    "romanian": "ro",
    "russian": "ru",
    "samoan": "sm",
    "sanskrit": "sa",
    "scots gaelic": "gd",
    "sepedi": "nso",
    "serbian": "sr",
    "sesotho": "st",
    "shona": "sn",
    "sindhi": "sd",
    "sinhala": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "spanish": "es",
    "sundanese": "su",
    "swahili": "sw",
    "swedish": "sv",
    "tajik": "tg",
    "tamil": "ta",
    "tatar": "tt",
    "telugu": "te",
    "thai": "th",
    "tigrinya": "ti",
    "tsonga": "ts",
    "turkish": "tr",
    "turkmen": "tk",
    "twi": "ak",
    "ukrainian": "uk",
    "urdu": "ur",
    "uyghur": "ug",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "xhosa": "xh",
    "yiddish": "yi",
    "yoruba": "yo",
    "zulu": "zu",
}


_request_token_log: contextvars.ContextVar = contextvars.ContextVar(
    "_request_token_log", default=None
)

def _record_token_usage(label: str, completion) -> dict:
    input_tokens = output_tokens = total_tokens = None
    try:
        usage = completion.usage
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
    except Exception as e:
        print(f"[TOKENS] {label}: could not read usage off completion ({e})")

    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    print(
        f"[TOKENS] {label}: input_tokens={input_tokens} "
        f"output_tokens={output_tokens} total_tokens={total_tokens}"
    )

    entry = {
        "label": label,
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "total_tokens": total_tokens or 0,
    }

    log = _request_token_log.get()
    if log is not None:
        log.append(entry)

    return entry





from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Lifespan] Server started.")
    yield
    print("[Lifespan] Shutting down.")

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "https://www.testing.storio.tech",
    "https://testing.storio.tech",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





class GenerateSpeechRequest(BaseModel):
    userId: str
    script: str
    voice: str
    langCode: str = "en"
    durationMinutes: int = 0
    speed: float | None = None  # FIX (voice too fast): 0.5-2.0, lower = slower; None = use TTS_SPEECH_SPEED
    volume: float | None = None  # Fish Audio prosody.volume — dB adjustment, roughly -20..20; None = Fish Audio's own default (0)
    loudnessNormalization: bool | None = None  # Fish Audio prosody.normalize_loudness; None = Fish Audio's own default
    textNormalization: bool | None = None  # Fish Audio top-level "normalize"; None = Fish Audio's own default (true)


async def _download_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


def _create_fish_model_sync(ref_audio_bytes: bytes, title: str) -> str:
    model = fish_session.create_model(
        title=title,
        description="Auto-created voice clone",
        voices=[ref_audio_bytes],
        visibility="private",
    )
    return model.id


def _run_fish_tts_sync(
    script: str,
    reference_id: str,
    speed: float = TTS_SPEECH_SPEED,
    volume: float | None = None,
    loudness_normalization: bool | None = None,
    text_normalization: bool | None = None,
) -> bytes:
    prosody_kwargs = {"speed": speed}
    if volume is not None:
        prosody_kwargs["volume"] = volume
    if loudness_normalization is not None:
        prosody_kwargs["normalize_loudness"] = loudness_normalization

    tts_request = TTSRequest(
        text=script,
        reference_id=reference_id,
        temperature=0.5,
        top_p=0.7,
        repetition_penalty=1.2,
        chunk_length=300,
        latency="normal",
        normalize=text_normalization if text_normalization is not None else True,
        format="mp3",
        mp3_bitrate=192,
        condition_on_previous_chunks=True,
        prosody=Prosody(**prosody_kwargs),
    )
    audio_chunks = []
    for chunk in fish_session.tts(tts_request):
        audio_chunks.append(chunk)
    return b"".join(audio_chunks)

def _credits_for_voice_minutes(duration_minutes: float) -> int:
    if duration_minutes <= 0:
        return 0
    credits = math.ceil(duration_minutes * VOICE_CREDITS_PER_MINUTE)
    return max(credits, 1)

import datetime

def _expire_stale_batches(batches: list[dict], now: datetime.datetime) -> list[dict]:
    active = []
    for b in batches:
        try:
            expires_at = datetime.datetime.fromisoformat(b["expires_at"])
        except Exception:
            continue
        if expires_at > now:
            active.append(b)
    return active


def _deduct_from_batches(batches: list[dict], amount: int) -> tuple[list[dict], int]:
    if _sum_batches(batches) < amount:
        return batches, 0

    sorted_batches = sorted(batches, key=lambda b: b["expires_at"])
    remaining_to_deduct = amount
    updated = []
    for b in sorted_batches:
        b = dict(b)
        if remaining_to_deduct > 0:
            take = min(b["remaining"], remaining_to_deduct)
            b["remaining"] -= take
            remaining_to_deduct -= take
        if b["remaining"] > 0:
            updated.append(b)

    return updated, amount


def _sum_batches(batches: list[dict]) -> int:
    return sum(int(b.get("remaining", 0)) for b in batches)



async def _deduct_profile_credits(user_id: str, amount: int):
    try:
        result = (
            supabase.table("user_profiles")
            .select("credit_batches")
            .eq("id", user_id)
            .single()
            .execute()
        )
        batches = (result.data or {}).get("credit_batches") or []

        now = datetime.datetime.now(datetime.timezone.utc)
        active_batches = _expire_stale_batches(batches, now)

        updated_batches, deducted = _deduct_from_batches(active_batches, amount)
        if deducted == 0:
            print(f"[CREDITS] user {user_id} has insufficient credits for {amount}, skipping deduction")
            return

        new_total = _sum_batches(updated_batches)

        supabase.table("user_profiles").update({
            "credit_batches": updated_batches,
            "credits_remaining": new_total,
        }).eq("id", user_id).execute()

        print(
            f"[CREDITS] (batch-aware FIFO) Deducted {amount} credits from user {user_id}, "
            f"oldest-expiring batch spent first, new total={new_total}"
        )
    except Exception as exc:
        print(f"[CREDITS] Failed to deduct batch credits for user {user_id}: {exc}")
        import traceback
        traceback.print_exc()


async def _deduct_credits_for_action(user_id: str, amount: int, action_label: str = "credits"):
    if amount <= 0:
        print(f"[CREDITS] ({action_label}) amount <= 0 ({amount}), skipping deduction for user {user_id}")
        return

    await _deduct_profile_credits(user_id, amount)
    print(f"[CREDITS] ({action_label}) user {user_id} — deducted {amount} credits from user_profiles only")


async def _deduct_voice_credits(user_id: str, duration_minutes: float):
    credits_to_deduct = _credits_for_voice_minutes(duration_minutes)
    if credits_to_deduct <= 0:
        print(f"[CREDITS] (voice_generation) nothing to deduct for user {user_id} (duration={duration_minutes:.2f} min)")
        return
    print(
        f"[CREDITS] (voice_generation) user {user_id} — {duration_minutes:.2f} min of audio "
        f"→ {credits_to_deduct} credits (rate: {VOICE_CREDITS_PER_MINUTE}/min)"
    )
    await _deduct_credits_for_action(user_id, credits_to_deduct, action_label="voice_generation")


def _get_public_url_sync(bucket: str, path: str) -> str:
    res = supabase.storage.from_(bucket).get_public_url(path)
    if isinstance(res, dict):
        return res.get("publicUrl") or res.get("public_url")
    return res


def _upload_audio_to_storage_sync(
    bucket: str,
    storage_path: str,
    file_bytes: bytes,
    content_type: str,
) -> None:
    supabase.storage.from_(bucket).upload(
        path=storage_path,
        file=file_bytes,
        file_options={
            "content-type": content_type or "application/octet-stream",
            "upsert": "false",
        },
    )


DEFAULT_LANGUAGE = "English"
USER_PROFILES_TABLE = "user_profiles"
USER_PROFILES_ID_COLUMN = "id"

import concurrent.futures

_IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("IO_EXECUTOR_WORKERS", "8")),
    thread_name_prefix="io",
)


async def _run_io(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_IO_EXECUTOR, lambda: fn(*args, **kwargs))

async def _user_exists_in_profiles(user_id: str | None) -> bool:
    if not user_id or not str(user_id).strip():
        return False

    user_id = str(user_id).strip()

    try:
        result = await _run_io(
            lambda: supabase.table(USER_PROFILES_TABLE)
            .select(USER_PROFILES_ID_COLUMN)
            .eq(USER_PROFILES_ID_COLUMN, user_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        print(f"[AUTH] user_profiles lookup failed for userId={user_id}: {e}")
        return False

    rows = result.data or []
    exists = len(rows) > 0

    if exists:
        print(f"[AUTH] userId={user_id} verified against '{USER_PROFILES_TABLE}.{USER_PROFILES_ID_COLUMN}'")
    else:
        print(f"[AUTH] REJECTED — userId={user_id} not found in '{USER_PROFILES_TABLE}.{USER_PROFILES_ID_COLUMN}'")

    return exists



async def require_valid_user(user_id: str | None) -> None:
    if not user_id or not str(user_id).strip():
        raise HTTPException(status_code=401, detail="userId is required")

    if not await _user_exists_in_profiles(user_id):
        raise HTTPException(
            status_code=403,
            detail="Access denied: userId not found in user_profiles",
        )

    
_LANG_CODE_TO_NAME = {v.lower(): k for k, v in SUPPORTED_LANGUAGES.items()}

def _normalize_language(language: str | None) -> str:
    if not language or not language.strip():
        return DEFAULT_LANGUAGE
    key = language.strip().lower()
    if key not in SUPPORTED_LANGUAGES:
        print(f"[LANG] unrecognized language '{language}', defaulting to English")
        return DEFAULT_LANGUAGE
    return "Odia" if key == "odia" else language.strip().title()


def _lang_name_from_code(lang_code: str) -> str:
    if not lang_code or not lang_code.strip():
        return DEFAULT_LANGUAGE
    name = _LANG_CODE_TO_NAME.get(lang_code.strip().lower())
    if not name:
        print(f"[TTS] unrecognized langCode '{lang_code}', defaulting to English")
        return DEFAULT_LANGUAGE
    return _normalize_language(name)

TRANSLATE_CHUNK_MAX_CHARS = 4000


def _chunk_text_for_translation(text_value: str, max_chars: int = TRANSLATE_CHUNK_MAX_CHARS) -> list[str]:
    """Splits on paragraph boundaries so we never cut a sentence mid-way and
    never exceed the translator's per-request character limit."""
    if len(text_value) <= max_chars:
        return [text_value]
 
    paragraphs = text_value.split("\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n" + para) if current else para
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
 


def _translate_with_library_sync(text_value: str, target_lang_code: str) -> str:
    chunks = _chunk_text_for_translation(text_value)
    translated_chunks = []
    for chunk in chunks:
        if not chunk.strip():
            translated_chunks.append(chunk)
            continue
        try:
            translated = GoogleTranslator(source="en", target=target_lang_code).translate(chunk)
            translated_chunks.append(translated or chunk)
        except Exception as e:
            print(f"[TRANSLATE] library translation failed for a chunk ({len(chunk)} chars): {e}")
            translated_chunks.append(chunk)  
    return "\n".join(translated_chunks)



def _lang_code(language: str) -> str:
    return SUPPORTED_LANGUAGES.get(language.strip().lower(), "en")

async def translate_with_library(text_value: str, target_language: str) -> str:
    if not text_value:
        return text_value
    target_lang_code = _lang_code(target_language)
    return await asyncio.to_thread(_translate_with_library_sync, text_value, target_lang_code)



TRANSLATION_QC_SYSTEM_PROMPT = """
You are a professional native {language} linguist, literary translator, localization specialist, and translation quality reviewer.

You will be given:

1. The ORIGINAL English script.
2. A DRAFT translation of that script into {language}.

Your task is NOT to translate from scratch.

Your task is to perform a professional Translation Quality Check (Translation QC), correcting only what is necessary so the final script reads as though it was originally written in {language}, while preserving the author's intent exactly.

────────────────────────
PRIMARY OBJECTIVE
────────────────────────

Produce a publication-ready {language} script that is:

• Semantically identical to the English original.
• Completely natural to native speakers.
• Fluent, idiomatic, and engaging.
• Optimized for spoken narration.
• Emotionally equivalent to the original.
• Free from machine translation artifacts.

Never:
- add information
- remove information
- change facts
- rewrite for personal preference

────────────────────────
QUALITY REQUIREMENTS
────────────────────────

1. Semantic Fidelity
- Preserve explicit and implicit meaning.
- Preserve logical relationships.
- Preserve chronology.
- Preserve cause and effect.
- Preserve emphasis.
- Preserve comparisons and negations.
- Never hallucinate.
- Never omit information.

2. Tone & Style
Preserve:
- writing style
- narrative voice
- documentary storytelling style
- emotional intensity
- suspense
- curiosity
- humor
- irony
- persuasion
- inspiration

3. Native Fluency
Rewrite unnatural machine-translated text into language that sounds completely native.

Ensure:
- natural grammar
- natural sentence structure
- natural word order
- natural punctuation
- natural vocabulary
- natural collocations
- smooth transitions
- idiomatic phrasing where appropriate

The reader should never feel the text was translated.

4. Spoken Narration
Since this script will be narrated:

- Prefer conversational but professional language.
- Improve rhythm and readability.
- Avoid awkward literal translations.
- Preserve dramatic pacing.
- Preserve emotional flow.

5. Context
Interpret every sentence using surrounding context.

Ensure:
- correct pronoun references
- correct meaning of ambiguous words
- correct domain terminology
- consistent terminology throughout the script

6. Cultural Localization
Where appropriate:

- Localize idioms naturally.
- Localize metaphors naturally.
- Replace literal expressions with native equivalents.

Do NOT localize:
- historical facts
- company names
- organization names
- government names
- product names
- official titles
unless an officially established equivalent exists.

7. Grammar
Correct:
- grammar
- spelling
- punctuation
- agreement
- tense consistency
- gender agreement
- plurality
- syntax

────────────────────────
PRESERVE EXACTLY
────────────────────────

Keep exactly as written unless an official localized form is universally used.

This includes:

- numbers
- dates
- statistics
- URLs
- email addresses
- product names
- brand names
- company names
- organization names
- government bodies
- ministry names
- official department names
- common English abbreviations
- well-known English acronyms
- programming languages
- APIs
- file names
- code
- technical standards

Examples include:

NASA
ISRO
WHO
UN
NATO
GDP
AI
ML
GPU
CPU
USB
Wi-Fi
Google
Microsoft
Windows
Linux
Android
iPhone

Keep these in English exactly as written.

Personal names should only be transliterated if that is the accepted convention in {language}.

────────────────────────
FORMATTING
────────────────────────

Preserve:

- paragraph breaks
- line breaks
- dialogue formatting
- bullet lists
- numbering
- quotation marks
- sentence boundaries

Do not merge or split paragraphs unless required for grammatically correct {language}.

────────────────────────
OUTPUT
────────────────────────

Return ONLY the corrected {language} script.

Do NOT include:

- explanations
- comments
- notes
- markdown
- analysis
- confidence scores
- comparisons
- introductory text
- closing text

Output only the final corrected translation.

"""


async def refine_translation_with_llm(
    original_text: str, draft_translation: str, target_language: str
) -> str:
    if not draft_translation:
        return draft_translation
 
    system_prompt = TRANSLATION_QC_SYSTEM_PROMPT.format(language=target_language)
    user_prompt = f"""ORIGINAL (English):
{original_text}
 
DRAFT TRANSLATION ({target_language}):
{draft_translation}
"""
 
    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
            ),
            timeout=max(OPENAI_CALL_TIMEOUT, 90.0),
        )
        _record_token_usage(f"translation_qc_{target_language.lower()}", res)
        refined = (res.choices[0].message.content or "").strip()
        return refined or draft_translation
    except Exception as e:
        print(f"[TRANSLATE] LLM grammar/QC pass failed for {target_language}: {e} — using library draft as-is")
        return draft_translation
 


async def translate_text_full_pipeline(text_value: str, target_language: str) -> str:
    if not text_value:
        return text_value
 
    target_language = _normalize_language(target_language)
    if target_language == "English":
        return text_value
 
    print(f"[TRANSLATE] translating text into {target_language} ({len(text_value)} chars)")
    try:
        draft = await translate_with_library(text_value, target_language)
        refined = await refine_translation_with_llm(text_value, draft, target_language)
        print(f"[TRANSLATE] translation into {target_language} complete ({len(refined)} chars)")
        return refined
    except Exception as e:
        print(f"[TRANSLATE] full pipeline failed for {target_language}, returning original English: {e}")
        return text_value



@app.post("/generate-speech")
async def generate_speech(body: GenerateSpeechRequest):
    userId = body.userId
    script = body.script
    voice = body.voice.strip() if body.voice else ""
    lang_code = (body.langCode or "en").strip()
    speed = body.speed if body.speed is not None else TTS_SPEECH_SPEED

    await require_valid_user(userId)

    if not script.strip():
        raise HTTPException(status_code=400, detail="script cannot be empty")

    if not voice:
        raise HTTPException(status_code=400, detail="voice cannot be empty")

    if not FISH_AUDIO_API_KEY:
        raise HTTPException(status_code=500, detail="Fish Audio API key not configured")

    target_language = _lang_name_from_code(lang_code)
    if target_language != "English":
        try:
            print(f"[TTS] translating script into {target_language} (langCode='{lang_code}') before TTS")
            script = await translate_text_full_pipeline(script, target_language)
        except Exception as e:
            print(f"[TTS] translation to {target_language} failed, using original script as-is: {e}")

    if voice.lower() == "user":
        try:
            result = await asyncio.to_thread(
                lambda: supabase.table("user_profiles")
                .select("audio-url")
                .eq("id", userId)
                .maybe_single()
                .execute()
            )
        except Exception as e:
            print(f"[TTS] failed to fetch user_profiles row: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch user profile: {e}")

        row = result.data if result else None
        audio_url = row.get("audio-url") if row else None

        if not audio_url:
            raise HTTPException(
                status_code=400,
                detail="No reference audio on file for this user. Upload one via /save-audio first.",
            )

        try:
            ref_audio_bytes = await _download_bytes(audio_url)
        except Exception as e:
            print(f"[TTS] failed to download user reference audio: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download user reference audio: {e}")

        try:
            reference_id = await asyncio.to_thread(
                _create_fish_model_sync, ref_audio_bytes, f"user-{userId}"
            )
        except Exception as e:
            print(f"[TTS] failed to create Fish Audio model: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=502, detail=f"Failed to create voice model: {e}")
    else:
        reference_id = voice

    try:
        audio_bytes = await asyncio.to_thread(_run_fish_tts_sync, script, reference_id,speed,body.volume,
            body.loudnessNormalization,
            body.textNormalization,
)
    except Exception as e:
        print(f"[TTS] Fish Audio TTS failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Fish Audio TTS failed: {e}")

    if not audio_bytes:
        raise HTTPException(status_code=502, detail="Fish Audio returned empty audio")

    storage_path = f"{userId}/{uuid.uuid4().hex}.mp3"

    try:
        await asyncio.to_thread(
            _upload_audio_to_storage_sync,
            GENERATED_AUDIO_BUCKET,
            storage_path,
            audio_bytes,
            "audio/mpeg",
        )
    except Exception as e:
        print(f"[TTS] failed to upload generated audio to storage: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save generated audio: {e}")

    try:
        public_url = await asyncio.to_thread(
            _get_public_url_sync, GENERATED_AUDIO_BUCKET, storage_path
        )
    except Exception as e:
        print(f"[TTS] failed to build public URL: {e}")
        raise HTTPException(status_code=500, detail="Generated audio saved but failed to create URL")

    if not public_url:
        raise HTTPException(status_code=500, detail="Generated audio saved but failed to create URL")

    try:
        duration_minutes = body.durationMinutes or 0
        await _deduct_voice_credits(userId, duration_minutes)
    except Exception as e:
        print(f"[TTS] credit deduction failed for user {userId}: {e}")
        import traceback
        traceback.print_exc()

    return {
        "message": "Speech generated successfully",
        "userId": userId,
        "voice": voice,
        "langCode": lang_code,
        "speed": speed,
        "reference_id": reference_id,
        "storage_path": storage_path,
        "url": public_url,
    }













































class AddScriptTagsRequest(BaseModel):
    userId: str
    script: str



SCRIPT_TAG_SYSTEM_PROMPT = f"""

# STORYBIT — FISH AUDIO S2 HUMAN PERFORMANCE ANNOTATION ENGINE

## ROLE

You are the **Storybit Voice Performance Annotation Agent**. The input is the user's **complete final script**. Convert it into a **Fish Audio S2/S2-Pro-ready performance script** by inserting carefully chosen inline performance tags so the audio sounds naturally narrated, emotionally believable, human-performed, and professionally dubbed.

You are **not** a writer, editor, summarizer, or rewriter.

## CORE RULE

Preserve the script exactly. You may **ONLY INSERT performance tags**.

Never:

* add, remove, paraphrase, reorder, or rewrite words
* change names, numbers, facts, dialogue, quotations, or meaning
* add narration or explanations
* invent emotions, sounds, or dialogue

Preserve original punctuation and structure unless a minimal punctuation change is essential for natural speech.

## FISH AUDIO S2/S2-PRO SYNTAX

Use **square brackets `[ ]`** for performance instructions.

Fish S2/S2-Pro supports **inline/localized control** and **free-form natural-language performance descriptions**, not merely a fixed tag dictionary.

Examples include:
`[pause] [short pause] [emphasis] [dramatic emphasis] [laughing] [chuckle] [inhale] [exhale] [sigh] [gasp] [whisper] [low voice] [low volume] [shouting] [screaming] [volume up] [volume down] [pitch up] [pitch down] [excited] [sad] [angry] [shocked] [surprised] [clearing throat] [tsk] [audience laughter]`

Concise custom descriptions are allowed when appropriate, e.g.:
`[quietly tense]`
`[restrained emotional tone]`
`[professional broadcast tone]`
`[quiet realization]`
`[building suspense]`

Do not create long instructions inside tags.

## ANALYZE THE ENTIRE SCRIPT FIRST

Silently identify:

* genre and narrator style
* emotional arc
* scene/idea transitions
* suspense and tension
* revelations and climaxes
* important facts and emphasis
* questions
* dialogue
* emotional moments
* pacing and conclusion

Then annotate according to the **context of the complete story**, not sentence-by-sentence isolation.

## HUMAN PERFORMANCE PRINCIPLE

**Natural/neutral narration is the default.**

Do not tag every sentence. Add a tag only when it meaningfully improves:

* emotion
* emphasis
* pacing
* tension
* realism
* conversational delivery
* narrative clarity

A human narrator does not constantly perform. Most ordinary sentences should remain untagged.

Prefer roughly:

* 0 tags for ordinary sentences
* 1 tag for meaningful delivery changes
* 1–2 tags around major moments
* occasional pauses at important boundaries

If removing a tag would sound equally natural, remove it.

## TAG PLACEMENT

Place each tag exactly where the vocal behavior should change.

Good:
`Nobody knew what would happen. [short pause] Then the door opened.`

Good:
`And then he [quiet realization] understood the truth.`

Good:
`[whisper] Nobody was supposed to know.`

Use inline placement rather than automatically placing tags at paragraph beginnings.

## PAUSES

Use pauses to reproduce natural thought and narrative timing:

`[short pause]` — brief separation
`[pause]` — meaningful pause
`[long dramatic pause]` — rare, major moment only

Use pauses for revelations, suspense, transitions, rhetorical questions, emotional realization, important statements, or before major payoffs.

Do not pause after every sentence.

## EMOTION & DELIVERY

Choose emotion from context, not keywords.

Useful controls:
`[serious] [solemn] [reflective] [authoritative] [excited] [sad] [angry] [shocked] [surprised] [fearful] [nervous] [mysterious] [ominous] [melancholic] [nostalgic] [urgent]`

For nuanced delivery use concise descriptions such as:
`[quietly tense]`
`[restrained emotion]`
`[serious documentary tone]`
`[warm conversational tone]`

Do not repeatedly restate a tone that naturally continues.

## EMPHASIS, INTENSITY & PITCH

Use `[emphasis]` or `[strong emphasis]` only on genuinely important words/phrases.

Use:
`[whisper] [soft voice] [low voice] [low volume] [loud] [shouting] [screaming] [volume up] [volume down]`

and, sparingly:
`[pitch up] [pitch down]`

Strong intensity is justified only by context. **Strongest does not mean loudest.**

## BREATH & PARALINGUISTICS

Use sparingly and only when context requires them:

`[inhale] [exhale] [deep breath] [sharp inhale] [sigh] [gasp] [chuckle] [laughing] [clearing throat] [tsk] [panting]`

Appropriate for shock, fear, exhaustion, physical action, emotional strain, or genuine conversational behavior.

Never add sounds merely to make audio seem "human."

## SUSPENSE & REVELATIONS

Build performance progressively rather than tagging everything dramatically.

Typical pattern:
normal narration → subtle tension → `[short pause]` → reveal → `[emphasis]`/`[quiet realization]` → normal narration.

Use `[dramatic]` or equivalent sparingly.

## DIALOGUE

Preserve dialogue exactly. Add delivery tags only when the dialogue clearly requires them.

Examples:
`[whisper] "Don't tell anyone."`
`[angry] "You knew."`
`[hesitant] "I... I don't know."`

Do not invent character voices or speaker labels.

## TAG COMBINATIONS

Avoid unnecessary stacking.

Bad:
`[excited] [dramatic] [loud] [pitch up] [emphasis]`

Prefer one precise instruction:
`[excited]`

Never use contradictory tags such as `[whisper] [shouting]` at the same location.

## HUMAN-NATURALNESS

The goal is **human performance, not maximum tagging**.

Create natural variation through purposeful changes in:

* pauses
* emphasis
* emotion
* intensity
* pace
* occasional breaths/reactions

Do not manufacture imperfections randomly.

## OUTPUT

Return **ONLY the fully annotated script**.

No analysis, explanation, headings, JSON, code fences, notes, summaries, or introductory text.

## FINAL VALIDATION

Before output, silently verify:

1. Every original word remains unchanged.
2. Nothing was invented, deleted, paraphrased, or reordered.
3. Tags use `[ ]`.
4. Tags are contextually justified and correctly placed.
5. Ordinary narration remains mostly untagged.
6. Pauses are natural.
7. Breaths/laughter/paralinguistic sounds are rare and justified.
8. No contradictory or excessive tags exist.
9. Major narrative beats receive appropriate performance treatment.
10. The result sounds like a skilled human narrator, not an over-directed TTS demo.
11. Output contains only the annotated script.

## INPUT

The attached/input content is the **complete final user-generated script**. Analyze the entire script first, then return the same script with only the necessary Fish Audio S2/S2-Pro performance tags inserted.
""".strip()

_TAG_PATTERN = re.compile(r"\[([^\[\]]{1,40})\]")


def _validate_tagged_script(original: str, tagged: str, min_tags: int = 3) -> bool:
    if not tagged or not tagged.strip():
        return False

    tag_matches = _TAG_PATTERN.findall(tagged)

    if len(tag_matches) < min_tags:
        print(
            f"[TAG-SCRIPT] validation failed: only {len(tag_matches)} tag(s) found "
            f"(minimum required: {min_tags})"
        )
        return False

    stripped = _TAG_PATTERN.sub("", tagged)
    stripped_words = stripped.split()
    original_words = original.split()

    if not original_words:
        return False

    len_ratio = len(stripped_words) / len(original_words)
    if len_ratio < 0.9 or len_ratio > 1.1:
        print(
            f"[TAG-SCRIPT] validation failed: word count ratio {len_ratio:.3f} "
            f"outside [0.9, 1.1] ({len(stripped_words)} vs {len(original_words)} words)"
        )
        return False

    return True


def _min_expected_tags(word_count: int) -> int:
    return max(2, word_count // 100)


def _insert_fallback_tags(script_text: str, min_tags: int) -> str:
    paragraphs = script_text.split("\n\n")
    if len(paragraphs) < 2:
        sentences = re.split(r"(?<=[.!?])\s+", script_text.strip())
        tagged_sentences = []
        tags_used = 0
        interval = max(2, len(sentences) // max(min_tags, 1))
        for i, sentence in enumerate(sentences):
            if i > 0 and i % interval == 0 and tags_used < min_tags:
                tagged_sentences.append("[pause] " + sentence)
                tags_used += 1
            else:
                tagged_sentences.append(sentence)
        return " ".join(tagged_sentences)

    tagged_paragraphs = [paragraphs[0]]
    tags_used = 0
    for para in paragraphs[1:]:
        if tags_used < min_tags and para.strip():
            tagged_paragraphs.append("[short pause] " + para)
            tags_used += 1
        else:
            tagged_paragraphs.append(para)
    return "\n\n".join(tagged_paragraphs)


def _word_count(text_value: str) -> int:
    return len(text_value.split())


async def generate_tagged_script(script_text: str) -> str:
    word_count = _word_count(script_text)
    min_tags = _min_expected_tags(word_count)

    user_prompt = f"""Script to tag (this script contains {word_count} words — you MUST insert at least {min_tags} tags total, distributed naturally throughout, not clustered):{script_text}
"""

    async def _call(extra_instruction: str = ""):
        messages = [
            {"role": "system", "content": SCRIPT_TAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt + extra_instruction},
        ]
        completion = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=messages,
                stream=False,
                temperature=0.5,
                top_p=0.9,
            ),
            timeout=max(OPENAI_CALL_TIMEOUT, 90.0),
        )
        _record_token_usage("generate_tagged_script", completion)
        return (completion.choices[0].message.content or "").strip()

    raw = ""
    try:
        raw = await _call()
    except Exception as e:
        print(f"[TAG-SCRIPT] generation failed: {e}")
        raw = ""

    if raw and not _validate_tagged_script(script_text, raw, min_tags=min_tags):
        tag_count_found = len(_TAG_PATTERN.findall(raw))
        print(
            f"[TAG-SCRIPT] attempt 1 invalid ({tag_count_found} tag(s) found, "
            f"need >= {min_tags}) — retrying with explicit correction. "
            f"Raw (truncated): {raw[:300]}"
        )
        try:
            raw = await _call(
                extra_instruction=(
                    f"\n\nIMPORTANT CORRECTION: Your previous attempt returned "
                    f"{tag_count_found} tag(s), which is BELOW the required minimum "
                    f"of {min_tags}. This is not optional — you must scan the entire "
                    f"script from beginning to end and insert at least {min_tags} tags "
                    f"from the supported list at genuine emotional or rhythmic turning "
                    f"points (paragraph openings, contrasts introduced by 'but'/'yet', "
                    f"rhetorical questions, reveals, transitions). Do not return the "
                    f"script with fewer tags than required. Preserve every original "
                    f"word exactly — only insert bracketed tags."
                )
            )
        except Exception as e:
            print(f"[TAG-SCRIPT] retry call failed: {e}")
            raw = ""

    if raw and _validate_tagged_script(script_text, raw, min_tags=min_tags):
        return raw

    tag_count_found = len(_TAG_PATTERN.findall(raw)) if raw else 0
    print(
        f"[TAG-SCRIPT] still invalid after retry ({tag_count_found} tag(s)) — "
        f"applying deterministic fallback tagging instead of returning "
        f"a fully untagged script"
    )
    return _insert_fallback_tags(script_text, min_tags)


def _get_token_usage_summary() -> dict:
    log = _request_token_log.get()
    if not log:
        return {"calls": [], "total_input_tokens": 0, "total_output_tokens": 0, "total_tokens": 0}

    total_input = sum(c["input_tokens"] for c in log)
    total_output = sum(c["output_tokens"] for c in log)
    total = sum(c["total_tokens"] for c in log)

    print(
        f"[TOKENS] REQUEST TOTAL across {len(log)} call(s): "
        f"input={total_input} output={total_output} total={total}"
    )

    return {
        "calls": log,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total,
    }




@app.post("/add-script-tags")
async def add_script_tags(request: AddScriptTagsRequest):
    await require_valid_user(request.userId)

    _start_token_tracking()

    script_text = (request.script or "").strip()
    if not script_text:
        raise HTTPException(status_code=400, detail="script must be a non-empty string")

    print(f"\n[TAG-SCRIPT] tagging script ({_word_count(script_text)} word(s)) for userId={request.userId}")

    try:
        tagged_script = await generate_tagged_script(script_text)
    except Exception as exc:
        print(f"--- /add-script-tags failed: {exc} ---")
        import traceback
        traceback.print_exc()
        return {
            "error": "Failed to generate tagged script.",
            "detail": str(exc),
            "script": script_text,
            "token_usage": _get_token_usage_summary(),
        }

    tag_count = len(_TAG_PATTERN.findall(tagged_script))
    token_usage = _get_token_usage_summary()

    print(f"[TAG-SCRIPT] done — {tag_count} tag(s) inserted")

    return {
        "tagged_script": tagged_script,
        "tag_count": tag_count,
        "word_count": _word_count(script_text),
        "token_usage": token_usage,
    }














































































import re
import os
import json
import math
import uuid
import shutil
import tempfile
import asyncio
from typing import Any, Optional, Literal

import httpx
import whisperx
import torch
from PIL import Image
import io as _io
from transformers import CLIPModel, CLIPProcessor
from fastapi import HTTPException
from pydantic import BaseModel



def _looks_like_playable_media_url(url: Optional[str]) -> bool:
    if not url:
        return False
    if "videos.pexels.com" in url or "images.pexels.com" in url:
        return True
    if re.search(r"\.(mp4|mov|webm|jpg|jpeg|png|webp)(\?|$)", url, re.IGNORECASE):
        return True
    return False


def _is_landscape_dimensions(width, height) -> bool:
    try:
        w = float(width)
        h = float(height)
    except (TypeError, ValueError):
        return False
    return w > 0 and h > 0 and w >= h * 1.2


def _resolve_broll_file_url(candidate: Optional[dict], source: Optional[str]) -> Optional[str]:
    if not candidate:
        return None

    if source == "image":
        src = candidate.get("src") or {}
        for key in ("original", "large2x", "large", "portrait", "landscape", "medium", "small", "tiny"):
            url = src.get(key)
            if url:
                return url
        return None

    if source == "video":
        video_files = candidate.get("video_files") or []
        if not video_files:
            return None

        def _area(f: dict) -> int:
            w = f.get("width") or 0
            h = f.get("height") or 0
            return w * h

        labeled_hd = [f for f in video_files if (f.get("quality") or "").lower() in ("hd", "uhd")]
        if labeled_hd:
            best = max(labeled_hd, key=_area)
            if best.get("link"):
                return best["link"]

        candidates_with_links = [f for f in video_files if f.get("link")]
        if not candidates_with_links:
            return None
        best = max(candidates_with_links, key=_area)
        return best["link"]

    return None


def _resolve_broll_file_url_any_orientation(candidate: dict, source: str) -> Optional[str]:
    if not candidate:
        return None

    existing = candidate.get("file_url")
    if _looks_like_playable_media_url(existing):
        return existing

    if source == "video":
        if candidate.get("video_url") and _looks_like_playable_media_url(candidate["video_url"]):
            return candidate["video_url"]
        video_files = candidate.get("video_files") or []
        if video_files:
            for vf in video_files:
                if vf.get("quality") == "hd" and vf.get("file_type") == "video/mp4":
                    return vf.get("link")
            return video_files[0].get("link")
        return candidate.get("url") if _looks_like_playable_media_url(candidate.get("url")) else None

    src = candidate.get("src") or {}
    for key in ("large2x", "large", "original", "medium"):
        if src.get(key):
            return src[key]
    return candidate.get("url") if _looks_like_playable_media_url(candidate.get("url")) else None


def _video_is_landscape(v: dict) -> bool:
    return _resolve_broll_file_url(v, "video") is not None


def _image_is_landscape(p: dict) -> bool:
    return _resolve_broll_file_url(p, "image") is not None



SCRIPT_SCENE_PROMPT = """ 
You are Storybit's Scene Planner, an AI that converts documentary-style
narration into (a) a single category classification for the whole video and
(b) a structured scene manifest for an automated video editing pipeline.

Your output is consumed directly by backend services, so it must be valid
JSON only with no markdown, explanations, comments, or code fences.

Step A — Classify the whole script ONCE

Choose exactly one category from this fixed list, based on the overall
subject and tone of the ENTIRE script (not any single sentence):

['anthropology', 'biography', 'business', 'economics', 'entrepreneurship', 'finance', 'health', 'knowledge', 'law', 'personal_development', 'philosophy', 'politics', 'psychology', 'self_help', 'sociology', 'history', 'religion', 'travel', 'geography', 'astronomy', 'technology', 'sports', 'communication', 'science', 'neuroscience', 'film_theatre', 'social_science', 'criminology', 'cultural_studies', 'general_documentary']

Category reference (use this to judge fit, do not invent new categories):
{'anthropology': 'Human societies, cultures, evolution, ethnography.', 'biography': "A specific person's life story.", 'business': 'Companies, corporate strategy, case studies, industry.', 'economics': 'Markets, macro/micro economics, trade, policy.', 'entrepreneurship': 'Startups, founders, building and scaling businesses.', 'finance': 'Personal finance, investing, markets, money management.', 'health': 'Medicine, wellness, fitness, nutrition.', 'knowledge': "General facts, trivia, 'did you know' style content spanning any subject.", 'law': 'Legal systems, court cases, legislation.', 'personal_development': 'Habits, growth frameworks, productivity, self-improvement systems.', 'philosophy': 'Abstract ideas, ethics, philosophers, thought experiments.', 'politics': 'Political systems, elections, government, policy.', 'psychology': 'Mind, behavior, cognitive concepts, mental processes (behavioral framing).', 'self_help': 'Direct, prescriptive advice and how-to guidance for personal problems.', 'sociology': 'Social structures, group behavior, societal trends.', 'history': 'Historical events and periods, any era.', 'religion': 'Religious traditions, theology, practices.', 'travel': 'Destinations, travel guides, culture of places.', 'geography': 'Physical geography, countries, natural formations, maps.', 'astronomy': 'Space, planets, cosmology.', 'technology': 'Tech products, engineering, innovation, computing.', 'sports': 'Sports history, athletes, competitions, stats.', 'communication': 'Language, media, rhetoric, interpersonal/mass communication.', 'science': 'General science: physics, chemistry, biology, experimentation.', 'neuroscience': 'Brain, nervous system, cognitive science (research/clinical framing).', 'film_theatre': 'Film and theatre history, analysis, industry.', 'social_science': 'Social science research and theory (methodology/research framing).', 'criminology': 'Study of crime, criminal behavior, and the justice system.', 'cultural_studies': 'Culture, identity, media/cultural analysis.', 'general_documentary': "Fallback for scripts that don't clearly fit another category."}

If nothing fits clearly, choose "general_documentary". This category applies
to the whole video and will be reused unchanged by later pipeline steps —
choose it once, carefully, from the full script.

Also detect script_language: the ISO 639-1 code of the language the
narration is actually written in (e.g. "en", "hi", "ta", "te", "ur").
Detect this from the actual text — do not assume it's English.

Step B — Segment into scenes

Objective: transform the narration into a sequence of visually coherent
scenes while preserving the original narration exactly. The output must
contain no timestamps — timing is generated later from voiceover alignment.

Scene Segmentation Rules

- Each scene's vo_text must represent NO more than approximately 2 minutes
  of spoken narration, estimated at ~140 words per minute (~280 words).
  This is a HARD limit that applies regardless of total script/video
  length — there is NO cap on the number of scenes. A 3-minute script might
  produce 2 scenes; a 20-minute script might produce 10+. Never merge
  scenes together purely to reduce scene count — the 2-minute-per-scene
  limit always wins over having fewer scenes.
- Split whenever the spoken idea or visual changes, in addition to
  splitting wherever needed to respect the 2-minute limit.
- Preserve the narration verbatim inside vo_text. Every word of the
  original script must appear in exactly one scene's vo_text, in order —
  do not drop or paraphrase any narration.

Animation density per scene

Using the category's baseline "animation_density" from the reference table
above, assign each scene a "scene_animation_density" of "low", "medium", or
"high". Default to the category's baseline. Only deviate for a specific
scene when its content clearly calls for it (e.g. a quiet emotional human
story inside an otherwise "high" density business video can be "low").

Output Schema

Return exactly one JSON object:

{{
  "category": "business",
  "script_language": "en",
  "scenes": [
    {{
      "scene_id": "s1",
      "vo_text": "Exact narration for this scene.",
      "visual_intent": "Concise documentary-style description of what should be shown.",
      "on_screen_text": "Short text overlay or empty string.",
      "requires_animation": true,
      "scene_animation_density": "medium",
      "estimated_duration_seconds": 95,
      "broll_keywords": ["query one", "query two", "query three", "query four", "query five"]
    }}
  ]
}}

Field Guidelines

script_language: ISO 639-1 code detected from the narration text itself.
This is passed unchanged to every later pipeline step — get it right once
here.

scene_id: Sequential: s1, s2, s3, ... — as many as the 2-minute-per-scene
rule requires. There is no upper bound.

vo_text: Copy the narration exactly. Do not paraphrase or rewrite.

visual_intent: Concise documentary-style search query suitable for B-roll
retrieval. Prefer real-world imagery matching the category's footage_style.
Mention important subjects, locations, time periods, or events. Avoid
cinematic adjectives like "epic" or "dramatic" unless explicitly stated.
Keep under roughly 15 words.

on_screen_text: Use only when helpful — years, dates, locations, people's
names, statistics, short titles. Otherwise "".

requires_animation: true only if the scene benefits from kinetic
typography, lower-thirds, maps, charts, timelines, or infographics.

scene_animation_density: see "Animation density per scene" above.

estimated_duration_seconds: word count of this scene's vo_text divided by
~2.33 words/sec (~140 wpm). Integer.

broll_keywords: 5-6 distinct stock-footage search phrases, 2-6 words each,
concrete and searchable, each targeting a different visual angle. These are
a scene-level seed/fallback only — beat-level keywords (generated next in the
pipeline) take precedence for actual footage rotation.

Constraints

Do not invent facts. Do not create timestamps. Do not include camera
directions unless they improve B-roll retrieval (e.g. "aerial view",
"satellite map", "close-up"). No scene may exceed ~280 words of vo_text.
Ensure the output is valid, parseable JSON.
"""


STYLE_PROFILES = {
  "anthropology": {
    "description": "Human societies, cultures, evolution, ethnography.",
    "footage_style": "Communities, ceremonies, artifacts, archaeological sites, cultural practices across regions/eras.",
    "animation_density": "low",
    "favored_animation_types": ["full_screen_quote_card", "ken_burns_pan_zoom", "lower_third", "callout_textbox"],
    "avoided_animation_types": ["stat_counter_overlay", "mascot_animation"],
  },
  "biography": {
    "description": "A specific person's life story.",
    "footage_style": "Portraits, era-appropriate settings, places tied to the person's life.",
    "animation_density": "low",
    "favored_animation_types": ["lower_third", "full_screen_quote_card", "ken_burns_pan_zoom"],
    "avoided_animation_types": ["stat_counter_overlay", "mascot_animation"],
  },
  "business": {
    "description": "Companies, corporate strategy, case studies, industry.",
    "footage_style": "Offices, meetings, product shots, people working, cities.",
    "animation_density": "high",
    "favored_animation_types": ["stat_counter_overlay", "icon_sequence", "bullet_list_reveal", "full_screen_data_viz", "callout_textbox"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "economics": {
    "description": "Markets, macro/micro economics, trade, policy.",
    "footage_style": "Markets, factories, trade, currency, charts/screens.",
    "animation_density": "high",
    "favored_animation_types": ["full_screen_data_viz", "stat_counter_overlay", "icon_sequence", "arrow_highlight"],
    "avoided_animation_types": ["mascot_animation", "emoji_reaction"],
  },
  "entrepreneurship": {
    "description": "Startups, founders, building and scaling businesses.",
    "footage_style": "Startup offices, founders working, product launches, pitching.",
    "animation_density": "high",
    "favored_animation_types": ["icon_pop_in", "stat_counter_overlay", "bullet_list_reveal", "callout_textbox"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "finance": {
    "description": "Personal finance, investing, markets, money management.",
    "footage_style": "Stock tickers, banks, currency, people managing money.",
    "animation_density": "high",
    "favored_animation_types": ["stat_counter_overlay", "full_screen_data_viz", "icon_sequence", "arrow_highlight"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "health": {
    "description": "Medicine, wellness, fitness, nutrition.",
    "footage_style": "Clinical settings, exercise, food, doctors/patients, wellness scenes.",
    "animation_density": "medium",
    "favored_animation_types": ["stat_counter_overlay", "icon_pop_in", "bullet_list_reveal", "full_screen_data_viz"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "knowledge": {
    "description": "General facts, trivia, 'did you know' style content spanning any subject.",
    "footage_style": "Broad real-world imagery matched directly to whichever fact is being discussed.",
    "animation_density": "medium",
    "favored_animation_types": ["stat_counter_overlay", "icon_pop_in", "bullet_list_reveal", "callout_textbox"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "law": {
    "description": "Legal systems, court cases, legislation.",
    "footage_style": "Courtrooms, legal documents, government/court buildings.",
    "animation_density": "medium",
    "favored_animation_types": ["full_screen_document_highlight", "lower_third", "callout_textbox", "bullet_list_reveal"],
    "avoided_animation_types": ["mascot_animation", "emoji_reaction"],
  },
  "personal_development": {
    "description": "Habits, growth frameworks, productivity, self-improvement systems.",
    "footage_style": "Everyday life, people building routines, journaling, incremental progress.",
    "animation_density": "medium",
    "favored_animation_types": ["bullet_list_reveal", "icon_pop_in", "callout_textbox", "stat_counter_overlay"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "philosophy": {
    "description": "Abstract ideas, ethics, philosophers, thought experiments.",
    "footage_style": "Contemplative real-world imagery, historical settings, symbolic everyday scenes.",
    "animation_density": "low",
    "favored_animation_types": ["full_screen_quote_card", "lower_third", "ken_burns_pan_zoom"],
    "avoided_animation_types": ["stat_counter_overlay", "mascot_animation", "icon_sequence"],
  },
  "politics": {
    "description": "Political systems, elections, government, policy.",
    "footage_style": "Government buildings, rallies, officials, maps.",
    "animation_density": "medium",
    "favored_animation_types": ["lower_third", "full_screen_data_viz", "callout_textbox", "arrow_highlight"],
    "avoided_animation_types": ["mascot_animation", "emoji_reaction"],
  },
  "psychology": {
    "description": "Mind, behavior, cognitive concepts, mental processes (behavioral framing).",
    "footage_style": "People and everyday behavior/interactions, relatable real-world scenes.",
    "animation_density": "medium",
    "favored_animation_types": ["icon_pop_in", "callout_textbox", "bullet_list_reveal", "full_screen_data_viz"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "self_help": {
    "description": "Direct, prescriptive advice and how-to guidance for personal problems.",
    "footage_style": "Relatable everyday life, people applying advice/techniques.",
    "animation_density": "medium",
    "favored_animation_types": ["bullet_list_reveal", "callout_textbox", "icon_pop_in"],
    "avoided_animation_types": ["full_screen_data_viz", "mascot_animation"],
  },
  "sociology": {
    "description": "Social structures, group behavior, societal trends.",
    "footage_style": "Communities, social settings, crowds, institutions.",
    "animation_density": "medium",
    "favored_animation_types": ["full_screen_data_viz", "stat_counter_overlay", "lower_third", "callout_textbox"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "history": {
    "description": "Historical events and periods, any era.",
    "footage_style": "Archival-style or era-appropriate imagery, artifacts, maps, timelines — matched to whichever era the specific script covers.",
    "animation_density": "medium",
    "favored_animation_types": ["full_screen_quote_card", "ken_burns_pan_zoom", "lower_third", "full_screen_data_viz"],
    "avoided_animation_types": ["mascot_animation", "emoji_reaction"],
  },
  "religion": {
    "description": "Religious traditions, theology, practices.",
    "footage_style": "Religious sites, symbols, ceremonies, texts.",
    "animation_density": "low",
    "favored_animation_types": ["full_screen_quote_card", "ken_burns_pan_zoom", "lower_third"],
    "avoided_animation_types": ["stat_counter_overlay", "mascot_animation", "emoji_reaction"],
  },
  "travel": {
    "description": "Destinations, travel guides, culture of places.",
    "footage_style": "Landmarks, landscapes, street scenes, local life.",
    "animation_density": "low",
    "favored_animation_types": ["lower_third", "ken_burns_pan_zoom", "callout_textbox"],
    "avoided_animation_types": ["stat_counter_overlay", "mascot_animation"],
  },
  "geography": {
    "description": "Physical geography, countries, natural formations, maps.",
    "footage_style": "Landscapes, maps, satellite-style views, natural formations.",
    "animation_density": "medium",
    "favored_animation_types": ["full_screen_data_viz", "lower_third", "arrow_highlight", "ken_burns_pan_zoom"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "astronomy": {
    "description": "Space, planets, cosmology.",
    "footage_style": "Space imagery, telescopes, night sky, planetary/scale visuals.",
    "animation_density": "high",
    "favored_animation_types": ["full_screen_data_viz", "icon_sequence", "stat_counter_overlay", "arrow_highlight"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "technology": {
    "description": "Tech products, engineering, innovation, computing.",
    "footage_style": "Devices, labs, close-ups of tech, digital interfaces.",
    "animation_density": "high",
    "favored_animation_types": ["full_screen_data_viz", "icon_sequence", "stat_counter_overlay", "arrow_highlight"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "sports": {
    "description": "Sports history, athletes, competitions, stats.",
    "footage_style": "Sports action, athletes, stadiums, equipment.",
    "animation_density": "high",
    "favored_animation_types": ["stat_counter_overlay", "lower_third", "full_screen_data_viz", "arrow_highlight"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "communication": {
    "description": "Language, media, rhetoric, interpersonal/mass communication.",
    "footage_style": "People talking, media/broadcast settings, writing, signals.",
    "animation_density": "medium",
    "favored_animation_types": ["icon_pop_in", "callout_textbox", "bullet_list_reveal", "lower_third"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "science": {
    "description": "General science: physics, chemistry, biology, experimentation.",
    "footage_style": "Labs, experiments, natural phenomena, close-ups of mechanisms.",
    "animation_density": "high",
    "favored_animation_types": ["full_screen_data_viz", "icon_sequence", "stat_counter_overlay", "arrow_highlight"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "neuroscience": {
    "description": "Brain, nervous system, cognitive science (research/clinical framing).",
    "footage_style": "Brain/medical imagery, labs, research settings.",
    "animation_density": "high",
    "favored_animation_types": ["full_screen_data_viz", "icon_pop_in", "stat_counter_overlay", "arrow_highlight"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "film_theatre": {
    "description": "Film and theatre history, analysis, industry.",
    "footage_style": "Theatres, film sets, performances, era-appropriate cinema imagery.",
    "animation_density": "medium",
    "favored_animation_types": ["full_screen_quote_card", "lower_third", "callout_textbox", "ken_burns_pan_zoom"],
    "avoided_animation_types": ["stat_counter_overlay", "mascot_animation"],
  },
  "social_science": {
    "description": "Social science research and theory (methodology/research framing).",
    "footage_style": "Research settings, communities, data-adjacent real-world imagery.",
    "animation_density": "medium",
    "favored_animation_types": ["full_screen_data_viz", "callout_textbox", "lower_third", "bullet_list_reveal"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "criminology": {
    "description": "Study of crime, criminal behavior, and the justice system.",
    "footage_style": "Evidence-style imagery, courtrooms, investigation settings, documents.",
    "animation_density": "medium",
    "favored_animation_types": ["full_screen_document_highlight", "lower_third", "callout_textbox", "arrow_highlight"],
    "avoided_animation_types": ["mascot_animation", "emoji_reaction", "icon_sequence"],
  },
  "cultural_studies": {
    "description": "Culture, identity, media/cultural analysis.",
    "footage_style": "Cultural settings, communities, symbols, everyday life across cultures.",
    "animation_density": "medium",
    "favored_animation_types": ["lower_third", "callout_textbox", "full_screen_quote_card", "bullet_list_reveal"],
    "avoided_animation_types": ["mascot_animation"],
  },
  "general_documentary": {
    "description": "Fallback for scripts that don't clearly fit another category.",
    "footage_style": "Real-world footage matched directly to narration subjects.",
    "animation_density": "low",
    "favored_animation_types": ["lower_third", "ken_burns_pan_zoom", "full_screen_quote_card"],
    "avoided_animation_types": ["mascot_animation"],
  },
}


BEAT_KEYWORDS_PROMPT = """

You are Storybit's Beat Director for a single scene of a documentary-style
video. Unlike a fixed clock grid, YOU decide how many beats this scene
splits into and roughly how long each beat should run, based on how much
visual/narrative complexity each stretch of narration carries.

You are given, as JSON:
- category: the video's overall category (fixed for the whole video).
- style_profile: the STYLE_PROFILES entry for that category (footage_style,
  animation_density baseline, favored/avoided animation types).
- script_language: ISO 639-1 code of the narration's language (fixed for
  the whole video).
- scene_id, scene_visual_intent, scene_animation_density: from the Scene
  Planner.
- scene_on_screen_text: the Scene Planner's scene-level suggestion for
  overlay-worthy content (a year, date, name, statistic, or short title) —
  "" if it flagged nothing. Treat this as a checklist, not literal text to
  copy: if scene_on_screen_text is non-empty, check whether the content it
  names actually appears in this beat's vo_text; if so, that beat's
  animation_signal should lean toward needs_animation=true with
  key_subject capturing that exact content (so it isn't lost between the
  scene-level suggestion and the beat-level decision). Don't force every
  beat to react to it — only the beat(s) whose vo_text actually contains
  it.
- scene_vo_text: the full exact narration for this scene.
- previous_scene_last_media_type: "video", "image", or null (null only for
  the very first scene) — the media_type of the last beat in the scene
  immediately before this one.
- known_entities: an array of {"name": ..., "entity_type": ...} accumulated
  from every scene processed so far in this video ([] for the very first
  scene). Use this to stay consistent — if a name here was already
  classified as "real_person", don't reclassify it as "fictional_character"
  (or vice versa) if it recurs in this scene.
- known_setting: {"location": ..., "time_period": ...} established by the
  most recent prior scene ({"location": "", "time_period": ""} for the
  very first scene). Use this to stay consistent — don't drift the
  location/era without the narration itself explicitly moving the story.

Your job

1. Segment scene_vo_text into a sequence of beats. Typical beat length is
   8-20 seconds of spoken narration (~19-47 words at ~140 wpm), but let
   content decide: a dense idea that will carry a detailed animation or an
   infographic can run toward the long end (up to ~20s); a fast-moving or
   simple stretch of narration can be a short beat (down to ~8s). Do not
   force a fixed count or fixed length — vary it scene to scene.
2. Preserve scene_vo_text verbatim across beats — every word must appear in
   exactly one beat's vo_text, in order, with nothing dropped, paraphrased,
   or duplicated.
3. For each beat, generate fresh B-roll keywords specific to that beat.
4. For each beat, decide media_type ("video" or "image").
5. For each beat, emit an animation SIGNAL — not a final decision. A later
   step (Animation Planner) has final authority and may accept, reject, or
   change what you propose here.
6. For each beat, direct the scene: identify characters/elements, setting,
   mood, and the central action (see "Scene Direction" below) — everything
   needed to visually construct this beat, not just who's named in it.

Scene Direction

You are directing this beat, not just labeling it. Provide:

- entities: every named character or significant recurring object in this
  beat's vo_text, each classified:
  - "real_person": an actual historical or contemporary person (e.g.
    Cleopatra, a scientist, a CEO named in the script).
  - "fictional_character": a character from a story, myth-as-narrative-
    device, or invented scenario the script is telling (not the same as a
    real mythological/religious figure being discussed factually — judge
    by whether the narration asserts the figure existed/acted, or is
    explicitly telling a fictional/illustrative story).
  - "element": a significant recurring object, place, or symbol that isn't
    a person but matters visually across the narration (e.g. "the Nile",
    "a locked vault", "the company's first office").
  Use known_entities to stay consistent: reuse an existing name's
  entity_type rather than reclassifying it. Only list what's new or
  freshly relevant in THIS beat — don't repeat an earlier beat's entity
  just because it was mentioned before in the scene.
- setting: {"location": ..., "time_period": ...} — where and when this
  beat visually takes place, grounded in what the narration states or
  strongly implies. Use "" for either field when the narration doesn't
  establish it. Use known_setting to stay consistent: don't drift the
  location/era from what's already established unless the narration
  itself explicitly moves the story somewhere/somewhen else.
- mood: one or two words for this beat's emotional/atmospheric tone (e.g.
  "tense", "triumphant", "solemn", "playful", "urgent"), grounded in the
  narration's actual content and phrasing — never invented flavor beyond
  what the words support. Use "neutral" if nothing distinct comes through.
- key_action: one concise sentence describing the central visual
  event/activity happening in this beat — what should be SHOWN occurring,
  not what's being said. This is often not a named entity at all (e.g.
  "a ship sinking during a storm", "a verdict being read in a courtroom",
  "two founders shaking hands over a contract") and should ground at least
  one of this beat's keyword phrases when it names a concrete action.

Keyword Rules (accuracy is critical — bad keywords produce wrong footage)

- Each phrase: 2-6 words, concrete, real-world, photographable — never an
  abstract concept on its own.
- Every phrase must be grounded in a concrete noun or subject literally
  present, or very directly implied, in this beat's vo_text.
- If this beat's narration names a specific person, place, object, or
  event, at least one phrase must center on that exact subject, and at
  least one other phrase should combine that subject with the category's
  era/style (footage_style) rather than being purely generic. Example: if
  category is "history" and the beat mentions Cleopatra, include
  something like "Cleopatra ancient Egyptian portrait" or "ancient Egyptian
  queen depiction" — not just "Cleopatra" alone (too likely to surface
  modern/unrelated results) and not just generic "ancient Egypt" (loses the
  specific subject).
- If the beat doesn't introduce new visual content beyond the scene's
  overall visual_intent, still produce phrases giving a DIFFERENT concrete
  angle on the same subject (wide vs. close-up, a related real subject, a
  different moment in the same activity) — never just repeat the scene-level
  query generically.
- No cinematic adjectives. Prefer real, photographable subjects over
  metaphors. Do not invent facts, names, dates, or subjects not present in
  the text. Always return 5-6 phrases.
- Always write keywords in ENGLISH, regardless of script_language. Stock
  footage libraries are English-indexed and search quality drops sharply
  on non-English queries — translate the concrete subject into English
  even when scene_vo_text is in another language.
- For a "real_person" entity, search using their actual name plus the
  category's era/style (the existing Cleopatra rule above).
- For a "fictional_character" entity, do NOT search using the character's
  name — no stock library has a photo of someone who doesn't exist, and a
  literal-name search returns irrelevant results. Instead, generate
  keywords describing the SETTING, MOOD, or visual archetype the narration
  implies for that moment (e.g. a shadowy detective scene → "person in
  trench coat silhouette", "rain-soaked city street at night" — not the
  character's invented name).
- At least one keyword phrase should reflect key_action (the central
  visual event you identified in Scene Direction) when it names a concrete
  activity — footage of the ACTION happening is often more useful than
  footage of just the subject standing still. Fold in setting.location
  and mood where they sharpen the search (e.g. "verdict being read" +
  "wood-paneled courtroom" for a tense key_action set in a courtroom).

Media Type Rules

Choose "image" for: a specific static subject best shown as a single frame
(portrait, named person, document, building exterior, product, a
statistic/data point, a historical/archival moment), or a quote/definition/
reflective beat with no motion implied.

Choose "video" for: motion, action, process, or change over time.

Do not default to "video" out of habit. Avoid repeating the same
media_type as the immediately preceding beat unless this beat's content
clearly calls for it regardless of variety — consecutive beats should
visually vary (motion, then a still, then motion again). This applies at
the scene boundary too: this scene's FIRST beat should avoid repeating
previous_scene_last_media_type unless its content clearly calls for it.

Animation Signal Rules

Set needs_animation based on whether this beat's content would genuinely
benefit from an overlay/full-screen animation treatment (not just B-roll) —
weigh this against scene_animation_density: a "low" density scene should
have animation signals on only a small minority of beats; "high" density
can flag most beats.

- intent: one short sentence — what the animation would communicate or
  emphasize (e.g. "highlight the specific casualty statistic just spoken").
- suggested_category: your best guess at one of: full_screen, overlay_text,
  overlay_graphic, pip, branding, transition — or null if needs_animation
  is false. This is a suggestion; the Animation Planner may override it.
- key_subject: the exact entity, number, quote, or short phrase from this
  beat's vo_text that the animation should center on or highlight. Ground
  this in the literal text — do not invent.
- source_type: "quoted_source" if this beat is narrating or referencing a
  specific quote from an article, study, publication, or named source that
  should be visually presented as that source (e.g. "a report from X
  found..." or a direct quotation); otherwise "narrative".
- quoted_excerpt: if source_type is "quoted_source", the exact quoted text
  from vo_text (verbatim substring). Otherwise null.
- source_name_guess: if source_type is "quoted_source", the name of the
  publication, study, report, or person the narration attributes this
  quote to, exactly as stated in vo_text (e.g. "Reuters", "a Harvard
  study", "the WHO report"). This is used downstream to attempt locating a
  real source page to screenshot — a real screenshot asset, not a mockup.
  If vo_text doesn't name a specific source, return null (the downstream
  system will need a fallback treatment in that case).
- key_subject_entity_type: if key_subject corresponds to one of this
  beat's identified entities, its entity_type ("real_person",
  "fictional_character", or "element"); otherwise null. This lets the
  Animation Planner choose a treatment appropriate to whether the subject
  is real (documentary-style, photo-realistic footage/animation) or
  fictional (illustrative/symbolic treatment, since no real depiction
  exists).

Output Schema

Return exactly one JSON object, nothing else — no markdown, no code fences:

{
  "beats": [
    {
      "beat_id": "s1_beat1",
      "vo_text": "Exact narration for this beat.",
      "estimated_duration_seconds": 12,
      "keywords": ["query one", "query two", "query three", "query four", "query five"],
      "media_type": "video",
      "entities": [
        {"name": "Cleopatra", "entity_type": "real_person"}
      ],
      "scene_direction": {
        "setting": {"location": "the royal palace in Alexandria", "time_period": "1st century BC"},
        "mood": "tense",
        "key_action": "Cleopatra negotiating with a Roman envoy"
      },
      "animation_signal": {
        "needs_animation": true,
        "intent": "Emphasize the named statistic just spoken.",
        "suggested_category": "overlay_text",
        "key_subject": "40 percent increase",
        "source_type": "narrative",
        "quoted_excerpt": null,
        "source_name_guess": null,
        "key_subject_entity_type": null
      }
    }
  ]
}

Constraints

Beats must be in narration order, contiguous, and non-overlapping.
estimated_duration_seconds is a hint for downstream timing reconciliation
against real voiceover alignment — not a hard timestamp. entities may be
an empty list [] when a beat introduces no new named character/element —
don't force an entry. scene_direction.setting fields may be "" when not
established; mood and key_action are always required (use "neutral" for
mood and a plain description of what's being said/shown for key_action if
nothing more specific applies — never leave them blank). Do not invent
facts. Ensure the output is valid, parseable JSON.
"""


ANIMATION_TAXONOMY = {
  "full_screen": [
    "full_screen_broll", "full_screen_title_card", "full_screen_data_viz",
    "full_screen_transition", "full_screen_color_wash", "full_screen_quote_card",
    "full_screen_document_highlight",
  ],
  "overlay_text": [
    "lower_third", "kinetic_caption", "bullet_list_reveal", "callout_textbox",
    "stat_counter_overlay",
  ],
  "overlay_graphic": [
    "icon_pop_in", "icon_sequence", "logo_watermark", "emoji_reaction",
    "arrow_highlight", "badge_sticker",
  ],
  "pip": ["pip_video", "split_screen", "multi_panel_grid"],
  "branding": ["avatar_overlay", "mascot_animation"],
  "transition": ["ken_burns_pan_zoom", "parallax_layering", "shake_impact", "speed_ramp_indicator"],
}

_ANIMATION_TYPE_TO_CATEGORY = {
    anim_type: category
    for category, anim_types in ANIMATION_TAXONOMY.items()
    for anim_type in anim_types
}
_VALID_ANIMATION_TYPES = set(_ANIMATION_TYPE_TO_CATEGORY.keys())
_VALID_PLACEMENTS = {
    "top_left", "top_center", "top_right",
    "center_left", "center", "center_right",
    "bottom_left", "bottom_center", "bottom_right",
    "full_frame",
}
_VALID_Z_LAYERS = {"background", "midground", "foreground"}
_VALID_TRIGGERS = {"time_offset", "on_keyword", "on_beat", "scene_start"}
_VALID_RENDER_HINTS = {"remotion", "ffmpeg"}
_VALID_ICON_LAYOUTS = {"sequence", "cluster", "pair"}

_ICON_LIBRARY_GROUPS = {
    "general_ui": ["arrow-right", "bell", "calendar", "camera", "check", "clock", "infinity", "lightbulb", "puzzle", "quote", "search", "sparkles", "target", "timer", "x"],
    "emotion_people": ["crown", "handshake", "heart", "heart-handshake", "user", "users", "users-round"],
    "business_finance_economics": ["banknote", "bar-chart", "briefcase", "building", "building-2", "chart-column", "chart-line", "coins", "credit-card", "dollar-sign", "factory", "line-chart", "pie-chart", "piggy-bank", "receipt", "trending-down", "trending-up", "wallet"],
    "law_criminology": ["archive", "file-text", "fingerprint-pattern", "folder", "gavel", "key", "lock", "scale", "shield"],
    "science_technology_neuroscience": ["atom", "battery", "brain", "brain-circuit", "code", "cpu", "database", "dna", "flask-conical", "gauge", "laptop", "microscope", "network", "server", "smartphone", "terminal", "wifi", "zap"],
    "astronomy_geography": ["compass", "globe", "map", "map-pin", "map-pinned", "moon", "moon-star", "mountain", "orbit", "rocket", "satellite", "snowflake", "star", "sun", "telescope", "thermometer", "waves", "wind"],
    "health": ["activity", "cross", "leaf", "pill", "sprout", "stethoscope"],
    "history_religion_anthropology_culture": ["castle", "church", "flag", "landmark", "scroll", "sword", "vote"],
    "travel_sports": ["anchor", "award", "bus", "car", "dumbbell", "luggage", "medal", "plane", "ribbon", "ship", "train", "trophy", "umbrella"],
    "communication_film_media": ["clapperboard", "drama", "film", "message-circle", "mic", "newspaper", "podcast", "radio", "rss", "theater", "tv"],
    "knowledge_education": ["book", "graduation-cap", "library", "pen", "pencil"],
    "nature_misc": ["bird", "bug", "cat", "coffee", "dog", "fish", "flame", "music", "paintbrush", "palette", "pizza", "shirt", "tent", "utensils"],
    "alerts": ["alert-triangle", "megaphone"],
}
_ICON_VOCAB = {icon for group in _ICON_LIBRARY_GROUPS.values() for icon in group}

# icon_name -> emoji glyph, used by the FFmpeg fallback renderer below when
# Remotion isn't available. Every key must exist in _ICON_VOCAB.
_ICON_EMOJI_FALLBACK = {
    "arrow-right": "\u27a1", "bell": "\U0001f514", "calendar": "\U0001f4c5", "camera": "\U0001f4f7",
    "check": "\u2705", "clock": "\U0001f550", "infinity": "\u267e", "lightbulb": "\U0001f4a1",
    "puzzle": "\U0001f9e9", "quote": "\U0001f4ac", "search": "\U0001f50d", "sparkles": "\u2728",
    "target": "\U0001f3af", "timer": "\u23f2", "x": "\u274c",
    "crown": "\U0001f451", "handshake": "\U0001f91d", "heart": "\u2764", "heart-handshake": "\U0001f491",
    "user": "\U0001f464", "users": "\U0001f465", "users-round": "\U0001f465",
    "banknote": "\U0001f4b5", "bar-chart": "\U0001f4ca", "briefcase": "\U0001f4bc", "building": "\U0001f3e2",
    "building-2": "\U0001f3ec", "chart-column": "\U0001f4ca", "chart-line": "\U0001f4c8", "coins": "\U0001fa99",
    "credit-card": "\U0001f4b3", "dollar-sign": "\U0001f4b2", "factory": "\U0001f3ed", "line-chart": "\U0001f4c8",
    "pie-chart": "\U0001f4c8", "piggy-bank": "\U0001f437", "receipt": "\U0001f9fe", "trending-down": "\U0001f4c9",
    "trending-up": "\U0001f4c8", "wallet": "\U0001f45b",
    "archive": "\U0001f5c4", "file-text": "\U0001f4c4", "fingerprint-pattern": "\U0001faf2", "folder": "\U0001f4c1",
    "gavel": "\U0001f528", "key": "\U0001f511", "lock": "\U0001f512", "scale": "\u2696", "shield": "\U0001f6e1",
    "atom": "\u269b", "battery": "\U0001f50b", "brain": "\U0001f9e0", "brain-circuit": "\U0001f9e0",
    "code": "\U0001f4bb", "cpu": "\U0001f5a5", "database": "\U0001f5c3", "dna": "\U0001f9ec",
    "flask-conical": "\U0001f9ea", "gauge": "\U0001f4dd", "laptop": "\U0001f4bb", "microscope": "\U0001f52c",
    "network": "\U0001f310", "server": "\U0001f5a5", "smartphone": "\U0001f4f1", "terminal": "\u2328",
    "wifi": "\U0001f4f6", "zap": "\u26a1",
    "compass": "\U0001f9ed", "globe": "\U0001f30d", "map": "\U0001f5fa", "map-pin": "\U0001f4cd",
    "map-pinned": "\U0001f4cd", "moon": "\U0001f319", "moon-star": "\U0001f319", "mountain": "\u26f0",
    "orbit": "\U0001fa90", "rocket": "\U0001f680", "satellite": "\U0001f6f0", "snowflake": "\u2744",
    "star": "\u2b50", "sun": "\u2600", "telescope": "\U0001f52d", "thermometer": "\U0001f321",
    "waves": "\U0001f30a", "wind": "\U0001f4a8",
    "activity": "\U0001f4c9", "cross": "\u271d", "leaf": "\U0001f343", "pill": "\U0001f48a",
    "sprout": "\U0001f331", "stethoscope": "\U0001fa7a",
    "castle": "\U0001f3f0", "church": "\u26ea", "flag": "\U0001f6a9", "landmark": "\U0001f3db",
    "scroll": "\U0001f4dc", "sword": "\u2694", "vote": "\U0001f5f3",
    "anchor": "\u2693", "award": "\U0001f3c5", "bus": "\U0001f68c", "car": "\U0001f697", "dumbbell": "\U0001f3cb",
    "luggage": "\U0001f9f3", "medal": "\U0001f3c5", "plane": "\u2708", "ribbon": "\U0001f397",
    "ship": "\U0001f6a2", "train": "\U0001f686", "trophy": "\U0001f3c6", "umbrella": "\u2602",
    "clapperboard": "\U0001f3ac", "drama": "\U0001f3ad", "film": "\U0001f39e", "message-circle": "\U0001f4ac",
    "mic": "\U0001f3a4", "newspaper": "\U0001f4f0", "podcast": "\U0001f399", "radio": "\U0001f4fb",
    "rss": "\U0001f4e1", "theater": "\U0001f3ad", "tv": "\U0001f4fa",
    "book": "\U0001f4d6", "graduation-cap": "\U0001f393", "library": "\U0001f4da", "pen": "\U0001f58a",
    "pencil": "\u270f",
    "bird": "\U0001f426", "bug": "\U0001f41b", "cat": "\U0001f431", "coffee": "\u2615", "dog": "\U0001f436",
    "fish": "\U0001f41f", "flame": "\U0001f525", "music": "\U0001f3b5", "paintbrush": "\U0001f58c",
    "palette": "\U0001f3a8", "pizza": "\U0001f355", "shirt": "\U0001f455", "tent": "\u26fa",
    "utensils": "\U0001f374",
    "alert-triangle": "\u26a0", "megaphone": "\U0001f4e2",
}
_DEFAULT_ICON_EMOJI = "\u2b50"


def _icon_glyph(icon_name: str) -> str:
    return _ICON_EMOJI_FALLBACK.get(icon_name, _DEFAULT_ICON_EMOJI)


_TEXT_ANIMATION_STYLES = [
    "fade_in", "slide_in_left", "slide_in_right", "slide_up", "slide_down",
    "zoom_in", "bounce", "pop", "typewriter", "wipe",
]

ANIMATION_CANVAS_WIDTH = 1920
ANIMATION_CANVAS_HEIGHT = 1080
CAPTION_SAFE_ZONE_Y = 918  # burned-in captions occupy the bottom 162px

PLACEMENT_ANCHORS_PX = {
    "top_left": (64, 64), "top_center": (960, 64), "top_right": (1856, 64),
    "center_left": (64, 540), "center": (960, 540), "center_right": (1856, 540),
    "bottom_left": (64, 854), "bottom_center": (960, 854), "bottom_right": (1856, 854),
    "full_frame": (0, 0),
}

ANIMATION_PLANNER_PROMPT = f"""
You are Storybit's Animation Director — the creative director of this
video, and the FINAL authority on animation decisions for this scene. The
rendering pipeline (hybrid Remotion + FFmpeg) draws exactly what you return
here — there is no other layout logic downstream.

You are given, as JSON:
- category, style_profile (footage_style, animation_density baseline,
  favored_animation_types, avoided_animation_types).
- script_language: ISO 639-1 code of the narration's language (fixed for
  the whole video).
- scene_id, scene_visual_intent, scene_on_screen_text, requires_animation,
  scene_animation_density.
- beats: the full array of this scene's beats from the Beat Director, each
  with beat_id, vo_text, estimated_duration_seconds, entities,
  scene_direction (setting, mood, key_action), and animation_signal
  (needs_animation, intent, suggested_category, key_subject, source_type,
  quoted_excerpt, source_name_guess, key_subject_entity_type).
- previous_scene_last_animation: null (first scene only), or
  {{"animation_type": ..., "placement": ..., "category": ...}} for the last
  animated beat of the immediately preceding scene.

Language rule: display_text must be written in script_language — it is
on-screen text tied to spoken narration, so it must match what the viewer
is hearing. icon_name (fixed vocabulary), content_binding, and
render_prompt stay in English regardless of script_language, since they
are internal/documentation values, not viewer-facing text.

Canvas: every video is 1920x1080px (16:9). All
positions and sizes you return must be real pixel values on this canvas —
not vague fractions or percentages.

Placement anchor reference (top-left corner in px for each placement zone,
before you add your own width/height offset):
{PLACEMENT_ANCHORS_PX}

Burned-in captions always occupy the bottom 162px of the
frame (y >= 918). For overlay_text/overlay_graphic types,
geometry_px must keep (y + height) at or above 918 minus
a small buffer — this is the same caption safe-zone rule as before, just
now expressed in exact pixels instead of only a placement name.

Your output is consumed by a deterministic renderer, not another LLM —
there is no interpretation step between your JSON and the rendered clip.
That means icon_name, display_text, and color_hint (defined below) must be
exact, final values, not prose for something else to parse.

Icon vocabulary: if this animation involves an icon (overlay_graphic or
branding category), every icon_name value must come from this fixed set —
nothing outside it will resolve to a real component prop:
{sorted(_ICON_VOCAB)}

Mixing icons for richer visuals: you are not limited to one icon per
animation. icon_name may be either a single string (the common case) or
an array of 2-4 icon names when a beat's content genuinely benefits from
combining concepts visually — e.g. "briefcase" + "trending-up" for a
promotion, "brain" + "lightbulb" for a psychological insight, "globe" +
"handshake" for an international deal. When icon_name is "icon_sequence",
it must be an array (that animation_type exists specifically to reveal
multiple icons in order). For other overlay_graphic types, use an array
only when a single icon can't carry the idea — don't mix icons just
because you can; an unearned combination reads as cluttered, not rich.
When icon_name is an array, set icon_layout to one of "sequence" (icons
appear one after another, cascading), "cluster" (icons appear together at
once, grouped), or "pair" (exactly two icons side by side representing a
relationship). icon_layout is null when icon_name is a single string.

Mood-aware styling: each beat carries a scene_direction.mood from the Beat
Director (e.g. "tense", "triumphant", "solemn", "playful"). Let it inform
color_hint (e.g. warmer/brighter for triumphant or playful moods, cooler
or higher-contrast for tense or solemn ones) and motion_style (e.g. sharp/
quick for urgent moods, slow/gentle for reflective ones) — within what the
category's footage_style/tone otherwise allows. This is a styling input,
not a license to override the category's established palette.

Real vs. fictional subjects: check key_subject_entity_type on the beat's
animation_signal. If it's "fictional_character", do not choose a
treatment that implies photographic reality (e.g. full_screen_broll
framed as if showing that character) — no real footage of them exists.
Prefer an illustrative/symbolic treatment instead: an icon-based animation
representing the idea/mood, a full_screen_quote_card, or on-screen text —
something that doesn't claim to depict a real image of someone who isn't
real. "real_person" and "element" subjects have no such restriction.

Your job

Hard gate first: if requires_animation (from the Scene Planner, passed in
as a scene-level field) is false, return an empty animations list for this
entire scene — no exceptions, regardless of scene_animation_density or any
beat's animation_signal. requires_animation is the scene-level "should
this scene have animation treatment AT ALL" decision; it overrides
everything below. Only proceed past this point if requires_animation is
true.

For each beat, decide whether it actually gets an animation. The beat's
animation_signal is a PROPOSAL from an earlier step, not a final decision —
you may accept it, reject it (needs_animation was true but you judge it
unnecessary), or add one it didn't flag, if the scene's overall
animation_density budget calls for it. As a guide: "low" density scenes
should end up with animation on roughly one beat, "medium" on a couple,
"high" on most beats — but use judgment over rigid counts.

Prefer favored_animation_types for this category and avoid
avoided_animation_types unless the specific beat content overrides that
default.

Vary treatment across beats within the scene — do not give consecutive
animated beats the same animation_type/placement combination back to back
unless the content specifically calls for repeating it, so the video keeps
visibly changing rather than looking static. This also applies at the
scene boundary: if this scene's first animated beat would otherwise match
previous_scene_last_animation's animation_type AND placement exactly,
change at least one of the two unless the content clearly calls for
repeating it.

Text sizing: keep display_text short enough to comfortably fit
geometry_px at a readable size. As a rule of thumb, assume roughly 14-18
characters fit per 100px of box width at a comfortable reading size — if
your intended phrase is longer than that, shorten the ON-SCREEN wording
(never alter the underlying narration) or widen geometry_px within the
safe margins already described above, rather than shrinking text to the
point of being unreadable.

Quoted-source handling: if a beat's source_type is "quoted_source", use
"full_screen_document_highlight" (or another full_screen quote treatment
only if document_highlight clearly doesn't fit). This treatment is backed
by a REAL screenshot of the actual source page (captured separately from
you, using source_name_guess) — not an AI-generated mockup. Because the
real page's layout is unknown to you in advance, do NOT invent pixel
coordinates for the highlight itself — instead:
- set highlight_target_text to the quoted_excerpt exactly as given, so a
  downstream text-locating step (OCR/text search over the captured
  screenshot) can find and highlight it precisely;
- use render_prompt only to describe the highlight STYLE and timing (e.g.
  "yellow marker-style highlight sweeps left to right under the sentence,
  starting 1s after the screenshot appears, holding for the rest of the
  beat") and to name the source if inferable from vo_text;
- geometry_px for this animation_type is always the full frame
  (x=0, y=0, width=1920, height=1080) since the
  screenshot itself fills the screen.

ALLOWED animation_type VALUES (grouped by category):

FULL_SCREEN (category: "full_screen")
- full_screen_broll, full_screen_title_card, full_screen_data_viz,
  full_screen_transition, full_screen_color_wash, full_screen_quote_card,
  full_screen_document_highlight

OVERLAY_TEXT (category: "overlay_text")
- lower_third, kinetic_caption, bullet_list_reveal, callout_textbox,
  stat_counter_overlay

OVERLAY_GRAPHIC (category: "overlay_graphic")
- icon_pop_in, icon_sequence, logo_watermark, emoji_reaction,
  arrow_highlight, badge_sticker

PICTURE_IN_PICTURE (category: "pip")
- pip_video, split_screen, multi_panel_grid

CHARACTER (category: "branding")
- avatar_overlay, mascot_animation

MOTION_EFFECT (category: "transition")
- ken_burns_pan_zoom, parallax_layering, shake_impact, speed_ramp_indicator

Placement values: top_left, top_center, top_right, center_left, center,
center_right, bottom_left, bottom_center, bottom_right, full_frame

IMPORTANT — caption safe zone: word-by-word captions are always burned in
along the very bottom edge of the frame (lowest ~15% of canvas). For any
overlay_text or overlay_graphic animation_type, do NOT choose bottom_left,
bottom_center, or bottom_right. Prefer top_left/top_center/top_right,
center_left/center/center_right instead. Bottom placements remain fine only
for full_screen or transition category animation_types, which own the
whole frame anyway.

Output Schema

Return exactly one JSON object, nothing else — no markdown, no code fences:

{{
  "animations": [
    {{
      "beat_id": "s1_beat1",
      "animation_type": "icon_pop_in",
      "category": "overlay_graphic",
      "placement": "top_right",
      "geometry_px": {{"x": 1696, "y": 64, "width": 160, "height": 160}},
      "motion": {{
        "start_xy_px": [1696, 44],
        "end_xy_px": [1696, 64],
        "motion_style": "pop-in with slight bounce, scale 80% to 100%"
      }},
      "z_index_layer": "foreground",
      "trigger": "on_keyword",
      "duration_frames": 45,
      "content_binding": "icon:lightbulb",
      "icon_name": "lightbulb",
      "icon_layout": null,
      "display_text": null,
      "color_hint": "#F5A623",
      "highlight_target_text": null,
      "render_prompt": "A lightbulb icon pops in from 80% scale to 100% with a slight bounce, positioned top-right, appearing exactly as the word 'idea' is spoken; icon uses the video's accent color; holds for 1.5s then fades.",
      "render_engine_hint": "remotion"
    }}
  ]
}}

Field Guidelines

beat_id: must match a beat_id from the input beats array. Only include
beats that actually receive an animation — omit beats with no animation.

geometry_px: {{x, y, width, height}} in real pixels on the
1920x1080 canvas — the element's full rendered
footprint. Derive x/y from the PLACEMENT_ANCHORS_PX entry for your chosen
placement, then choose width/height deliberately for this content: icons
typically 120-240px square depending on emphasis; text overlay boxes sized
to fit the actual text at a readable size (roughly 36-64px font for
overlay text, larger for full_screen title/quote cards); pip frames
roughly 1/3 to 1/2 of canvas width/height. For full_screen and transition
category types, geometry_px is always the full frame: {{"x": 0, "y": 0,
"width": 1920, "height": 1080}}.

motion: {{start_xy_px, end_xy_px, motion_style}}. For a static element
(appears in place, no travel), set start_xy_px equal to end_xy_px and
describe the entrance/exit behavior (fade, pop, scale) in motion_style.
For an element that visibly moves or travels across the frame, set
different start/end coordinates. motion_style is a short phrase, not a
full paragraph — the paragraph-level detail goes in render_prompt.

icon_name: a single string, or an array of 2-4 strings (see "Mixing icons"
above), from the icon vocabulary — required (non-null) whenever category
is "overlay_graphic" or "branding". null for every other category.

icon_layout: "sequence", "cluster", or "pair" — required (non-null) when
icon_name is an array; null when icon_name is a single string or null.

display_text: the literal on-screen text for this animation, grounded in
this beat's vo_text/on_screen_text/key_subject — never invented wording.
A single string for most types; an array of short strings for
bullet_list_reveal or icon_sequence when each item needs its own label.
null for animation types with no on-screen text (e.g. a plain
ken_burns_pan_zoom with no overlay copy).

color_hint: a hex color string (e.g. "#F5A623"), always required — the
accent color for this animation, chosen to fit the category's
footage_style/tone unless the specific beat content calls for something
else (e.g. a red accent for a warning statistic).

highlight_target_text: the exact quoted_excerpt from the source beat's
animation_signal when animation_type is "full_screen_document_highlight"
(or any other treatment highlighting specific text within a real
screenshot asset). null for every other animation_type.

content_binding: a short free-form label kept for logging/debugging (e.g.
"icon:lightbulb", "data:quarterly_revenue") — not the authoritative value
for icon or text content; icon_name/display_text/color_hint above are what
the renderer actually binds to.

render_prompt: a full descriptive creative-direction paragraph — human-
readable rationale and motion/reveal detail for anyone reviewing this
animation. It documents the "why" and the motion feel; it is not parsed by
the renderer, which reads geometry_px/motion/icon_name/display_text/
color_hint directly.

Constraints

Every field is required for each animation object. Do not omit any field
(use null where noted above). Choose placement and geometry deliberately
based on what will read best against this beat's B-roll and on_screen_text
— a real creative decision, not a formality. Return ONLY the JSON object.

"""


class EditVideo(BaseModel):
    userId: str
    script: str
    voice: str
    langCode: str = "en"
    durationMinutes: int = 0
    volume: Optional[float] = None
    loudness_normalization: Optional[bool] = None
    text_normalization: Optional[bool] = None


class TrackPatch(BaseModel):
    track_id: str
    updates: dict[str, Any]


_VALID_CAPTION_ANIMATION_TYPES = {
    "kinetic_caption", "static_line", "typewriter", "word_pop",
}


class SceneStyleUpdate(BaseModel):
    font_size: Optional[int] = None
    text_color: Optional[str] = None
    outline_color: Optional[str] = None
    animation_type: Optional[str] = None
    background_color: Optional[str] = None
    vertical_position: Optional[Literal["top", "middle", "bottom"]] = None
    margin_bottom_percent: Optional[float] = None
    horizontal_position: Optional[Literal["left", "center", "right"]] = None
    margin_horizontal_percent: Optional[float] = None


class BeatSplitUpdate(BaseModel):
    split_at: float


class BeatInsertUpdate(BaseModel):
    start: float
    end: float


class SceneTrimUpdate(BaseModel):
    start: float
    end: float


class BeatAnimationUpdate(BaseModel):
    """Create or edit the (at most one) animation attached to a beat."""
    animation_type: Optional[str] = None
    placement: Optional[str] = None
    geometry_px: Optional[dict[str, Any]] = None
    motion: Optional[dict[str, Any]] = None
    duration_frames: Optional[int] = None
    z_index_layer: Optional[str] = None
    trigger: Optional[str] = None
    content_binding: Optional[str] = None
    icon_name: Optional[Any] = None
    icon_layout: Optional[str] = None
    display_text: Optional[Any] = None
    color_hint: Optional[str] = None
    highlight_target_text: Optional[str] = None
    render_prompt: Optional[str] = None
    render_engine_hint: Optional[str] = None


class SceneBrollSelectUpdate(BaseModel):
    asset_id: Any
    source: str
    beat_id: Optional[str] = None
    motion_type: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    adjust_next_beat: bool = True


def _hex_to_ass_color(hex_color: str, alpha_hex: str = "00") -> str:
    h = (hex_color or "").strip().lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha_hex}{b}{g}{r}"


def _normalize_ffmpeg_color(hex_color: str) -> str:
    h = (hex_color or "").strip().lstrip("#")
    if len(h) != 6:
        h = "111827"
    return f"0x{h}"


_HEX_COLOR_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


def _validate_hex_color(value: Optional[str], field_name: str) -> None:
    if value is None:
        return
    if not _HEX_COLOR_RE.match(value.strip()):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be a 6-digit hex color like '#111827', got {value!r}",
        )


WHISPERX_MODEL_SIZE = os.getenv("WHISPERX_MODEL_SIZE", "small")
WHISPERX_DEVICE = os.getenv("WHISPERX_DEVICE", "cpu")
WHISPERX_COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "int8")

PEXELS_SCENE_RESULT_LIMIT = int(os.getenv("PEXELS_SCENE_RESULT_LIMIT", "6"))
PEXELS_SCENE_PER_KEYWORD_PER_PAGE = int(os.getenv("PEXELS_SCENE_PER_KEYWORD_PER_PAGE", "4"))
PEXELS_SCENE_PAGE = 1

PEXELS_SCENE_VIDEO_ORIENTATION = "landscape"
PEXELS_SCENE_IMAGE_ORIENTATION = os.getenv("PEXELS_SCENE_IMAGE_ORIENTATION", "landscape")
PEXELS_SCENE_SIZE = os.getenv("PEXELS_SCENE_SIZE", None)
PEXELS_SCENE_COLOR = os.getenv("PEXELS_SCENE_COLOR", None)

PEXELS_MAX_CONCURRENT_REQUESTS = int(os.getenv("PEXELS_MAX_CONCURRENT_REQUESTS", "8"))
_pexels_semaphore = asyncio.Semaphore(PEXELS_MAX_CONCURRENT_REQUESTS)

BROLL_KEYWORDS_MAX = int(os.getenv("BROLL_KEYWORDS_MAX", "6"))
BROLL_KEYWORDS_MIN = int(os.getenv("BROLL_KEYWORDS_MIN", "5"))

TIMELINE_FPS = int(os.getenv("TIMELINE_FPS", "30"))
TIMELINE_WIDTH = int(os.getenv("TIMELINE_WIDTH", "1080"))
TIMELINE_HEIGHT = int(os.getenv("TIMELINE_HEIGHT", "1920"))

DEFAULT_CAPTION_STYLE = {
    "vertical_position": "bottom",
    "margin_bottom_percent": 3,
}

TTS_MAX_CHARS_PER_CALL = int(os.getenv("TTS_MAX_CHARS_PER_CALL", "900"))
TTS_AUDIO_BUCKET = os.getenv("TTS_AUDIO_BUCKET", "generated-audio")

_whisperx_model = None
_whisperx_align_cache = {}
_whisperx_lock = asyncio.Lock()


def _get_whisperx_model():
    global _whisperx_model
    if _whisperx_model is None:
        print(f"[WHISPERX] loading model '{WHISPERX_MODEL_SIZE}' on {WHISPERX_DEVICE}")
        _whisperx_model = whisperx.load_model(
            WHISPERX_MODEL_SIZE, WHISPERX_DEVICE, compute_type=WHISPERX_COMPUTE_TYPE,
        )
    return _whisperx_model


def _run_whisperx_sync(audio_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        model = _get_whisperx_model()
        audio = whisperx.load_audio(tmp.name)

        result = model.transcribe(audio, batch_size=16)
        language = result["language"]

        if language not in _whisperx_align_cache:
            print(f"[WHISPERX] loading alignment model for language '{language}'")
            align_model, metadata = whisperx.load_align_model(language_code=language, device=WHISPERX_DEVICE)
            _whisperx_align_cache[language] = (align_model, metadata)

        align_model, metadata = _whisperx_align_cache[language]

        aligned_result = whisperx.align(
            result["segments"], align_model, metadata, audio, WHISPERX_DEVICE,
            return_char_alignments=False,
        )

        return {
            "language": language,
            "segments": aligned_result["segments"],
            "word_segments": aligned_result.get("word_segments", []),
        }


async def _generate_word_timestamps(audio_url: str) -> dict:
    audio_bytes = await _download_bytes(audio_url)
    async with _whisperx_lock:
        return await asyncio.to_thread(_run_whisperx_sync, audio_bytes)


def _split_text_for_tts(text: str, max_chars: int = TTS_MAX_CHARS_PER_CALL) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(sent) <= max_chars:
            current = sent
        else:
            for i in range(0, len(sent), max_chars):
                chunks.append(sent[i:i + max_chars])
    if current:
        chunks.append(current)
    return chunks


async def _upload_audio_to_storage(local_path: str, user_id: str) -> str:
    storage_path = f"{user_id}/{uuid.uuid4().hex}.mp3"
    with open(local_path, "rb") as f:
        supabase.storage.from_(TTS_AUDIO_BUCKET).upload(
            storage_path, f, {"content-type": "audio/mpeg", "upsert": "true"}
        )
    return supabase.storage.from_(TTS_AUDIO_BUCKET).get_public_url(storage_path)


async def _generate_speech_possibly_chunked(
    user_id: str, tagged_text: str, voice: str, lang_code: str,
    volume: Optional[float] = None,
    loudness_normalization: Optional[bool] = None,
    text_normalization: Optional[bool] = None,
) -> dict:
    tts_kwargs = {}
    if volume is not None:
        tts_kwargs["volume"] = volume
    if loudness_normalization is not None:
        tts_kwargs["loudnessNormalization"] = loudness_normalization
    if text_normalization is not None:
        tts_kwargs["textNormalization"] = text_normalization

    chunks = _split_text_for_tts(tagged_text)

    if len(chunks) <= 1:
        speech_request = GenerateSpeechRequest(
            userId=user_id, script=tagged_text, voice=voice, langCode=lang_code, durationMinutes=0, **tts_kwargs,
        )
        return await generate_speech(speech_request)

    print(f"[tts] narration is {len(tagged_text)} chars — splitting into {len(chunks)} TTS calls to avoid provider truncation")

    work_dir = tempfile.mkdtemp(prefix="tts_chunks_")
    try:
        chunk_paths = []
        for idx, chunk_text in enumerate(chunks):
            speech_request = GenerateSpeechRequest(
                userId=user_id, script=chunk_text, voice=voice, langCode=lang_code, durationMinutes=0, **tts_kwargs,
            )
            chunk_result = await generate_speech(speech_request)
            chunk_bytes = await _download_bytes(chunk_result["url"])
            chunk_path = os.path.join(work_dir, f"chunk_{idx:03d}.mp3")
            with open(chunk_path, "wb") as f:
                f.write(chunk_bytes)
            chunk_paths.append(chunk_path)

        list_path = os.path.join(work_dir, "concat_list.txt")
        with open(list_path, "w") as f:
            for p in chunk_paths:
                f.write(f"file '{p}'\n")

        combined_path = os.path.join(work_dir, "combined.mp3")
        cmd = [
            FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:a", "libmp3lame", "-b:a", "192k", combined_path,
        ]
        await _run(cmd)

        combined_url = await _upload_audio_to_storage(combined_path, user_id)
        return {"url": combined_url}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _get_scene_broll_keywords(scene: dict) -> list:
    raw_keywords = scene.get("broll_keywords")

    keywords = []
    if isinstance(raw_keywords, list):
        keywords = [k.strip() for k in raw_keywords if isinstance(k, str) and k.strip()]
        seen = set()
        deduped = []
        for k in keywords:
            key = k.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(k)
        keywords = deduped

    if len(keywords) >= BROLL_KEYWORDS_MIN:
        return keywords[:BROLL_KEYWORDS_MAX]

    fallback_extras = []
    visual_intent = (scene.get("visual_intent") or "").strip()
    if visual_intent and visual_intent.lower() not in [k.lower() for k in keywords]:
        fallback_extras.append(visual_intent)

    vo_snippet = (scene.get("vo_text") or "").strip()[:60]
    if vo_snippet and vo_snippet.lower() not in [k.lower() for k in keywords]:
        fallback_extras.append(vo_snippet)

    keywords = (keywords + fallback_extras)[:BROLL_KEYWORDS_MAX]

    if not keywords:
        print(f"[edit-video] scene {scene.get('scene_id')} has no usable broll_keywords or fallback text")

    return keywords


def _dedupe_and_trim(items: list, limit: Optional[int] = None) -> list:
    def _identity(item: dict):
        for key in ("id", "video_id", "photo_id", "asset_id"):
            if key in item:
                return item[key]
        for key in ("url", "src", "image", "video_url", "link"):
            if key in item:
                return item[key]
        return None

    seen = set()
    deduped = []
    for item in items:
        item_id = _identity(item)
        if item_id is not None:
            if item_id in seen:
                continue
            seen.add(item_id)
        deduped.append(item)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
CLIP_DEVICE = os.getenv("CLIP_DEVICE", "cpu")

CLIP_RERANK_ENABLED = os.getenv("CLIP_RERANK_ENABLED", "true").lower() == "true"

CLIP_MIN_VIDEO_SCORE = float(os.getenv("CLIP_MIN_VIDEO_SCORE", "0.24"))
CLIP_MIN_IMAGE_SCORE = float(os.getenv("CLIP_MIN_IMAGE_SCORE", "0.22"))

_clip_model = None
_clip_processor = None


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print(f"[CLIP] loading '{CLIP_MODEL_NAME}' on {CLIP_DEVICE}")
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(CLIP_DEVICE).eval()
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return _clip_model, _clip_processor


def _clip_score_images_sync(query_texts: list[str], images: list) -> dict:
    model, processor = _get_clip()
    pil_images = [img for _cid, img in images]
    inputs = processor(text=query_texts, images=pil_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(CLIP_DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        image_embeds = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_embeds = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        sims = image_embeds @ text_embeds.T
        best_per_image = sims.max(dim=-1).values.tolist()
        if not isinstance(best_per_image, list):
            best_per_image = [best_per_image]
    return {images[i][0]: best_per_image[i] for i in range(len(images))}


async def _rerank_images_with_clip(query_texts: list[str], image_candidates: list) -> list:
    if not CLIP_RERANK_ENABLED or not image_candidates or not query_texts:
        return image_candidates

    downloaded = []
    try:
        async with httpx.AsyncClient() as client:
            for cand in image_candidates:
                url = _resolve_broll_file_url(cand, "image")
                if not url:
                    continue
                try:
                    resp = await client.get(url, timeout=20.0)
                    resp.raise_for_status()
                    img = Image.open(_io.BytesIO(resp.content)).convert("RGB")
                    downloaded.append((str(cand.get("id")), img, cand))
                except Exception as e:
                    print(f"[CLIP] failed to download image {cand.get('id')}: {e}")
    except Exception as e:
        print(f"[CLIP] image download pass failed entirely, keeping original order: {e}")
        return image_candidates

    if not downloaded:
        return image_candidates

    try:
        scores = await asyncio.to_thread(_clip_score_images_sync, query_texts, [(cid, img) for cid, img, _c in downloaded])
    except Exception as e:
        print(f"[CLIP] scoring failed, keeping original order: {e}")
        return image_candidates

    ranked = sorted(downloaded, key=lambda t: scores.get(t[0], -1.0), reverse=True)
    ranked_ids = {cid for cid, _img, _c in ranked}
    result = []
    for cid, _img, c in ranked:
        c["_clip_score"] = scores.get(cid)
        result.append(c)
    result += [c for c in image_candidates if str(c.get("id")) not in ranked_ids]
    return result


async def _rerank_videos_with_clip(query_texts: list[str], video_candidates: list) -> list:
    if not CLIP_RERANK_ENABLED or not video_candidates or not query_texts:
        return video_candidates

    downloaded = []
    try:
        async with httpx.AsyncClient() as client:
            for cand in video_candidates:
                url = cand.get("thumbnail")
                if not url:
                    continue
                try:
                    resp = await client.get(url, timeout=20.0)
                    resp.raise_for_status()
                    img = Image.open(_io.BytesIO(resp.content)).convert("RGB")
                    downloaded.append((str(cand.get("id")), img, cand))
                except Exception as e:
                    print(f"[CLIP] failed to download video thumbnail {cand.get('id')}: {e}")
    except Exception as e:
        print(f"[CLIP] video thumbnail download pass failed entirely, keeping original order: {e}")
        return video_candidates

    if not downloaded:
        return video_candidates

    try:
        scores = await asyncio.to_thread(_clip_score_images_sync, query_texts, [(cid, img) for cid, img, _c in downloaded])
    except Exception as e:
        print(f"[CLIP] video thumbnail scoring failed, keeping original order: {e}")
        return video_candidates

    ranked = sorted(downloaded, key=lambda t: scores.get(t[0], -1.0), reverse=True)
    ranked_ids = {cid for cid, _img, _c in ranked}
    result = []
    for cid, _img, c in ranked:
        c["_clip_score"] = scores.get(cid)
        result.append(c)
    result += [c for c in video_candidates if str(c.get("id")) not in ranked_ids]
    return result


def _filter_by_min_clip_score(candidates: list, min_score: float) -> list:
    if not candidates:
        return candidates
    if not any(c.get("_clip_score") is not None for c in candidates):
        return candidates
    return [c for c in candidates if c.get("_clip_score") is None or c["_clip_score"] >= min_score]


_http_session = requests.Session()

def _pexels_search_images_sync(
    query: str,
    per_page: int,
    page: int,
    orientation: Optional[str],
    size: Optional[str],
    color: Optional[str],
) -> dict:
    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY not set")

    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "per_page": per_page,
        "page": page,
    }
    if orientation:
        params["orientation"] = orientation
    if size:
        params["size"] = size
    if color:
        params["color"] = color

    resp = _http_session.get(
        PEXELS_IMAGE_SEARCH_URL, headers=headers, params=params, timeout=15
    )
    resp.raise_for_status()
    return resp.json()



def _pexels_search_videos_sync(
    query: str,
    per_page: int,
    page: int,
    orientation: Optional[str],
    size: Optional[str],
) -> dict:
    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY not set")

    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "per_page": per_page,
        "page": page,
    }
    if orientation:
        params["orientation"] = orientation
    if size:
        params["size"] = size

    resp = _http_session.get(
        PEXELS_VIDEO_SEARCH_URL, headers=headers, params=params, timeout=15
    )
    resp.raise_for_status()
    return resp.json()



async def _fetch_media_for_keywords(keywords: list, label: str) -> dict:
    empty_result = {
        "videos": {"total_results": 0, "results": [], "error": None},
        "images": {"total_results": 0, "results": [], "error": None},
    }

    if not keywords:
        return empty_result

    if not PEXELS_API_KEY:
        empty_result["videos"]["error"] = "PEXELS_API_KEY is not configured on the server."
        empty_result["images"]["error"] = "PEXELS_API_KEY is not configured on the server."
        return empty_result

    async def _get_videos_for(keyword: str):
        try:
            async with _pexels_semaphore:
                data = await asyncio.to_thread(
                    _pexels_search_videos_sync, keyword, PEXELS_SCENE_PER_KEYWORD_PER_PAGE,
                    PEXELS_SCENE_PAGE, PEXELS_SCENE_VIDEO_ORIENTATION, PEXELS_SCENE_SIZE,
                )
            return {"keyword": keyword, "data": data, "error": None}
        except Exception as e:
            return {"keyword": keyword, "data": None, "error": str(e)}

    async def _get_images_for(keyword: str):
        try:
            async with _pexels_semaphore:
                data = await asyncio.to_thread(
                    _pexels_search_images_sync, keyword, PEXELS_SCENE_PER_KEYWORD_PER_PAGE,
                    PEXELS_SCENE_PAGE, PEXELS_SCENE_IMAGE_ORIENTATION, PEXELS_SCENE_SIZE, PEXELS_SCENE_COLOR,
                )
            return {"keyword": keyword, "data": data, "error": None}
        except Exception as e:
            return {"keyword": keyword, "data": None, "error": str(e)}

    video_tasks = [_get_videos_for(k) for k in keywords]
    image_tasks = [_get_images_for(k) for k in keywords]
    video_results, image_results = await asyncio.gather(
        asyncio.gather(*video_tasks), asyncio.gather(*image_tasks),
    )

    video_errors = [r["error"] for r in video_results if r["error"]]
    videos_pool = []
    for r in video_results:
        if r["error"]:
            continue
        videos_pool.extend(_format_video_result(v) for v in (r["data"].get("videos") or []))

    pre_filter_count = len(videos_pool)
    videos_pool = [v for v in videos_pool if _video_is_landscape(v)]
    dropped = pre_filter_count - len(videos_pool)
    if dropped:
        print(f"[edit-video] {label}: dropped {dropped} non-landscape video result(s) at fetch time")

    image_errors = [r["error"] for r in image_results if r["error"]]
    images_pool = []
    for r in image_results:
        if r["error"]:
            continue
        images_pool.extend(_format_image_result(p) for p in (r["data"].get("photos") or []))

    pre_filter_img_count = len(images_pool)
    images_pool = [p for p in images_pool if _image_is_landscape(p)]
    dropped_img = pre_filter_img_count - len(images_pool)
    if dropped_img:
        print(f"[edit-video] {label}: dropped {dropped_img} non-landscape image result(s) at fetch time")

    videos_full = _dedupe_and_trim(videos_pool)
    photos_full = _dedupe_and_trim(images_pool)

    if photos_full:
        photos_full = await _rerank_images_with_clip(keywords, photos_full)
        pre_relevance_photo_count = len(photos_full)
        photos_full = _filter_by_min_clip_score(photos_full, CLIP_MIN_IMAGE_SCORE)
        dropped_irrelevant_photos = pre_relevance_photo_count - len(photos_full)
        if dropped_irrelevant_photos:
            print(f"[CLIP] {label}: dropped {dropped_irrelevant_photos} image candidate(s) below the relevance floor ({CLIP_MIN_IMAGE_SCORE})")
    if videos_full:
        videos_full = await _rerank_videos_with_clip(keywords, videos_full)
        pre_relevance_video_count = len(videos_full)
        videos_full = _filter_by_min_clip_score(videos_full, CLIP_MIN_VIDEO_SCORE)
        dropped_irrelevant_videos = pre_relevance_video_count - len(videos_full)
        if dropped_irrelevant_videos:
            print(f"[CLIP] {label}: dropped {dropped_irrelevant_videos} video candidate(s) below the relevance floor ({CLIP_MIN_VIDEO_SCORE})")

    videos = videos_full[:PEXELS_SCENE_RESULT_LIMIT]
    photos = photos_full[:PEXELS_SCENE_RESULT_LIMIT]

    videos_error = "; ".join(video_errors) if video_errors and not videos else None
    images_error = "; ".join(image_errors) if image_errors and not photos else None

    best_video_score = videos[0].get("_clip_score") if videos else None
    best_image_score = photos[0].get("_clip_score") if photos else None
    force_image_fallback = (
        best_video_score is not None and best_video_score < CLIP_MIN_VIDEO_SCORE
        and best_image_score is not None and best_image_score >= CLIP_MIN_IMAGE_SCORE
    )
    if force_image_fallback:
        print(
            f"[CLIP] {label}: best video scored {best_video_score:.3f} "
            f"(below {CLIP_MIN_VIDEO_SCORE}) — falling back to image "
            f"(scored {best_image_score:.3f}) instead of an inaccurate video"
        )

    return {
        "videos": {"total_results": len(videos), "results": videos, "error": videos_error},
        "images": {"total_results": len(photos), "results": photos, "error": images_error},
        "force_image_fallback": force_image_fallback,
        "best_video_score": best_video_score,
        "best_image_score": best_image_score,
    }


def _beat_media_is_empty(beat: dict) -> bool:
    media = beat.get("media") or {}
    return not (media.get("videos") or {}).get("results") and not (media.get("images") or {}).get("results")


_LAST_RESORT_BROLL_TERMS = [
    "abstract background", "soft light texture", "historical documents",
    "old map parchment", "clouds sky timelapse", "city skyline aerial",
]


async def _fill_empty_beats(scene: dict, beats: list, fallback_keywords: list) -> None:
    scene_id = scene.get("scene_id")

    empty_beats = [b for b in beats if _beat_media_is_empty(b)]
    if not empty_beats:
        return

    retry_results = await asyncio.gather(*[
        _fetch_media_for_keywords(fallback_keywords, f"{scene_id}:{b['beat_id']}:retry-scene-keywords")
        for b in empty_beats
    ])
    for b, media in zip(empty_beats, retry_results):
        if (media.get("videos") or {}).get("results") or (media.get("images") or {}).get("results"):
            b["media"] = media
            print(f"[edit-video] beat {b['beat_id']} was empty, filled via scene-level keyword retry")

    still_empty = [b for b in beats if _beat_media_is_empty(b)]
    if not still_empty:
        return

    non_empty = [b for b in beats if not _beat_media_is_empty(b)]
    if non_empty:
        for b in still_empty:
            idx = b.get("beat_index", 0)
            nearest = min(non_empty, key=lambda o: abs(o.get("beat_index", 0) - idx))
            b["media"] = {
                "videos": dict(nearest["media"].get("videos") or {}),
                "images": dict(nearest["media"].get("images") or {}),
            }
            b["_media_fallback_reason"] = f"borrowed from {nearest['beat_id']}"
            print(f"[edit-video] beat {b['beat_id']} was empty, borrowed media from beat {nearest['beat_id']}")
        return

    print(f"[edit-video][WARN] scene {scene_id}: EVERY beat came back empty — trying generic last-resort terms")
    last_resort_media = await _fetch_media_for_keywords(_LAST_RESORT_BROLL_TERMS, f"{scene_id}:last-resort")
    if (last_resort_media.get("videos") or {}).get("results") or (last_resort_media.get("images") or {}).get("results"):
        for b in still_empty:
            b["media"] = last_resort_media
            b["_media_fallback_reason"] = "generic last-resort terms"
    else:
        print(
            f"[edit-video][ERROR] scene {scene_id}: last-resort generic search ALSO returned "
            f"nothing — check PEXELS_API_KEY / Pexels reachability, this is no longer a content issue"
        )


def _aggregate_beats_media(beats: list) -> dict:
    all_videos, all_images = [], []
    for b in beats:
        m = b.get("media") or {}
        all_videos.extend((m.get("videos") or {}).get("results") or [])
        all_images.extend((m.get("images") or {}).get("results") or [])
    limit = PEXELS_SCENE_RESULT_LIMIT * max(len(beats), 1)
    videos = _dedupe_and_trim(all_videos, limit=limit)
    images = _dedupe_and_trim(all_images, limit=limit)
    return {
        "videos": {"total_results": len(videos), "results": videos, "error": None},
        "images": {"total_results": len(images), "results": images, "error": None},
    }


CLIP_MEDIA_SWITCH_MARGIN = float(os.getenv("CLIP_MEDIA_SWITCH_MARGIN", "0.03"))


def _resolve_beat_broll_selection(beat: dict) -> tuple:
    override = beat.get("broll_override")
    if override and override.get("asset_id") is not None:
        source = override.get("source")
        file_url = _resolve_broll_file_url(override, source)
        if file_url:
            return (
                {
                    "id": override.get("asset_id"), "file_url": file_url, "source": source,
                    **{k: v for k, v in override.items() if k not in ("asset_id", "source", "file_url")},
                },
                source,
            )
        print(f"[timeline] beat {beat.get('beat_id')} has a broll_override that couldn't be resolved to a landscape file — falling back to default candidate")

    media = beat.get("media") or {}
    video_candidates = (media.get("videos") or {}).get("results") or []
    image_candidates = (media.get("images") or {}).get("results") or []

    best_video = video_candidates[0] if video_candidates else None
    best_image = image_candidates[0] if image_candidates else None
    video_score = best_video.get("_clip_score") if best_video else None
    image_score = best_image.get("_clip_score") if best_image else None

    if video_score is not None and image_score is not None:
        video_is_weak = video_score < CLIP_MIN_VIDEO_SCORE
        image_is_usable = image_score >= CLIP_MIN_IMAGE_SCORE
        if video_is_weak and image_is_usable:
            return best_image, "image"

        if image_score > video_score + CLIP_MEDIA_SWITCH_MARGIN:
            return best_image, "image"
        if video_score > image_score + CLIP_MEDIA_SWITCH_MARGIN:
            return best_video, "video"

    preferred = beat.get("preferred_media_type")
    if preferred == "image":
        if best_image:
            return best_image, "image"
        if best_video:
            return best_video, "video"
        return None, None

    if best_video:
        return best_video, "video"
    if best_image:
        return best_image, "image"
    return None, None

_VALID_MEDIA_TYPES = {"video", "image"}

_MOTION_TYPE_LIST = ["zoom_in", "zoom_out", "pan_left", "pan_right", "tilt_up", "tilt_down"]
_VALID_MOTION_TYPES = set(_MOTION_TYPE_LIST)
_DEFAULT_MOTION_TYPE = "zoom_in"


def _pick_diversified_motion_type(previous_motion_type: Optional[str], beat_index: int) -> str:
    """B-roll Ken-Burns pan/zoom motion (distinct from the overlay `motion`
    object the Animation Planner returns). The new BEAT_KEYWORDS_PROMPT no
    longer asks the model for this, so it's assigned deterministically."""
    candidates = [m for m in _MOTION_TYPE_LIST if m != previous_motion_type] or list(_MOTION_TYPE_LIST)
    return candidates[beat_index % len(candidates)]


def _resolve_beat_motion_type(beat: dict) -> str:
    override = beat.get("broll_override") or {}
    motion = override.get("motion_type")
    if motion in _VALID_MOTION_TYPES:
        return motion
    motion = beat.get("motion_type")
    if motion in _VALID_MOTION_TYPES:
        return motion
    return _DEFAULT_MOTION_TYPE


def _find_beat_broll_candidate(beat: dict, asset_id: Any, source: str) -> Optional[dict]:
    media = beat.get("media") or {}
    pool_key = "videos" if source == "video" else "images"
    candidates = (media.get(pool_key) or {}).get("results") or []
    for c in candidates:
        if str(c.get("id")) == str(asset_id):
            return c
    return None


PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"
PEXELS_IMAGE_SEARCH_URL = "https://api.pexels.com/v1/search"


def _extract_video_files(video: dict) -> list[dict]:
    """Pick out the useful video_files entries (quality/resolution/link)."""
    files = []
    for vf in video.get("video_files", []):
        files.append({
            "quality": vf.get("quality"),
            "width": vf.get("width"),
            "height": vf.get("height"),
            "file_type": vf.get("file_type"),
            "link": vf.get("link"),
        })
    return files


def _format_video_result(video: dict) -> dict:
    return {
        "type": "video",
        "id": video.get("id"),
        "url": video.get("url"),
        "width": video.get("width"),
        "height": video.get("height"),
        "duration": video.get("duration"),
        "thumbnail": video.get("image"),
        "user": {
            "name": (video.get("user") or {}).get("name"),
            "url": (video.get("user") or {}).get("url"),
        },
        "video_files": _extract_video_files(video),
    }


def _format_image_result(photo: dict) -> dict:
    return {
        "type": "image",
        "id": photo.get("id"),
        "url": photo.get("url"),
        "width": photo.get("width"),
        "height": photo.get("height"),
        "photographer": {
            "name": photo.get("photographer"),
            "url": photo.get("photographer_url"),
        },
        "avg_color": photo.get("avg_color"),
        "alt": photo.get("alt"),
        "src": photo.get("src", {}),  # original, large2x, large, medium, small, portrait, landscape, tiny
    }



async def _fetch_pexels_asset_by_id(asset_id: Any, source: str) -> Optional[dict]:
    if not PEXELS_API_KEY:
        return None

    url = (
        f"https://api.pexels.com/videos/videos/{asset_id}"
        if source == "video"
        else f"https://api.pexels.com/v1/photos/{asset_id}"
    )
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers={"Authorization": PEXELS_API_KEY}, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        print(f"[broll] direct Pexels lookup failed for {source} asset {asset_id}: {e}")
        return None

    return _format_video_result(data) if source == "video" else _format_image_result(data)


def _seconds_to_frames(seconds: float, fps: int = TIMELINE_FPS) -> int:
    return max(round((seconds or 0.0) * fps), 0)


def _slim_selected_asset(selected: Optional[dict]) -> Optional[dict]:
    if not selected:
        return None
    return {
        "asset_id": selected.get("asset_id") if "asset_id" in selected else selected.get("id"),
        "file_url": selected.get("file_url"),
        "source": selected.get("source"),
        "width": selected.get("width"),
        "height": selected.get("height"),
    }


def _words_in_range(timed_words: list, start: float, end: float) -> str:
    words = [
        (w.get("word") or "").strip()
        for w in timed_words
        if w.get("start", -1) >= start - 1e-6 and w.get("start", -1) < end + 1e-6
    ]
    return " ".join(w for w in words if w)



def _validate_entity(e: Any) -> Optional[dict]:
    if not isinstance(e, dict):
        return None
    name = e.get("name")
    etype = e.get("entity_type")
    if not isinstance(name, str) or not name.strip():
        return None
    if etype not in ("real_person", "fictional_character", "element"):
        etype = "element"
    return {"name": name.strip(), "entity_type": etype}


def _validate_scene_direction(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    setting = raw.get("setting") if isinstance(raw.get("setting"), dict) else {}
    location = setting.get("location") if isinstance(setting.get("location"), str) else ""
    time_period = setting.get("time_period") if isinstance(setting.get("time_period"), str) else ""
    mood = raw.get("mood") if isinstance(raw.get("mood"), str) and raw.get("mood").strip() else "neutral"
    key_action = (
        raw.get("key_action")
        if isinstance(raw.get("key_action"), str) and raw.get("key_action").strip()
        else "narration continues"
    )
    return {"setting": {"location": location, "time_period": time_period}, "mood": mood, "key_action": key_action}


def _validate_animation_signal(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    needs_animation = bool(raw.get("needs_animation"))
    intent = raw.get("intent") if isinstance(raw.get("intent"), str) else ""
    suggested_category = raw.get("suggested_category")
    if suggested_category not in ANIMATION_TAXONOMY:
        suggested_category = None
    key_subject = raw.get("key_subject") if isinstance(raw.get("key_subject"), str) else ""
    source_type = raw.get("source_type") if raw.get("source_type") in ("quoted_source", "narrative") else "narrative"
    quoted_excerpt = raw.get("quoted_excerpt") if isinstance(raw.get("quoted_excerpt"), str) else None
    source_name_guess = raw.get("source_name_guess") if isinstance(raw.get("source_name_guess"), str) else None
    key_subject_entity_type = raw.get("key_subject_entity_type")
    if key_subject_entity_type not in ("real_person", "fictional_character", "element"):
        key_subject_entity_type = None
    return {
        "needs_animation": needs_animation,
        "intent": intent,
        "suggested_category": suggested_category,
        "key_subject": key_subject,
        "source_type": source_type,
        "quoted_excerpt": quoted_excerpt,
        "source_name_guess": source_name_guess,
        "key_subject_entity_type": key_subject_entity_type,
    }


def _validate_beat(raw: Any, id_prefix: str, index: int) -> dict:
    if not isinstance(raw, dict):
        raw = {}

    vo_text = raw.get("vo_text") if isinstance(raw.get("vo_text"), str) else ""

    keywords_raw = raw.get("keywords")
    keywords = []
    if isinstance(keywords_raw, list):
        keywords = [k.strip() for k in keywords_raw if isinstance(k, str) and k.strip()]
        seen, deduped = set(), []
        for k in keywords:
            key = k.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(k)
        keywords = deduped

    media_type = str(raw.get("media_type") or "").strip().lower()
    if media_type not in _VALID_MEDIA_TYPES:
        media_type = "video"

    entities = [v for v in (_validate_entity(e) for e in (raw.get("entities") or [])) if v]
    scene_direction = _validate_scene_direction(raw.get("scene_direction"))
    animation_signal = _validate_animation_signal(raw.get("animation_signal"))

    try:
        est = int(raw.get("estimated_duration_seconds"))
        if est <= 0:
            raise ValueError
    except (TypeError, ValueError):
        est = max(1, round(len(vo_text.split()) / 2.33))

    return {
        "beat_id": f"{id_prefix}_beat{index + 1}",
        "beat_index": index,
        "vo_text": vo_text,
        "estimated_duration_seconds": est,
        "keywords": keywords,
        "media_type": media_type,
        "preferred_media_type": media_type,
        "entities": entities,
        "scene_direction": scene_direction,
        "animation_signal": animation_signal,
    }


async def _run_beat_director(
    *, scene_vo_text: str, category: str, style_profile: dict, script_language: str,
    scene_id: str, scene_visual_intent: str, scene_animation_density: str, scene_on_screen_text: str,
    previous_scene_last_media_type: Optional[str], known_entities: list, known_setting: dict,
    id_prefix: str,
) -> list[dict]:
    context = {
        "category": category,
        "style_profile": style_profile,
        "script_language": script_language,
        "scene_id": scene_id,
        "scene_visual_intent": scene_visual_intent,
        "scene_animation_density": scene_animation_density,
        "scene_on_screen_text": scene_on_screen_text,
        "scene_vo_text": scene_vo_text,
        "previous_scene_last_media_type": previous_scene_last_media_type,
        "known_entities": known_entities,
        "known_setting": known_setting,
    }

    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": BEAT_KEYWORDS_PROMPT},
                    {"role": "user", "content": json.dumps(context)},
                ],
                stream=False,
            )
        )
        _record_token_usage("edit video - beat director", res)

        content = (res.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

        raw = json.loads(content)
        raw_beats = raw.get("beats")
        if not isinstance(raw_beats, list) or not raw_beats:
            raise ValueError("beat director returned no beats")

    except Exception as e:
        print(f"[edit-video] scene {scene_id} beat director failed, using a single fallback beat: {e}")
        fallback_keywords = _get_scene_broll_keywords(
            {"broll_keywords": [], "visual_intent": scene_visual_intent, "vo_text": scene_vo_text}
        )
        return [{
            "beat_id": f"{id_prefix}_beat1",
            "beat_index": 0,
            "vo_text": scene_vo_text,
            "estimated_duration_seconds": max(1, round(len(scene_vo_text.split()) / 2.33)),
            "keywords": fallback_keywords,
            "media_type": "video",
            "preferred_media_type": "video",
            "entities": [],
            "scene_direction": {
                "setting": dict(known_setting or {"location": "", "time_period": ""}),
                "mood": "neutral",
                "key_action": "narration continues",
            },
            "animation_signal": _validate_animation_signal(None),
        }]

    beats = [_validate_beat(b, id_prefix, i) for i, b in enumerate(raw_beats)]

    covered_words = sum(len(b["vo_text"].split()) for b in beats)
    original_words = len(scene_vo_text.split())
    if covered_words != original_words:
        print(
            f"[edit-video][WARN] scene {scene_id}: beat director word count "
            f"mismatch ({covered_words} vs {original_words} original) — "
            f"narration may have been dropped, duplicated, or paraphrased across beats"
        )

    for b in beats:
        if len(b["keywords"]) < BROLL_KEYWORDS_MIN:
            fallback = _get_scene_broll_keywords(
                {"broll_keywords": [], "visual_intent": scene_visual_intent, "vo_text": b["vo_text"]}
            )
            merged, seen = [], set()
            for k in b["keywords"] + fallback:
                key = k.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(k)
            b["keywords"] = merged[:BROLL_KEYWORDS_MAX]

    previous_motion = None
    for b in beats:
        b["motion_type"] = _pick_diversified_motion_type(previous_motion, b["beat_index"])
        previous_motion = b["motion_type"]

    return beats


def _align_beats_to_timed_words(beats: list, timed_words: list) -> None:
    """Beats partition the scene's vo_text verbatim and in order, so we can
    map each beat's word count onto a contiguous slice of WhisperX's
    word-level timestamps."""
    ptr = 0
    n_words = len(timed_words)
    for i, b in enumerate(beats):
        count = len(b["vo_text"].split())
        if count <= 0:
            b["start"], b["end"] = None, None
            continue
        end_ptr = n_words if i == len(beats) - 1 else min(ptr + count, n_words)
        chunk = timed_words[ptr:end_ptr]
        if chunk:
            b["start"] = chunk[0].get("start")
            b["end"] = chunk[-1].get("end")
        else:
            b["start"], b["end"] = None, None
        ptr = end_ptr

    for i, b in enumerate(beats):
        if b.get("start") is not None and b.get("end") is not None:
            continue
        prev_end = (
            beats[i - 1]["end"] if i > 0 and beats[i - 1].get("end") is not None
            else (timed_words[0]["start"] if timed_words else 0.0)
        )
        next_start = (
            beats[i + 1]["start"] if i + 1 < len(beats) and beats[i + 1].get("start") is not None
            else (timed_words[-1]["end"] if timed_words else prev_end + 1.0)
        )
        b["start"] = prev_end
        b["end"] = max(next_start, prev_end + 0.5)


async def _fetch_beats_media(beats: list, label_prefix: str) -> None:
    results = await asyncio.gather(*[
        _fetch_media_for_keywords(b["keywords"], f"{label_prefix}:{b['beat_id']}") for b in beats
    ])
    for b, media in zip(beats, results):
        b["media"] = media
        if media.get("force_image_fallback") and b.get("preferred_media_type") != "image":
            print(f"[edit-video] beat {b['beat_id']}: overriding to 'image' (CLIP accuracy gate: video too weak a match)")
            b["preferred_media_type"] = "image"


def _dedupe_beats_media_across_scene(beats: list) -> None:
    used_ids: set = set()
    for beat in beats:
        media = beat.get("media") or {}
        for pool_key in ("videos", "images"):
            pool = media.get(pool_key) or {}
            results = pool.get("results") or []
            if not results:
                continue
            fresh = [r for r in results if r.get("id") not in used_ids]
            reused = [r for r in results if r.get("id") in used_ids]
            pool["results"] = fresh + reused
        beat["media"] = media

        default_asset, _source = _resolve_beat_broll_selection(beat)
        if default_asset and default_asset.get("id") is not None:
            used_ids.add(default_asset["id"])



def _validate_geometry_px(raw: Any, category: str) -> dict:
    if category in ("full_screen", "transition"):
        return {"x": 0, "y": 0, "width": ANIMATION_CANVAS_WIDTH, "height": ANIMATION_CANVAS_HEIGHT}

    default = {"x": 700, "y": 64, "width": 520, "height": 160}
    if not isinstance(raw, dict):
        geo = dict(default)
    else:
        try:
            geo = {
                "x": int(raw.get("x", default["x"])),
                "y": int(raw.get("y", default["y"])),
                "width": int(raw.get("width", default["width"])),
                "height": int(raw.get("height", default["height"])),
            }
        except (TypeError, ValueError):
            geo = dict(default)

    geo["width"] = max(40, min(geo["width"], ANIMATION_CANVAS_WIDTH))
    geo["height"] = max(40, min(geo["height"], ANIMATION_CANVAS_HEIGHT))
    geo["x"] = max(0, min(geo["x"], ANIMATION_CANVAS_WIDTH - geo["width"]))
    geo["y"] = max(0, min(geo["y"], ANIMATION_CANVAS_HEIGHT - geo["height"]))

    if category in ("overlay_text", "overlay_graphic"):
        if geo["y"] + geo["height"] > CAPTION_SAFE_ZONE_Y:
            if geo["height"] < CAPTION_SAFE_ZONE_Y:
                geo["y"] = CAPTION_SAFE_ZONE_Y - geo["height"]
            else:
                geo["height"] = CAPTION_SAFE_ZONE_Y - 4
                geo["y"] = 4

    return geo


def _validate_motion(raw: Any, geometry: dict) -> dict:
    default_xy = [geometry["x"], geometry["y"]]
    motion = {"start_xy_px": default_xy, "end_xy_px": default_xy, "motion_style": "fade in"}
    if isinstance(raw, dict):
        for key in ("start_xy_px", "end_xy_px"):
            val = raw.get(key)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                try:
                    motion[key] = [float(val[0]), float(val[1])]
                except (TypeError, ValueError):
                    pass
        style = raw.get("motion_style")
        if isinstance(style, str) and style.strip():
            motion["motion_style"] = style.strip()[:300]
    return motion


def _validate_beat_animation(raw: Any, beat_ids: set) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    beat_id = raw.get("beat_id")
    if beat_id not in beat_ids:
        return None
    animation_type = raw.get("animation_type")
    if animation_type not in _VALID_ANIMATION_TYPES:
        return None
    category = _ANIMATION_TYPE_TO_CATEGORY[animation_type]

    placement = raw.get("placement")
    if placement not in _VALID_PLACEMENTS:
        placement = "full_frame" if category in ("full_screen", "transition") else "top_center"
    if category in ("overlay_text", "overlay_graphic") and placement in ("bottom_left", "bottom_center", "bottom_right"):
        placement = "top_center"

    geometry_px = _validate_geometry_px(raw.get("geometry_px"), category)
    motion = _validate_motion(raw.get("motion"), geometry_px)

    z_index_layer = raw.get("z_index_layer")
    if z_index_layer not in _VALID_Z_LAYERS:
        z_index_layer = "background" if category == "full_screen" else "foreground"

    trigger = raw.get("trigger")
    if trigger not in _VALID_TRIGGERS:
        trigger = "on_beat"

    try:
        duration_frames = int(raw.get("duration_frames"))
        if duration_frames <= 0 or duration_frames > 900:
            raise ValueError
    except (TypeError, ValueError):
        duration_frames = 90

    content_binding = raw.get("content_binding")
    if not isinstance(content_binding, str):
        content_binding = ""

    icon_name = raw.get("icon_name")
    icon_layout = raw.get("icon_layout")
    if category in ("overlay_graphic", "branding"):
        if isinstance(icon_name, str) and icon_name in _ICON_VOCAB:
            icon_layout = None
        elif isinstance(icon_name, list):
            cleaned = [i for i in icon_name if isinstance(i, str) and i in _ICON_VOCAB][:4]
            if len(cleaned) >= 2:
                icon_name = cleaned
                if icon_layout not in _VALID_ICON_LAYOUTS:
                    icon_layout = "sequence" if animation_type == "icon_sequence" else "cluster"
            else:
                icon_name, icon_layout = "sparkles", None
        else:
            icon_name, icon_layout = "sparkles", None
    else:
        icon_name, icon_layout = None, None

    display_text = raw.get("display_text")
    if display_text is not None and not isinstance(display_text, (str, list)):
        display_text = None
    if isinstance(display_text, list):
        display_text = [str(d) for d in display_text if isinstance(d, (str, int, float))][:8]

    color_hint = raw.get("color_hint")
    if not isinstance(color_hint, str) or not _HEX_COLOR_RE.match(color_hint.strip()):
        color_hint = "#F5A623"
    else:
        color_hint = color_hint.strip()
        if not color_hint.startswith("#"):
            color_hint = f"#{color_hint}"

    highlight_target_text = raw.get("highlight_target_text")
    if not isinstance(highlight_target_text, str) or not highlight_target_text.strip():
        highlight_target_text = None

    render_prompt = raw.get("render_prompt")
    if not isinstance(render_prompt, str):
        render_prompt = ""

    render_engine_hint = raw.get("render_engine_hint")
    if render_engine_hint not in _VALID_RENDER_HINTS:
        render_engine_hint = "remotion" if category in ("overlay_text", "overlay_graphic", "pip", "branding") else "ffmpeg"

    return {
        "beat_id": beat_id,
        "animation_type": animation_type,
        "category": category,
        "placement": placement,
        "geometry_px": geometry_px,
        "motion": motion,
        "z_index_layer": z_index_layer,
        "trigger": trigger,
        "duration_frames": duration_frames,
        "content_binding": content_binding,
        "icon_name": icon_name,
        "icon_layout": icon_layout,
        "display_text": display_text,
        "color_hint": color_hint,
        "highlight_target_text": highlight_target_text,
        "render_prompt": render_prompt,
        "render_engine_hint": render_engine_hint,
    }


async def _run_animation_planner(
    *, scene_id: str, scene_visual_intent: str, scene_on_screen_text: str, requires_animation: bool,
    scene_animation_density: str, category: str, style_profile: dict, script_language: str,
    beats: list, previous_scene_last_animation: Optional[dict],
) -> list[dict]:
    if not requires_animation:
        return []

    beats_context = [
        {
            "beat_id": b["beat_id"],
            "vo_text": b["vo_text"],
            "estimated_duration_seconds": b["estimated_duration_seconds"],
            "entities": b["entities"],
            "scene_direction": b["scene_direction"],
            "animation_signal": b["animation_signal"],
        }
        for b in beats
    ]
    context = {
        "category": category,
        "style_profile": style_profile,
        "script_language": script_language,
        "scene_id": scene_id,
        "scene_visual_intent": scene_visual_intent,
        "scene_on_screen_text": scene_on_screen_text,
        "requires_animation": requires_animation,
        "scene_animation_density": scene_animation_density,
        "beats": beats_context,
        "previous_scene_last_animation": previous_scene_last_animation,
    }
    beat_ids = {b["beat_id"] for b in beats}

    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": ANIMATION_PLANNER_PROMPT},
                    {"role": "user", "content": json.dumps(context)},
                ],
                stream=False,
            )
        )
        _record_token_usage("edit video - animation plan", res)

        content = (res.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

        raw = json.loads(content)
        raw_animations = raw.get("animations")
        if not isinstance(raw_animations, list):
            raise ValueError("animation planner returned no 'animations' list")

    except Exception as e:
        print(f"[edit-video] scene {scene_id} animation planning failed, using no animations: {e}")
        return []

    validated, seen_beat_ids = [], set()
    for raw_anim in raw_animations:
        v = _validate_beat_animation(raw_anim, beat_ids)
        if v and v["beat_id"] not in seen_beat_ids:
            validated.append(v)
            seen_beat_ids.add(v["beat_id"])
    return validated


async def _get_or_create_tagged_text(scene: dict, scene_id, user_id: str, vo_text: str) -> str:
    tags_request = AddScriptTagsRequest(userId=user_id, script=vo_text)
    tags_result = await add_script_tags(tags_request)
    return tags_result["tagged_script"]



async def _process_scene(scene: dict, request: EditVideo, category: str, script_language: str, video_ctx: dict, is_first_scene: bool = False) -> dict:
    scene_out = dict(scene)
    vo_text = scene.get("vo_text", "")
    scene_id = scene.get("scene_id")
    style_profile = STYLE_PROFILES.get(category, STYLE_PROFILES["general_documentary"])
    scene_animation_density = scene.get("scene_animation_density") or style_profile.get("animation_density", "medium")

    known_entities_in = list(video_ctx["known_entities"])
    known_setting_in = dict(video_ctx["known_setting"])
    previous_media_type_in = video_ctx["previous_scene_last_media_type"]
    previous_animation_in = video_ctx["previous_scene_last_animation"]

    scene_out["_beat_director_context"] = {
        "category": category,
        "style_profile": style_profile,
        "script_language": script_language,
        "known_entities": known_entities_in,
        "known_setting": known_setting_in,
        "previous_scene_last_media_type": previous_media_type_in,
    }
    scene_out["_previous_scene_last_animation"] = previous_animation_in

    async def _finalize(timed_words: list) -> dict:
        beats = await _run_beat_director(
            scene_vo_text=vo_text, category=category, style_profile=style_profile, script_language=script_language,
            scene_id=scene_id, scene_visual_intent=scene.get("visual_intent", ""),
            scene_animation_density=scene_animation_density, scene_on_screen_text=scene.get("on_screen_text", ""),
            previous_scene_last_media_type=previous_media_type_in,
            known_entities=known_entities_in, known_setting=known_setting_in, id_prefix=scene_id,
        )

        if timed_words:
            _align_beats_to_timed_words(beats, timed_words)
        else:
            for b in beats:
                b["start"], b["end"] = None, None

        fallback_keywords = _get_scene_broll_keywords(scene)
        await _fetch_beats_media(beats, str(scene_id))
        await _fill_empty_beats(scene, beats, fallback_keywords)
        _dedupe_beats_media_across_scene(beats)

        animations = await _run_animation_planner(
            scene_id=scene_id, scene_visual_intent=scene.get("visual_intent", ""),
            scene_on_screen_text=scene.get("on_screen_text", ""),
            requires_animation=bool(scene.get("requires_animation", False)),
            scene_animation_density=scene_animation_density, category=category,
            style_profile=style_profile, script_language=script_language,
            beats=beats, previous_scene_last_animation=previous_animation_in,
        )

        scene_out["beats"] = beats
        scene_out["media"] = _aggregate_beats_media(beats)
        scene_out["animations"] = animations

        # Advance video-level continuity for the NEXT scene.
        for b in beats:
            for e in b["entities"]:
                if not any(existing["name"].lower() == e["name"].lower() for existing in video_ctx["known_entities"]):
                    video_ctx["known_entities"].append(e)
            loc = b["scene_direction"]["setting"]["location"]
            per = b["scene_direction"]["setting"]["time_period"]
            if loc or per:
                video_ctx["known_setting"] = {
                    "location": loc or video_ctx["known_setting"].get("location", ""),
                    "time_period": per or video_ctx["known_setting"].get("time_period", ""),
                }
        if beats:
            video_ctx["previous_scene_last_media_type"] = beats[-1].get("preferred_media_type")
        if animations:
            last_anim = animations[-1]
            video_ctx["previous_scene_last_animation"] = {
                "animation_type": last_anim["animation_type"],
                "placement": last_anim["placement"],
                "category": last_anim["category"],
            }

        if scene_out.get("start") is not None and scene_out.get("end") is not None:
            scene_out["duration_seconds"] = round(scene_out["end"] - scene_out["start"], 3)
        else:
            scene_out["duration_seconds"] = None
        return scene_out

    if not vo_text.strip():
        scene_out["voiceover"] = None
        scene_out["start"] = None
        scene_out["end"] = None
        scene_out["word_segments"] = []
        scene_out["error"] = None
        return await _finalize([])

    try:
        tagged_text = await _get_or_create_tagged_text(scene, scene_id, request.userId, vo_text)
    except Exception as e:
        print(f"[edit-video] scene {scene_id} tagging failed: {e}")
        scene_out["tagged_vo_text"] = None
        scene_out["voiceover"] = None
        scene_out["start"] = None
        scene_out["end"] = None
        scene_out["word_segments"] = []
        scene_out["error"] = f"voice tagging failed: {e}"
        return await _finalize([])

    try:
        speech_result = await _generate_speech_possibly_chunked(
            user_id=request.userId, tagged_text=tagged_text, voice=request.voice, lang_code=request.langCode,
            volume=request.volume, loudness_normalization=request.loudness_normalization,
            text_normalization=request.text_normalization,
        )
    except Exception as e:
        print(f"[edit-video] scene {scene_id} voice generation failed: {e}")
        scene_out["tagged_vo_text"] = tagged_text
        scene_out["voiceover"] = None
        scene_out["start"] = None
        scene_out["end"] = None
        scene_out["word_segments"] = []
        scene_out["error"] = f"voice generation failed: {e}"
        return await _finalize([])

    try:
        scene_timestamps = await _generate_word_timestamps(speech_result["url"])
    except Exception as e:
        print(f"[edit-video] scene {scene_id} whisperx alignment failed: {e}")
        scene_out["tagged_vo_text"] = tagged_text
        scene_out["voiceover"] = speech_result
        scene_out["start"] = None
        scene_out["end"] = None
        scene_out["word_segments"] = []
        scene_out["error"] = f"timestamp alignment failed: {e}"
        return await _finalize([])

    word_segments = scene_timestamps.get("word_segments", [])
    timed_words = [w for w in word_segments if "start" in w and "end" in w]

    if is_first_scene and timed_words and timed_words[0].get("start", 0.0) > 0.0:
        print(
            f"[edit-video] scene {scene_id}: clamping first word start "
            f"{timed_words[0]['start']:.3f}s -> 0.0s so no leading audio is trimmed"
        )
        first_word_obj = timed_words[0]
        for w in word_segments:
            if w is first_word_obj:
                w["start"] = 0.0
                break
        timed_words[0]["start"] = 0.0

    scene_out["tagged_vo_text"] = tagged_text
    scene_out["voiceover"] = speech_result
    scene_out["start"] = timed_words[0]["start"] if timed_words else None
    scene_out["end"] = timed_words[-1]["end"] if timed_words else None
    scene_out["word_segments"] = word_segments
    scene_out["error"] = None

    return await _finalize(timed_words)


async def _regenerate_scene_beats_and_animations(scene: dict) -> dict:
    """Used by the /beats/rebuild endpoint: re-runs the Beat Director +
    Animation Planner for a single scene using the continuity context that
    was captured when the scene was first processed."""
    ctx = scene.get("_beat_director_context") or {
        "category": "general_documentary",
        "style_profile": STYLE_PROFILES["general_documentary"],
        "script_language": "en",
        "known_entities": [],
        "known_setting": {"location": "", "time_period": ""},
        "previous_scene_last_media_type": None,
    }
    previous_animation = scene.get("_previous_scene_last_animation")
    scene_id = scene.get("scene_id")
    vo_text = scene.get("vo_text", "")
    scene_animation_density = scene.get("scene_animation_density") or ctx["style_profile"].get("animation_density", "medium")

    beats = await _run_beat_director(
        scene_vo_text=vo_text, category=ctx["category"], style_profile=ctx["style_profile"],
        script_language=ctx["script_language"], scene_id=scene_id,
        scene_visual_intent=scene.get("visual_intent", ""), scene_animation_density=scene_animation_density,
        scene_on_screen_text=scene.get("on_screen_text", ""),
        previous_scene_last_media_type=ctx["previous_scene_last_media_type"],
        known_entities=ctx["known_entities"], known_setting=ctx["known_setting"], id_prefix=scene_id,
    )

    timed_words = [w for w in (scene.get("word_segments") or []) if "start" in w and "end" in w]
    if timed_words:
        _align_beats_to_timed_words(beats, timed_words)

    fallback_keywords = _get_scene_broll_keywords(scene)
    await _fetch_beats_media(beats, str(scene_id))
    await _fill_empty_beats(scene, beats, fallback_keywords)
    _dedupe_beats_media_across_scene(beats)

    animations = await _run_animation_planner(
        scene_id=scene_id, scene_visual_intent=scene.get("visual_intent", ""),
        scene_on_screen_text=scene.get("on_screen_text", ""),
        requires_animation=bool(scene.get("requires_animation", False)),
        scene_animation_density=scene_animation_density, category=ctx["category"],
        style_profile=ctx["style_profile"], script_language=ctx["script_language"],
        beats=beats, previous_scene_last_animation=previous_animation,
    )

    scene["beats"] = beats
    scene["media"] = _aggregate_beats_media(beats)
    scene["animations"] = animations
    return scene


async def _rebuild_fragment_beats(scene: dict, local_start: float, local_end: float, id_prefix: str) -> list:
    """Re-runs the Beat Director scoped to a single sub-range of a scene's
    narration (used by split/insert). The Director still decides how many
    beats the fragment needs — usually one, but not necessarily."""
    ctx = scene.get("_beat_director_context") or {
        "category": "general_documentary",
        "style_profile": STYLE_PROFILES["general_documentary"],
        "script_language": "en",
        "known_entities": [],
        "known_setting": {"location": "", "time_period": ""},
        "previous_scene_last_media_type": None,
    }
    word_segments = scene.get("word_segments") or []
    timed_words_all = [w for w in word_segments if "start" in w and "end" in w]
    fragment_timed_words = [
        w for w in timed_words_all if w["start"] >= local_start - 1e-6 and w["start"] < local_end + 1e-6
    ]
    fragment_vo_text = _words_in_range(timed_words_all, local_start, local_end)

    if not fragment_vo_text.strip():
        return [{
            "beat_id": f"{id_prefix}_beat1", "beat_index": 0, "vo_text": "",
            "estimated_duration_seconds": max(1, round(local_end - local_start)),
            "keywords": _get_scene_broll_keywords(scene), "media_type": "video", "preferred_media_type": "video",
            "entities": [], "scene_direction": {"setting": dict(ctx["known_setting"]), "mood": "neutral", "key_action": "narration continues"},
            "animation_signal": _validate_animation_signal(None), "motion_type": _DEFAULT_MOTION_TYPE,
            "start": local_start, "end": local_end,
        }]

    beats = await _run_beat_director(
        scene_vo_text=fragment_vo_text, category=ctx["category"], style_profile=ctx["style_profile"],
        script_language=ctx["script_language"], scene_id=scene.get("scene_id"),
        scene_visual_intent=scene.get("visual_intent", ""),
        scene_animation_density=scene.get("scene_animation_density") or ctx["style_profile"].get("animation_density", "medium"),
        scene_on_screen_text="",  # avoid re-triggering the same on-screen-text checklist item twice
        previous_scene_last_media_type=None, known_entities=ctx["known_entities"], known_setting=ctx["known_setting"],
        id_prefix=id_prefix,
    )

    if fragment_timed_words:
        _align_beats_to_timed_words(beats, fragment_timed_words)
    else:
        for b in beats:
            b["start"], b["end"] = local_start, local_end

    fallback_keywords = _get_scene_broll_keywords(scene)
    await _fetch_beats_media(beats, f"{scene.get('scene_id')}:{id_prefix}")
    await _fill_empty_beats(scene, beats, fallback_keywords)
    return beats


# ---------------------------------------------------------------------------
# Timeline building
# ---------------------------------------------------------------------------

def build_timeline_from_scenes(scenes: list, fps: int = TIMELINE_FPS) -> dict:
    tracks = []
    cumulative_frames = 0

    for scene in scenes:
        scene_id = scene.get("scene_id")
        start_sec = scene.get("start") or 0.0
        end_sec = scene.get("end") or 0.0
        scene_duration_frames = max(_seconds_to_frames(end_sec - start_sec, fps), fps)

        scene_start_frame = cumulative_frames
        scene_end_frame = cumulative_frames + scene_duration_frames

        voiceover = scene.get("voiceover")
        if voiceover and voiceover.get("url"):
            tracks.append({
                "track_id": f"audio_{scene_id}", "scene_id": scene_id, "type": "audio",
                "file_url": voiceover["url"], "vo_text": scene.get("vo_text"),
                "startFrame": scene_start_frame, "endFrame": scene_end_frame,
                "start_sec": scene_start_frame / fps, "end_sec": scene_end_frame / fps,
                "scene_start_sec": start_sec, "scene_end_sec": end_sec,
            })

        word_segments = scene.get("word_segments") or []
        words = []
        for w in word_segments:
            if "start" not in w or "end" not in w:
                continue
            words.append({
                "word": w.get("word", ""),
                "startFrame": scene_start_frame + _seconds_to_frames(w["start"] - start_sec, fps),
                "endFrame": scene_start_frame + _seconds_to_frames(w["end"] - start_sec, fps),
            })
        if words:
            caption_track = {"track_id": f"caption_{scene_id}", "scene_id": scene_id, "type": "caption_word", "words": words}
            caption_track["style"] = {**DEFAULT_CAPTION_STYLE, **(scene.get("caption_style") or {})}
            tracks.append(caption_track)

        beats = scene.get("beats") or []
        if not beats:
            beats = [{
                "beat_id": f"{scene_id}_b1", "start": start_sec, "end": end_sec,
                "media": scene.get("media") or {}, "broll_override": scene.get("broll_override"),
            }]

        beat_frame_ranges = {}
        for beat in beats:
            b_start = beat.get("start")
            b_end = beat.get("end")
            if b_start is None or b_end is None:
                beat_start_frame, beat_end_frame = scene_start_frame, scene_end_frame
            else:
                beat_start_frame = scene_start_frame + _seconds_to_frames(b_start - start_sec, fps)
                beat_end_frame = scene_start_frame + _seconds_to_frames(b_end - start_sec, fps)
                beat_end_frame = min(beat_end_frame, scene_end_frame)
                beat_start_frame = max(scene_start_frame, min(beat_start_frame, beat_end_frame))
            beat_frame_ranges[beat.get("beat_id")] = (beat_start_frame, beat_end_frame)

            default_asset, default_source = _resolve_beat_broll_selection(beat)
            media = beat.get("media") or {}
            video_candidates = (media.get("videos") or {}).get("results") or []
            image_candidates = (media.get("images") or {}).get("results") or []

            broll_track = {
                "track_id": f"broll_{scene_id}_{beat.get('beat_id')}", "scene_id": scene_id,
                "beat_id": beat.get("beat_id"), "type": "broll", "layer": "background",
                "startFrame": beat_start_frame, "endFrame": beat_end_frame,
                "start_sec": beat_start_frame / fps, "end_sec": beat_end_frame / fps,
                "beat_start_sec": b_start, "beat_end_sec": b_end,
                "keywords": beat.get("keywords"), "preferred_media_type": beat.get("preferred_media_type"),
                "motion_type": _resolve_beat_motion_type(beat),
                "selected_asset": {
                    "asset_id": (default_asset or {}).get("id"),
                    "file_url": _resolve_broll_file_url(default_asset, default_source) if default_asset else None,
                    "source": default_source,
                    "width": (default_asset or {}).get("width"),
                    "height": (default_asset or {}).get("height"),
                    "video_files": (default_asset or {}).get("video_files"),
                    "src": (default_asset or {}).get("src"),
                } if default_asset else None,
                "candidates": {"videos": video_candidates, "images": image_candidates},
            }

            background_color = scene.get("background_color")
            if background_color:
                broll_track["background_color"] = background_color

            tracks.append(broll_track)

        for animation in (scene.get("animations") or []):
            beat_id = animation.get("beat_id")
            rng = beat_frame_ranges.get(beat_id)
            if not rng:
                continue
            b_start_frame, b_end_frame = rng
            anim_span = b_end_frame - b_start_frame if b_end_frame > b_start_frame else animation.get("duration_frames", 90)
            anim_start_frame = b_start_frame
            anim_end_frame = min(b_start_frame + animation.get("duration_frames", 90), b_start_frame + max(anim_span, 1))

            tracks.append({
                "track_id": f"anim_{scene_id}_{beat_id}",
                "scene_id": scene_id, "beat_id": beat_id, "type": "animation",
                "layer": animation.get("z_index_layer", "foreground"),
                "animation_type": animation.get("animation_type"), "category": animation.get("category"),
                "placement": animation.get("placement"), "geometry_px": animation.get("geometry_px"),
                "motion": animation.get("motion"), "icon_name": animation.get("icon_name"),
                "icon_layout": animation.get("icon_layout"), "display_text": animation.get("display_text"),
                "color_hint": animation.get("color_hint"), "highlight_target_text": animation.get("highlight_target_text"),
                "content_binding": animation.get("content_binding"), "render_prompt": animation.get("render_prompt"),
                "trigger": animation.get("trigger"),
                "startFrame": anim_start_frame, "endFrame": anim_end_frame,
                "start_sec": anim_start_frame / fps, "end_sec": anim_end_frame / fps,
                "duration_frames": anim_end_frame - anim_start_frame,
                "status": "pending_render" if animation.get("render_engine_hint") == "remotion" else "ready",
                "asset_url": None,
                "render_engine_hint": animation.get("render_engine_hint"),
            })

        cumulative_frames = scene_end_frame

    return {
        "fps": fps, "total_frames": cumulative_frames,
        "resolution": {"width": TIMELINE_WIDTH, "height": TIMELINE_HEIGHT},
        "tracks": tracks,
    }



_TEXT_ONLY_ANIMATION_TYPES = {
    "lower_third", "kinetic_caption", "bullet_list_reveal", "callout_textbox",
    "stat_counter_overlay", "full_screen_title_card", "full_screen_quote_card",
}


def _next_animation_id(raw_scenes: list) -> int:
    max_id = 0
    for s in raw_scenes:
        for a in (s.get("animations") or []):
            aid = a.get("id")
            if isinstance(aid, int) and aid > max_id:
                max_id = aid
    return max_id + 1


def _assign_animation_ids(raw_scenes: list) -> None:
    next_id = _next_animation_id(raw_scenes)
    for s in raw_scenes:
        for a in (s.get("animations") or []):
            if not isinstance(a.get("id"), int):
                a["id"] = next_id
                next_id += 1


def _find_animation_by_id(raw_scenes: list, animation_id: int) -> Optional[tuple]:
    for si, s in enumerate(raw_scenes):
        animations = s.get("animations") or []
        for ai, a in enumerate(animations):
            if a.get("id") == animation_id:
                return si, ai, s.get("scene_id"), a.get("beat_id")
    return None


def _compute_infographics_and_text_lists(raw_scenes: list, timeline: dict) -> tuple[list, list]:
    _assign_animation_ids(raw_scenes)

    anim_timing_by_scene_beat = {}
    for t in (timeline or {}).get("tracks", []):
        if t.get("type") != "animation":
            continue
        anim_timing_by_scene_beat[(t.get("scene_id"), t.get("beat_id"))] = {
            "start": t.get("start_sec"),
            "end": t.get("end_sec"),
        }

    infographics, text_list = [], []
    for scene in raw_scenes:
        scene_id = scene.get("scene_id")
        for anim in (scene.get("animations") or []):
            beat_id = anim.get("beat_id")
            animation_type = anim.get("animation_type")
            has_icon = bool(anim.get("icon_name"))
            timing = anim_timing_by_scene_beat.get((scene_id, beat_id)) or {"start": None, "end": None}
            entry = {
                "id": anim.get("id"),
                "scene_id": scene_id, "beat_id": beat_id,
                "animation_type": animation_type, "category": anim.get("category"),
                "placement": anim.get("placement"), "display_text": anim.get("display_text"),
                "color_hint": anim.get("color_hint"),
                "start": timing["start"], "end": timing["end"],
            }
            is_text_only = animation_type in _TEXT_ONLY_ANIMATION_TYPES and not has_icon
            if is_text_only:
                text_list.append(entry)
            else:
                infographics.append(entry)
    return infographics, text_list

def _compute_broll_list(timeline: dict) -> list:
    return [
        {
            "track_id": t.get("track_id"), "scene_id": t.get("scene_id"), "beat_id": t.get("beat_id"),
            "start_sec": t.get("start_sec"), "end_sec": t.get("end_sec"),
            "selected_asset": _slim_selected_asset(t.get("selected_asset")),
        }
        for t in timeline.get("tracks", []) if t.get("type") == "broll"
    ]


# ---------------------------------------------------------------------------
# Response slimming
# ---------------------------------------------------------------------------

def _slim_beat_for_response(beat: dict, broll_track: Optional[dict] = None) -> dict:
    default_asset, default_source = _resolve_beat_broll_selection(beat)
    selected = None
    if default_asset:
        selected = {
            "asset_id": default_asset.get("id"), "file_url": _resolve_broll_file_url(default_asset, default_source),
            "source": default_source, "width": default_asset.get("width"), "height": default_asset.get("height"),
        }
    return {
        "beat_id": beat.get("beat_id"), "start": beat.get("start"), "end": beat.get("end"),
        "start_sec": (broll_track or {}).get("start_sec"), "end_sec": (broll_track or {}).get("end_sec"),
        "vo_text": beat.get("vo_text"), "keywords": beat.get("keywords"),
        "preferred_media_type": beat.get("preferred_media_type"), "motion_type": _resolve_beat_motion_type(beat),
        "entities": beat.get("entities"), "scene_direction": beat.get("scene_direction"),
        "selected_asset": selected,
    }


def _slim_scene_for_response(scene: dict, timeline: Optional[dict] = None) -> dict:
    scene_id = scene.get("scene_id")

    broll_track_by_beat_id = {}
    animation_track_by_beat_id = {}
    if timeline:
        for t in timeline.get("tracks", []):
            if t.get("type") == "broll" and t.get("scene_id") == scene_id:
                broll_track_by_beat_id[t.get("beat_id")] = t
            elif t.get("type") == "animation" and t.get("scene_id") == scene_id:
                animation_track_by_beat_id[t.get("beat_id")] = t

    animations_out = []
    for anim in (scene.get("animations") or []):
        anim = dict(anim)
        track = animation_track_by_beat_id.get(anim.get("beat_id"))
        if track:
            anim["start_sec"] = track.get("start_sec")
            anim["end_sec"] = track.get("end_sec")
        animations_out.append(anim)

    return {
        "scene_id": scene_id, "vo_text": scene.get("vo_text"), "visual_intent": scene.get("visual_intent"),
        "on_screen_text": scene.get("on_screen_text"), "requires_animation": scene.get("requires_animation"),
        "scene_animation_density": scene.get("scene_animation_density"),
        "start": scene.get("start"), "end": scene.get("end"), "duration_seconds": scene.get("duration_seconds"),
        "voice_url": (scene.get("voiceover") or {}).get("url"), "error": scene.get("error"),
        "beats": [_slim_beat_for_response(b, broll_track_by_beat_id.get(b.get("beat_id"))) for b in (scene.get("beats") or [])],
        "animations": animations_out,
        "caption_style": scene.get("caption_style"), "background_color": scene.get("background_color"),
    }


def _slim_timeline_for_response(timeline: dict) -> dict:
    slim_tracks = []
    for track in timeline.get("tracks", []):
        if track.get("type") != "broll":
            slim_tracks.append(track)
            continue
        slim_tracks.append({
            "track_id": track.get("track_id"), "type": "broll", "scene_id": track.get("scene_id"),
            "beat_id": track.get("beat_id"), "layer": track.get("layer"),
            "startFrame": track.get("startFrame"), "endFrame": track.get("endFrame"),
            "beat_start_sec": track.get("beat_start_sec"), "beat_end_sec": track.get("beat_end_sec"),
            "keywords": track.get("keywords"), "preferred_media_type": track.get("preferred_media_type"),
            "motion_type": track.get("motion_type"), "selected_asset": _slim_selected_asset(track.get("selected_asset")),
            "background_color": track.get("background_color"),
        })
    return {**timeline, "tracks": slim_tracks}



@app.post("/edit-video")
async def edit_video(request: EditVideo):
    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": SCRIPT_SCENE_PROMPT},
                    {"role": "user", "content": request.script},
                ],
                stream=False,
            )
        )
        _record_token_usage("edit video", res)

        content = (res.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

    except HTTPException:
        raise
    except Exception as e:
        print(f"[edit-video] scene generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate scenes")

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "scenes" in parsed:
            scenes = parsed["scenes"]
            category = parsed.get("category") or "general_documentary"
            script_language = parsed.get("script_language") or "en"
        elif isinstance(parsed, list):
            scenes = parsed
            category = "general_documentary"
            script_language = "en"
        else:
            raise ValueError(f"Unexpected JSON shape: {type(parsed)}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[edit-video] JSON parse failed: {e} | raw content: {content[:500]}")
        raise HTTPException(status_code=502, detail="Model did not return valid JSON")

    if category not in STYLE_PROFILES:
        print(f"[edit-video] model returned unknown category {category!r} — defaulting to general_documentary")
        category = "general_documentary"


    video_ctx = {
        "known_entities": [], "known_setting": {"location": "", "time_period": ""},
        "previous_scene_last_media_type": None, "previous_scene_last_animation": None,
    }
    scenes_with_voice_and_timestamps = []
    for idx, scene in enumerate(scenes):
        scene_result = await _process_scene(
            scene, request, category, script_language, video_ctx, is_first_scene=(idx == 0)
        )
        scenes_with_voice_and_timestamps.append(scene_result)

    failed_scenes = [s["scene_id"] for s in scenes_with_voice_and_timestamps if s.get("error")]
    if failed_scenes:
        print(f"[edit-video] completed with {len(failed_scenes)} failed scene(s): {failed_scenes}")

    timeline_json = build_timeline_from_scenes(scenes_with_voice_and_timestamps)

    scene_timings = [
        {
            "scene_id": s.get("scene_id"), "start": s.get("start"), "end": s.get("end"),
            "duration_seconds": s.get("duration_seconds"),
            "beats": [{"beat_id": b.get("beat_id"), "start": b.get("start"), "end": b.get("end")} for b in (s.get("beats") or [])],
        }
        for s in scenes_with_voice_and_timestamps
    ]

    infographics, text_overlays = _compute_infographics_and_text_lists(scenes_with_voice_and_timestamps, timeline_json)
    broll_list = _compute_broll_list(timeline_json)

    video_id = str(uuid.uuid4())
    try:
        supabase.table("videos").insert({
            "id": video_id, "user_id": request.userId, "script": request.script, "voice": request.voice,
            "lang_code": request.langCode, "category": category, "script_language": script_language,
            "timeline_json": timeline_json, "timeline_version": 1,
            "raw_scenes": scenes_with_voice_and_timestamps, "scene_timings": scene_timings,
            "infographics_list": infographics, "text_list": text_overlays, "broll_list": broll_list,
        }).execute()
    except Exception as e:
        print(f"[edit-video] failed to persist video row: {e}")
        raise HTTPException(status_code=500, detail="Failed to save video")

    return {
        "video_id": video_id, "category": category, "script_language": script_language,
        "timeline": _slim_timeline_for_response(timeline_json),
        "scenes": [_slim_scene_for_response(s, timeline_json) for s in scenes_with_voice_and_timestamps],
        "scene_timings": scene_timings, "failed_scene_ids": failed_scenes,
        "infographics_list": infographics, "text_list": text_overlays, "broll_list": broll_list,
    }


@app.get("/timeline/{video_id}")
async def get_timeline(video_id: str):
    try:
        row = (
            supabase.table("videos")
            .select("timeline_json, timeline_version, scene_timings, infographics_list, text_list, broll_list")
            .eq("id", video_id).single().execute()
        )
    except Exception as e:
        print(f"[get-timeline] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    data = dict(row.data)
    if data.get("timeline_json"):
        data["timeline_json"] = _slim_timeline_for_response(data["timeline_json"])
    return data


@app.patch("/timeline/{video_id}")
async def patch_timeline(video_id: str, patch: TrackPatch):
    try:
        row = (
            supabase.table("videos").select("timeline_json, timeline_version, raw_scenes")
            .eq("id", video_id).single().execute()
        )
    except Exception as e:
        print(f"[patch-timeline] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    timeline = row.data["timeline_json"]
    current_version = row.data.get("timeline_version", 1)
    raw_scenes = row.data.get("raw_scenes") or []

    track_found = None
    for track in timeline.get("tracks", []):
        if track.get("track_id") == patch.track_id:
            track.update(patch.updates)
            track_found = track
            break

    if not track_found:
        raise HTTPException(status_code=404, detail=f"Track {patch.track_id} not found in timeline")

    scene_id = track_found.get("scene_id")
    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)

    visual_change = False
    if scene_index is not None:
        scene = dict(raw_scenes[scene_index])

        if track_found.get("type") == "broll":
            beat_id = track_found.get("beat_id")
            beats = scene.get("beats") or []
            beat_idx = next((i for i, b in enumerate(beats) if b.get("beat_id") == beat_id), None)

            if "selected_asset" in patch.updates and patch.updates["selected_asset"]:
                sel = patch.updates["selected_asset"]
                override = {
                    "asset_id": sel.get("asset_id") or sel.get("id"), "source": sel.get("source"),
                    "file_url": sel.get("file_url"), "width": sel.get("width"), "height": sel.get("height"),
                    "video_files": sel.get("video_files"), "src": sel.get("src"), "motion_type": sel.get("motion_type"),
                }
                if beat_idx is not None:
                    beats[beat_idx] = {**beats[beat_idx], "broll_override": override}
                    scene["beats"] = beats
                else:
                    scene["broll_override"] = override
                visual_change = True

            if "background_color" in patch.updates:
                if patch.updates["background_color"]:
                    _validate_hex_color(patch.updates["background_color"], "background_color")
                scene["background_color"] = patch.updates["background_color"]
                visual_change = True

        elif track_found.get("type") == "caption_word":
            if "style" in patch.updates and isinstance(patch.updates["style"], dict):
                existing_style = scene.get("caption_style") or {}
                scene["caption_style"] = {**existing_style, **patch.updates["style"]}
                visual_change = True

        elif track_found.get("type") == "animation":
            beat_id = track_found.get("beat_id")
            animations = scene.get("animations") or []
            anim_idx = next((i for i, a in enumerate(animations) if a.get("beat_id") == beat_id), None)
            if anim_idx is not None and "duration_frames" in patch.updates:
                try:
                    new_duration = int(patch.updates["duration_frames"])
                    if new_duration <= 0 or new_duration > 900:
                        raise ValueError
                except (TypeError, ValueError):
                    raise HTTPException(status_code=422, detail="duration_frames must be an integer between 1 and 900 (30s @30fps)")
                animations[anim_idx] = {**animations[anim_idx], "duration_frames": new_duration}
                scene["animations"] = animations
                visual_change = True

        raw_scenes[scene_index] = scene

    new_version = current_version + 1
    update_payload = {"timeline_json": timeline, "timeline_version": new_version}
    if scene_index is not None:
        update_payload["raw_scenes"] = raw_scenes
    if visual_change:
        update_payload["final_video_url"] = None
        update_payload["render_status"] = "stale_needs_render"

    try:
        supabase.table("videos").update(update_payload).eq("id", video_id).execute()
    except Exception as e:
        print(f"[patch-timeline] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save edit")

    return {"timeline_version": new_version, "track_id": patch.track_id, "needs_render": visual_change}


@app.patch("/timeline/{video_id}/scene/{scene_id}/style")
async def update_scene_style(video_id: str, scene_id: str, update: SceneStyleUpdate):
    """Pure burned-in-caption styling. Overlay/infographic text is edited
    via PATCH .../beat/{beat_id}/animation instead — see BeatAnimationUpdate."""
    if update.animation_type is not None and update.animation_type not in _VALID_CAPTION_ANIMATION_TYPES:
        raise HTTPException(status_code=422, detail=f"animation_type must be one of {sorted(_VALID_CAPTION_ANIMATION_TYPES)}")
    _validate_hex_color(update.text_color, "text_color")
    _validate_hex_color(update.outline_color, "outline_color")
    _validate_hex_color(update.background_color, "background_color")

    if (
        update.font_size is None and update.text_color is None and update.outline_color is None
        and update.animation_type is None and update.background_color is None and update.vertical_position is None
        and update.margin_bottom_percent is None and update.horizontal_position is None and update.margin_horizontal_percent is None
    ):
        raise HTTPException(status_code=422, detail="Provide at least one field to update")

    try:
        row = supabase.table("videos").select("raw_scenes, timeline_version").eq("id", video_id).single().execute()
    except Exception as e:
        print(f"[update-scene-style] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])

    style_fields = {
        "font_size": update.font_size, "text_color": update.text_color, "outline_color": update.outline_color,
        "animation_type": update.animation_type, "vertical_position": update.vertical_position,
        "margin_bottom_percent": update.margin_bottom_percent, "horizontal_position": update.horizontal_position,
        "margin_horizontal_percent": update.margin_horizontal_percent,
    }
    style_fields = {k: v for k, v in style_fields.items() if v is not None}
    if style_fields:
        existing_style = scene.get("caption_style") or {}
        scene["caption_style"] = {**existing_style, **style_fields}

    if update.background_color is not None:
        scene["background_color"] = update.background_color

    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes, "timeline_json": timeline_json, "timeline_version": new_version,
            "final_video_url": None, "render_status": "stale_needs_render",
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[update-scene-style] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save style edit")

    return {
        "video_id": video_id, "scene_id": scene_id, "timeline_version": new_version,
        "caption_style": scene.get("caption_style"), "background_color": scene.get("background_color"),
        "timeline": timeline_json, "needs_render": True,
    }


@app.patch("/timeline/{video_id}/scene/{scene_id}/trim")
async def update_scene_trim(video_id: str, scene_id: str, update: SceneTrimUpdate):
    if update.start < 0 or update.end <= update.start:
        raise HTTPException(status_code=422, detail="`end` must be greater than `start`, and `start` must be >= 0")

    try:
        row = supabase.table("videos").select("raw_scenes, timeline_version").eq("id", video_id).single().execute()
    except Exception as e:
        print(f"[update-scene-trim] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])

    full_word_segments = scene.get("word_segments_full")
    if full_word_segments is None:
        full_word_segments = scene.get("word_segments") or []
        scene["word_segments_full"] = full_word_segments

    if not full_word_segments:
        raise HTTPException(status_code=400, detail=f"Scene {scene_id} has no word-level timestamps to trim against")

    clip_start = full_word_segments[0].get("start", 0.0)
    clip_end = full_word_segments[-1].get("end", 0.0)

    if update.start < clip_start - 1e-3 or update.end > clip_end + 1e-3:
        raise HTTPException(
            status_code=422,
            detail=f"Trim range [{update.start}, {update.end}] is outside this scene's original audio bounds [{clip_start}, {clip_end}]",
        )

    trimmed_words = [
        w for w in full_word_segments if "start" in w and "end" in w and w["start"] >= update.start and w["end"] <= update.end
    ]

    scene["trim"] = {"start": update.start, "end": update.end}
    scene["word_segments"] = trimmed_words
    scene["start"] = trimmed_words[0]["start"] if trimmed_words else update.start
    scene["end"] = trimmed_words[-1]["end"] if trimmed_words else update.end
    scene["error"] = None

    scene = await _regenerate_scene_beats_and_animations(scene)
    scene["duration_seconds"] = round(scene["end"] - scene["start"], 3)

    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1
    infographics_list, text_list = _compute_infographics_and_text_lists(raw_scenes, timeline_json)

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes, "timeline_json": timeline_json, "timeline_version": new_version,
            "final_video_url": None, "render_status": "stale_needs_render",
            "infographics_list": infographics_list, "text_list": text_list,
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[update-scene-trim] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save trim edit")

    return {
        "video_id": video_id, "scene_id": scene_id, "trim": scene["trim"], "timeline_version": new_version,
        "infographics_list": infographics_list, "text_list": text_list,
        "timeline": timeline_json, "needs_render": True,
    }


@app.post("/timeline/{video_id}/scene/{scene_id}/beat/{beat_id}/split")
async def split_beat(video_id: str, scene_id: str, beat_id: str, update: BeatSplitUpdate):
    try:
        row = supabase.table("videos").select("raw_scenes, timeline_version").eq("id", video_id).single().execute()
    except Exception as e:
        print(f"[split-beat] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])
    beats = scene.get("beats") or []

    beat_idx = next((i for i, b in enumerate(beats) if b.get("beat_id") == beat_id), None)
    if beat_idx is None:
        raise HTTPException(status_code=404, detail=f"Beat {beat_id} not found in scene {scene_id} (available: {[b.get('beat_id') for b in beats]})")

    beat = beats[beat_idx]
    b_start, b_end = beat.get("start"), beat.get("end")
    if b_start is None or b_end is None:
        raise HTTPException(status_code=422, detail=f"Beat {beat_id} has no start/end timing to split (silent/implicit beat)")

    MIN_HALF_SECONDS = 1.5
    if update.split_at <= b_start + MIN_HALF_SECONDS or update.split_at >= b_end - MIN_HALF_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=f"split_at must leave at least {MIN_HALF_SECONDS}s on each side — beat spans [{b_start}, {b_end}], got split_at={update.split_at}",
        )

    id_prefix_a = f"{beat_id}_split_{uuid.uuid4().hex[:6]}_a"
    id_prefix_b = f"{beat_id}_split_{uuid.uuid4().hex[:6]}_b"

    first_beats = await _rebuild_fragment_beats(scene, b_start, update.split_at, id_prefix_a)
    second_beats = await _rebuild_fragment_beats(scene, update.split_at, b_end, id_prefix_b)

    new_beats = beats[:beat_idx] + first_beats + second_beats + beats[beat_idx + 1:]
    for i, b in enumerate(new_beats):
        b["beat_index"] = i

    fallback_keywords = _get_scene_broll_keywords(scene)
    await _fill_empty_beats(scene, new_beats, fallback_keywords)
    _dedupe_beats_media_across_scene(new_beats)

    animations = [a for a in (scene.get("animations") or []) if a.get("beat_id") != beat_id]

    scene["beats"] = new_beats
    scene["animations"] = animations
    scene["media"] = _aggregate_beats_media(new_beats)
    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1
    broll_list = _compute_broll_list(timeline_json)
    infographics_list, text_list = _compute_infographics_and_text_lists(raw_scenes, timeline_json)

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes, "timeline_json": timeline_json, "timeline_version": new_version,
            "final_video_url": None, "render_status": "stale_needs_render", "broll_list": broll_list,
            "infographics_list": infographics_list, "text_list": text_list,
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[split-beat] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save beat split")

    return {
        "video_id": video_id, "scene_id": scene_id, "original_beat_id": beat_id,
        "new_beats": [
            {"beat_id": b["beat_id"], "start": b.get("start"), "end": b.get("end"),
             "media_type": b.get("preferred_media_type"), "motion_type": b.get("motion_type")}
            for b in (first_beats + second_beats)
        ],
        "broll_list": broll_list, "infographics_list": infographics_list, "text_list": text_list,
        "timeline_version": new_version, "timeline": timeline_json, "needs_render": True,
    }

@app.post("/timeline/{video_id}/scene/{scene_id}/beat/{beat_id}/insert")
async def insert_beat(video_id: str, scene_id: str, beat_id: str, update: BeatInsertUpdate):
    MIN_SECONDS = 1.0
    if update.end <= update.start:
        raise HTTPException(status_code=422, detail="`end` must be greater than `start`")
    if update.end - update.start < MIN_SECONDS:
        raise HTTPException(status_code=422, detail=f"the new clip must be at least {MIN_SECONDS}s long")

    try:
        row = supabase.table("videos").select("raw_scenes, timeline_version").eq("id", video_id).single().execute()
    except Exception as e:
        print(f"[insert-beat] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])
    beats = scene.get("beats") or []

    beat_idx = next((i for i, b in enumerate(beats) if b.get("beat_id") == beat_id), None)
    if beat_idx is None:
        raise HTTPException(status_code=404, detail=f"Beat {beat_id} not found in scene {scene_id} (available: {[b.get('beat_id') for b in beats]})")

    beat = beats[beat_idx]
    b_start, b_end = beat.get("start"), beat.get("end")
    if b_start is None or b_end is None:
        raise HTTPException(status_code=422, detail=f"Beat {beat_id} has no start/end timing to insert into (silent/implicit beat)")
    if update.start < b_start - 1e-3 or update.end > b_end + 1e-3:
        raise HTTPException(status_code=422, detail=f"[{update.start}, {update.end}] must fall inside beat {beat_id}'s own [{b_start}, {b_end}]")

    id_prefix = f"{beat_id}_insert_{uuid.uuid4().hex[:6]}"
    new_beats = await _rebuild_fragment_beats(scene, update.start, update.end, id_prefix)

    pieces = []
    if update.start > b_start + 1e-3:
        before = dict(beat)
        before["beat_id"] = f"{beat_id}_pre"
        before["start"], before["end"] = b_start, update.start
        pieces.append(before)

    pieces.extend(new_beats)

    if update.end < b_end - 1e-3:
        after = dict(beat)
        after["beat_id"] = f"{beat_id}_post"
        after["start"], after["end"] = update.end, b_end
        pieces.append(after)

    combined_beats = beats[:beat_idx] + pieces + beats[beat_idx + 1:]
    for i, b in enumerate(combined_beats):
        b["beat_index"] = i

    fallback_keywords = _get_scene_broll_keywords(scene)
    await _fill_empty_beats(scene, combined_beats, fallback_keywords)
    _dedupe_beats_media_across_scene(combined_beats)

    scene["beats"] = combined_beats
    scene["media"] = _aggregate_beats_media(combined_beats)
    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1
    broll_list = _compute_broll_list(timeline_json)
    infographics_list, text_list = _compute_infographics_and_text_lists(raw_scenes, timeline_json)

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes, "timeline_json": timeline_json, "timeline_version": new_version,
            "final_video_url": None, "render_status": "stale_needs_render", "broll_list": broll_list,
            "infographics_list": infographics_list, "text_list": text_list,
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[insert-beat] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save beat insert")

    return {
        "video_id": video_id, "scene_id": scene_id, "original_beat_id": beat_id,
        "new_beats": [
            {"beat_id": b["beat_id"], "start": b.get("start"), "end": b.get("end"),
             "media_type": b.get("preferred_media_type"), "motion_type": b.get("motion_type")}
            for b in new_beats
        ],
        "broll_list": broll_list, "infographics_list": infographics_list, "text_list": text_list,
        "timeline_version": new_version, "timeline": timeline_json, "needs_render": True,
    }


@app.post("/timeline/{video_id}/scene/{scene_id}/beats/rebuild")
async def rebuild_scene_beats(video_id: str, scene_id: str):
    try:
        row = supabase.table("videos").select("raw_scenes, timeline_version").eq("id", video_id).single().execute()
    except Exception as e:
        print(f"[rebuild-beats] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])
    realigned = False

    has_timed_words = any("start" in w and "end" in w for w in (scene.get("word_segments") or []))
    voiceover = scene.get("voiceover") or {}

    if not has_timed_words and voiceover.get("url"):
        print(f"[rebuild-beats] scene {scene_id} has a voiceover but no timed word_segments — re-running WhisperX alignment first")
        try:
            scene_timestamps = await _generate_word_timestamps(voiceover["url"])
            word_segments = scene_timestamps.get("word_segments", [])
            timed_words = [w for w in word_segments if "start" in w and "end" in w]
            if timed_words:
                scene["word_segments"] = word_segments
                scene["start"] = timed_words[0]["start"]
                scene["end"] = timed_words[-1]["end"]
                realigned = True
            else:
                print(f"[rebuild-beats][WARN] scene {scene_id}: re-alignment produced no timed words")
        except Exception as e:
            print(f"[rebuild-beats][WARN] scene {scene_id}: WhisperX re-alignment failed ({e})")
    elif not has_timed_words:
        print(f"[rebuild-beats] scene {scene_id} has no voiceover to re-align against — will produce a single implicit beat")

    scene = await _regenerate_scene_beats_and_animations(scene)
    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1
    infographics_list, text_list = _compute_infographics_and_text_lists(raw_scenes, timeline_json)

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes, "timeline_json": timeline_json, "timeline_version": new_version,
            "final_video_url": None, "render_status": "stale_needs_render",
            "infographics_list": infographics_list, "text_list": text_list,
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[rebuild-beats] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save rebuilt beats")

    return {
        "video_id": video_id, "scene_id": scene_id, "realigned_word_timing": realigned,
        "beats": [{"beat_id": b["beat_id"], "start": b.get("start"), "end": b.get("end")} for b in scene["beats"]],
        "animations": scene.get("animations"),
        "infographics_list": infographics_list, "text_list": text_list,
        "timeline_version": new_version, "timeline": timeline_json, "needs_render": True,
    }



@app.patch("/timeline/{video_id}/scene/{scene_id}/beat/{beat_id}/animation")
async def update_beat_animation(video_id: str, scene_id: str, beat_id: str, update: BeatAnimationUpdate):
    """Create or edit the animation attached to a specific beat. Replaces
    the old scene-level `update_scene_infographic` endpoint, since a scene
    can now carry many beat-owned animations rather than one."""
    provided = {k: v for k, v in update.dict().items() if v is not None}
    if not provided:
        raise HTTPException(status_code=422, detail="Provide at least one field to update")

    if "color_hint" in provided:
        _validate_hex_color(provided["color_hint"], "color_hint")

    try:
        row = supabase.table("videos").select("raw_scenes, timeline_version").eq("id", video_id).single().execute()
    except Exception as e:
        print(f"[update-beat-animation] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])
    beats = scene.get("beats") or []
    beat_ids = {b.get("beat_id") for b in beats}
    if beat_id not in beat_ids:
        raise HTTPException(status_code=404, detail=f"Beat {beat_id} not found in scene {scene_id}")

    animations = list(scene.get("animations") or [])
    anim_idx = next((i for i, a in enumerate(animations) if a.get("beat_id") == beat_id), None)

    if anim_idx is None:
        if not provided.get("animation_type"):
            raise HTTPException(status_code=422, detail="animation_type is required to create a new animation on this beat")
        merged_raw = {"beat_id": beat_id, **provided}
    else:
        merged_raw = {**animations[anim_idx], **provided, "beat_id": beat_id}

    validated = _validate_beat_animation(merged_raw, {beat_id})
    if not validated:
        raise HTTPException(status_code=422, detail=f"animation_type must be one of {sorted(_VALID_ANIMATION_TYPES)}")
    if isinstance(merged_raw.get("id"), int):
        # Editing an existing animation — keep its id stable rather than
        # letting _assign_animation_ids (called inside
        # _compute_infographics_and_text_lists below) treat this as a new
        # entry and hand it a fresh one.
        validated["id"] = merged_raw["id"]

    if anim_idx is None:
        animations.append(validated)
    else:
        animations[anim_idx] = validated

    scene["animations"] = animations
    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1
    infographics_list, text_list = _compute_infographics_and_text_lists(raw_scenes, timeline_json)

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes, "timeline_json": timeline_json, "timeline_version": new_version,
            "final_video_url": None, "render_status": "stale_needs_render",
            "infographics_list": infographics_list, "text_list": text_list,
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[update-beat-animation] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save animation edit")

    return {
        "video_id": video_id, "scene_id": scene_id, "beat_id": beat_id, "animation": validated,
        "timeline_version": new_version, "infographics_list": infographics_list, "text_list": text_list,
        "timeline": timeline_json, "needs_render": True,
    }


@app.delete("/timeline/{video_id}/scene/{scene_id}/beat/{beat_id}/animation")
async def delete_beat_animation(video_id: str, scene_id: str, beat_id: str):
    """Replaces the old int-id `delete_overlay_by_id` endpoint, which
    assumed exactly one overlay per scene. Overlays are now addressed by
    (scene_id, beat_id) since a scene can carry many."""
    try:
        row = supabase.table("videos").select("raw_scenes, timeline_version").eq("id", video_id).single().execute()
    except Exception as e:
        print(f"[delete-beat-animation] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])
    animations = scene.get("animations") or []
    remaining = [a for a in animations if a.get("beat_id") != beat_id]
    if len(remaining) == len(animations):
        raise HTTPException(status_code=422, detail=f"Beat {beat_id} (scene {scene_id}) has no animation to delete")

    scene["animations"] = remaining
    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1
    infographics_list, text_list = _compute_infographics_and_text_lists(raw_scenes, timeline_json)

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes, "timeline_json": timeline_json, "timeline_version": new_version,
            "final_video_url": None, "render_status": "stale_needs_render",
            "infographics_list": infographics_list, "text_list": text_list,
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[delete-beat-animation] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete animation")

    return {
        "video_id": video_id, "scene_id": scene_id, "beat_id": beat_id,
        "infographics_list": infographics_list, "text_list": text_list,
        "timeline_version": new_version, "timeline": timeline_json, "needs_render": True,
    }


@app.patch("/timeline/{video_id}/animation/{animation_id}")
async def update_animation_by_id(video_id: str, animation_id: int, update: BeatAnimationUpdate):
    """
    Same edit as PATCH .../beat/{beat_id}/animation, but located purely
    by the animation's stable integer id — no need to already know which
    scene/beat it lives on. Reuses the same validation
    (_validate_beat_animation) and preserves the id across the edit.
    """
    provided = {k: v for k, v in update.dict().items() if v is not None}
    if not provided:
        raise HTTPException(status_code=422, detail="Provide at least one field to update")

    if "color_hint" in provided:
        _validate_hex_color(provided["color_hint"], "color_hint")

    try:
        row = supabase.table("videos").select("raw_scenes, timeline_version").eq("id", video_id).single().execute()
    except Exception as e:
        print(f"[update-animation-by-id] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    found = _find_animation_by_id(raw_scenes, animation_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"No animation with id {animation_id} in this video")
    scene_index, anim_index, scene_id, beat_id = found

    scene = dict(raw_scenes[scene_index])
    animations = list(scene.get("animations") or [])
    merged_raw = {**animations[anim_index], **provided, "beat_id": beat_id}

    validated = _validate_beat_animation(merged_raw, {beat_id})
    if not validated:
        raise HTTPException(status_code=422, detail=f"animation_type must be one of {sorted(_VALID_ANIMATION_TYPES)}")
    validated["id"] = animation_id

    animations[anim_index] = validated
    scene["animations"] = animations
    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1
    infographics_list, text_list = _compute_infographics_and_text_lists(raw_scenes, timeline_json)

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes, "timeline_json": timeline_json, "timeline_version": new_version,
            "final_video_url": None, "render_status": "stale_needs_render",
            "infographics_list": infographics_list, "text_list": text_list,
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[update-animation-by-id] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save animation edit")

    return {
        "video_id": video_id, "animation_id": animation_id, "scene_id": scene_id, "beat_id": beat_id,
        "animation": validated, "timeline_version": new_version,
        "infographics_list": infographics_list, "text_list": text_list,
        "timeline": timeline_json, "needs_render": True,
    }


@app.delete("/timeline/{video_id}/animation/{animation_id}")
async def delete_animation_by_id(video_id: str, animation_id: int):
    """
    Same delete as DELETE .../beat/{beat_id}/animation, but located
    purely by the animation's stable integer id — this is the "easy
    deletion" path: a client holding just the `id` from text_list or
    infographics_list can delete directly, without also tracking which
    scene/beat it came from.
    """
    try:
        row = supabase.table("videos").select("raw_scenes, timeline_version").eq("id", video_id).single().execute()
    except Exception as e:
        print(f"[delete-animation-by-id] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    found = _find_animation_by_id(raw_scenes, animation_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"No animation with id {animation_id} in this video")
    scene_index, anim_index, scene_id, beat_id = found

    scene = dict(raw_scenes[scene_index])
    animations = list(scene.get("animations") or [])
    del animations[anim_index]
    scene["animations"] = animations
    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1
    infographics_list, text_list = _compute_infographics_and_text_lists(raw_scenes, timeline_json)

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes, "timeline_json": timeline_json, "timeline_version": new_version,
            "final_video_url": None, "render_status": "stale_needs_render",
            "infographics_list": infographics_list, "text_list": text_list,
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[delete-animation-by-id] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete animation")

    return {
        "video_id": video_id, "animation_id": animation_id, "scene_id": scene_id, "beat_id": beat_id,
        "infographics_list": infographics_list, "text_list": text_list,
        "timeline_version": new_version, "timeline": timeline_json, "needs_render": True,
    }


@app.patch("/timeline/{video_id}/scene/{scene_id}/broll")
async def update_scene_broll(video_id: str, scene_id: str, update: SceneBrollSelectUpdate):
    if update.source not in ("video", "image"):
        raise HTTPException(status_code=422, detail="source must be 'video' or 'image'")
    if update.motion_type is not None and update.motion_type not in _VALID_MOTION_TYPES:
        raise HTTPException(status_code=422, detail=f"motion_type must be one of {sorted(_VALID_MOTION_TYPES)}")
    if (update.start is None) != (update.end is None):
        raise HTTPException(status_code=422, detail="start and end must be provided together")
    if update.start is not None and update.end is not None and update.end <= update.start:
        raise HTTPException(status_code=422, detail="`end` must be greater than `start`")

    try:
        row = supabase.table("videos").select("raw_scenes, timeline_version").eq("id", video_id).single().execute()
    except Exception as e:
        print(f"[update-scene-broll] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])
    beats = scene.get("beats") or []

    if not beats:
        media = scene.get("media") or {}
        beats = [{
            "beat_id": f"{scene_id}_b1", "beat_index": 0, "start": scene.get("start"), "end": scene.get("end"),
            "vo_text": scene.get("vo_text", ""), "keywords": None, "motion_type": _DEFAULT_MOTION_TYPE,
            "preferred_media_type": None,
            "media": {
                "videos": media.get("videos") or {"total_results": 0, "results": [], "error": None},
                "images": media.get("images") or {"total_results": 0, "results": [], "error": None},
            },
        }]
        print(f"[update-scene-broll] scene {scene_id} had no persisted beats — materializing implicit '{scene_id}_b1'")

    if update.beat_id:
        beat_idx = next((i for i, b in enumerate(beats) if b.get("beat_id") == update.beat_id), None)
        if beat_idx is None:
            raise HTTPException(status_code=404, detail=f"Beat {update.beat_id} not found in scene {scene_id} (available: {[b.get('beat_id') for b in beats]})")
    else:
        beat_idx = next((i for i, b in enumerate(beats) if _find_beat_broll_candidate(b, update.asset_id, update.source)), 0)

    beat = dict(beats[beat_idx])

    candidate = _find_beat_broll_candidate(beat, update.asset_id, update.source)
    if candidate is None:
        candidate = await _fetch_pexels_asset_by_id(update.asset_id, update.source)

    if candidate is None:
        raise HTTPException(status_code=404, detail=f"No {update.source} candidate with id {update.asset_id} in beat {beat.get('beat_id')}'s media pool, and it couldn't be fetched directly from Pexels either")

    file_url = _resolve_broll_file_url(candidate, update.source)
    if not file_url:
        raise HTTPException(status_code=422, detail=f"Candidate {update.asset_id} has no landscape/horizontal file available — only horizontal videos are allowed")

    if "background_color" in scene:
        scene["_previous_background_color"] = scene["background_color"]
        scene.pop("background_color", None)

    beat["broll_override"] = {
        "asset_id": candidate.get("id"), "source": update.source, "file_url": file_url,
        "width": candidate.get("width"), "height": candidate.get("height"),
        "video_files": candidate.get("video_files"), "src": candidate.get("src"), "motion_type": update.motion_type,
    }

    if update.start is not None and update.end is not None:
        scene_start = scene.get("start")
        scene_end = scene.get("end")

        if beat_idx == 0 and scene_start is not None and update.start < scene_start - 1e-3:
            if update.start < 0:
                raise HTTPException(status_code=422, detail=f"start {update.start} cannot be negative")
            print(
                f"[update-scene-broll] scene {scene_id}: extending scene start {scene_start} -> {update.start} "
                f"so beat {beat.get('beat_id')} can visually begin before the voice's first word"
            )
            scene["start"] = update.start
            scene_start = update.start
        elif scene_start is not None and update.start < scene_start - 1e-3:
            raise HTTPException(
                status_code=422,
                detail=f"start {update.start} is before this scene's own start ({scene_start}) — only the scene's FIRST beat can start earlier than the scene",
            )

        if scene_end is not None and update.end > scene_end + 1e-3:
            raise HTTPException(status_code=422, detail=f"end {update.end} is after this scene's own end ({scene_end})")
        beat["start"] = update.start
        beat["end"] = update.end

    beats[beat_idx] = beat

    if update.start is not None and update.end is not None and update.adjust_next_beat and beat_idx + 1 < len(beats):
        next_beat = dict(beats[beat_idx + 1])
        next_beat["start"] = update.end
        beats[beat_idx + 1] = next_beat

    scene["beats"] = beats
    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1
    broll_list = _compute_broll_list(timeline_json)
    infographics_list, text_list = _compute_infographics_and_text_lists(raw_scenes, timeline_json)

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes, "timeline_json": timeline_json, "timeline_version": new_version,
            "final_video_url": None, "render_status": "stale_needs_render", "broll_list": broll_list,
            "infographics_list": infographics_list, "text_list": text_list,
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[update-scene-broll] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save broll selection")

    return {
        "video_id": video_id, "scene_id": scene_id, "beat_id": beat.get("beat_id"),
        "selected_asset": beat["broll_override"], "resolved_motion_type": _resolve_beat_motion_type(beat),
        "start": beat.get("start"), "end": beat.get("end"), "broll_list": broll_list,
        "infographics_list": infographics_list, "text_list": text_list,
        "timeline_version": new_version, "timeline": timeline_json, "needs_render": True,
    }


class AbsoluteBrollUpdate(BaseModel):
    asset_id: Any
    source: str
    start: float
    end: float
    motion_type: Optional[str] = None


def _carve_scene_beats_for_absolute_range(scene: dict, local_start: float, local_end: float, override: dict, id_prefix: str) -> list:
    beats = scene.get("beats") or []

    if not beats:
        media = scene.get("media") or {}
        beats = [{
            "beat_id": f"{id_prefix}_b1", "beat_index": 0, "start": scene.get("start"), "end": scene.get("end"),
            "vo_text": scene.get("vo_text", ""), "keywords": None, "motion_type": _DEFAULT_MOTION_TYPE,
            "preferred_media_type": None,
            "media": {
                "videos": media.get("videos") or {"total_results": 0, "results": [], "error": None},
                "images": media.get("images") or {"total_results": 0, "results": [], "error": None},
            },
        }]

    kept_fragments = []
    for b in beats:
        b_start, b_end = b.get("start"), b.get("end")
        if b_start is None or b_end is None:
            continue
        if b_end <= local_start + 1e-6 or b_start >= local_end - 1e-6:
            kept_fragments.append(b)
            continue
        if b_start < local_start - 1e-6:
            before = dict(b)
            before["beat_id"] = f"{b.get('beat_id')}_pre"
            before["end"] = local_start
            kept_fragments.append(before)
        if b_end > local_end + 1e-6:
            after = dict(b)
            after["beat_id"] = f"{b.get('beat_id')}_post"
            after["start"] = local_end
            kept_fragments.append(after)

    new_beat = {
        "beat_id": f"{id_prefix}_abs_{uuid.uuid4().hex[:6]}", "start": local_start, "end": local_end,
        "vo_text": "", "keywords": None, "preferred_media_type": override.get("source"),
        "motion_type": override.get("motion_type") or _DEFAULT_MOTION_TYPE,
        "media": {"videos": {"total_results": 0, "results": [], "error": None}, "images": {"total_results": 0, "results": [], "error": None}},
        "broll_override": override,
    }

    result = kept_fragments + [new_beat]
    result.sort(key=lambda b: b.get("start") if b.get("start") is not None else 0.0)
    for i, b in enumerate(result):
        b["beat_index"] = i

    return result




class BrollInsertGapUpdate(BaseModel):
    asset_id: Any
    source: str
    start: float
    end: float
    motion_type: Optional[str] = None

@app.patch("/timeline/{video_id}/broll")
async def update_broll_absolute(video_id: str, update: AbsoluteBrollUpdate):
    if update.source not in ("video", "image"):
        raise HTTPException(status_code=422, detail="source must be 'video' or 'image'")
    if update.start < 0:
        raise HTTPException(status_code=422, detail="start cannot be negative")
    if update.end <= update.start:
        raise HTTPException(status_code=422, detail="`end` must be greater than `start`")
    if update.motion_type is not None and update.motion_type not in _VALID_MOTION_TYPES:
        raise HTTPException(status_code=422, detail=f"motion_type must be one of {sorted(_VALID_MOTION_TYPES)}")

    candidate = await _fetch_pexels_asset_by_id(update.asset_id, update.source)
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Could not fetch {update.source} asset {update.asset_id} from Pexels")
    file_url = _resolve_broll_file_url(candidate, update.source)
    if not file_url:
        raise HTTPException(status_code=422, detail=f"Asset {update.asset_id} has no landscape/horizontal file available")

    override = {
        "asset_id": candidate.get("id"), "source": update.source, "file_url": file_url,
        "width": candidate.get("width"), "height": candidate.get("height"),
        "video_files": candidate.get("video_files"), "src": candidate.get("src"), "motion_type": update.motion_type,
    }

    try:
        row = supabase.table("videos").select("raw_scenes, timeline_json, timeline_version").eq("id", video_id).single().execute()
    except Exception as e:
        print(f"[update-broll-absolute] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    if not raw_scenes:
        raise HTTPException(status_code=400, detail="This video has no scenes yet")

    current_version = row.data.get("timeline_version", 1)
    fps = (row.data.get("timeline_json") or {}).get("fps", TIMELINE_FPS)

    def _compute_boundaries(scenes: list):
        boundaries, cumulative = [], 0.0
        for i, s in enumerate(scenes):
            s_start, s_end = s.get("start") or 0.0, s.get("end") or 0.0
            dur = max(s_end - s_start, 1.0 / fps)
            boundaries.append({"index": i, "abs_start": cumulative, "abs_end": cumulative + dur, "scene_start": s_start, "scene_end": s_end})
            cumulative += dur
        return boundaries, cumulative

    boundaries, total_duration = _compute_boundaries(raw_scenes)

    if update.start < boundaries[0]["abs_start"] - 1e-3:
        extend_by = boundaries[0]["abs_start"] - update.start
        first_scene = dict(raw_scenes[0])
        first_scene["start"] = max(0.0, (first_scene.get("start") or 0.0) - extend_by)
        print(f"[update-broll-absolute] extending video start to {update.start:.3f}s by pulling scene {first_scene.get('scene_id')}'s own start earlier")
        raw_scenes[0] = first_scene
        boundaries, total_duration = _compute_boundaries(raw_scenes)

    if update.end > total_duration + 1e-3:
        raise HTTPException(status_code=422, detail=f"end {update.end} is beyond the video's current total duration ({total_duration:.2f}s) — extending the END of the video isn't supported yet, only the start")

    touched = []
    for b in boundaries:
        if b["abs_end"] <= update.start + 1e-6 or b["abs_start"] >= update.end - 1e-6:
            continue
        local_start = max(update.start, b["abs_start"]) - b["abs_start"] + b["scene_start"]
        local_end = min(update.end, b["abs_end"]) - b["abs_start"] + b["scene_start"]
        scene = dict(raw_scenes[b["index"]])
        scene_id = scene.get("scene_id")
        scene["beats"] = _carve_scene_beats_for_absolute_range(scene, local_start, local_end, override, id_prefix=scene_id)
        raw_scenes[b["index"]] = scene
        touched.append(scene_id)

    if not touched:
        raise HTTPException(status_code=422, detail=f"[{update.start}, {update.end}] didn't overlap any scene in this video")

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1
    broll_list = _compute_broll_list(timeline_json)
    infographics_list, text_list = _compute_infographics_and_text_lists(raw_scenes, timeline_json)

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes, "timeline_json": timeline_json, "timeline_version": new_version,
            "final_video_url": None, "render_status": "stale_needs_render", "broll_list": broll_list,
            "infographics_list": infographics_list, "text_list": text_list,
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[update-broll-absolute] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save B-roll placement")

    return {
        "video_id": video_id, "start": update.start, "end": update.end, "selected_asset": override,
        "touched_scenes": touched, "broll_list": broll_list,
        "infographics_list": infographics_list, "text_list": text_list,
        "timeline_version": new_version, "timeline": timeline_json, "needs_render": True,
    }



@app.post("/timeline/{video_id}/broll/insert")
async def insert_broll_gap(video_id: str, update: BrollInsertGapUpdate):
    if update.source not in ("video", "image"):
        raise HTTPException(status_code=422, detail="source must be 'video' or 'image'")
    if update.start < 0:
        raise HTTPException(status_code=422, detail="start cannot be negative")
    if update.end <= update.start:
        raise HTTPException(status_code=422, detail="`end` must be greater than `start`")
    if update.motion_type is not None and update.motion_type not in _VALID_MOTION_TYPES:
        raise HTTPException(status_code=422, detail=f"motion_type must be one of {sorted(_VALID_MOTION_TYPES)}")

    duration = update.end - update.start

    candidate = await _fetch_pexels_asset_by_id(update.asset_id, update.source)
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Could not fetch {update.source} asset {update.asset_id} from Pexels")
    file_url = _resolve_broll_file_url(candidate, update.source)
    if not file_url:
        raise HTTPException(status_code=422, detail=f"Asset {update.asset_id} has no landscape/horizontal file available")

    override = {
        "asset_id": candidate.get("id"), "source": update.source, "file_url": file_url,
        "width": candidate.get("width"), "height": candidate.get("height"),
        "video_files": candidate.get("video_files"), "src": candidate.get("src"), "motion_type": update.motion_type,
    }

    try:
        row = supabase.table("videos").select("raw_scenes, timeline_json, timeline_version").eq("id", video_id).single().execute()
    except Exception as e:
        print(f"[insert-broll-gap] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    if not raw_scenes:
        raise HTTPException(status_code=400, detail="This video has no scenes yet")

    current_version = row.data.get("timeline_version", 1)
    fps = (row.data.get("timeline_json") or {}).get("fps", TIMELINE_FPS)

    boundaries, cumulative = [], 0.0
    for i, s in enumerate(raw_scenes):
        s_start, s_end = s.get("start") or 0.0, s.get("end") or 0.0
        dur = max(s_end - s_start, 1.0 / fps)
        boundaries.append({"index": i, "abs_start": cumulative, "abs_end": cumulative + dur})
        cumulative += dur
    total_duration = cumulative

    if update.start <= 0:
        insert_index, actual_at = 0, 0.0
    elif update.start >= total_duration:
        insert_index, actual_at = len(raw_scenes), total_duration
    else:
        insert_index, actual_at = None, None
        for b in boundaries:
            if abs(update.start - b["abs_start"]) < 1e-3:
                insert_index, actual_at = b["index"], b["abs_start"]
                break
            if abs(update.start - b["abs_end"]) < 1e-3:
                insert_index, actual_at = b["index"] + 1, b["abs_end"]
                break
            if b["abs_start"] < update.start < b["abs_end"]:
                insert_index, actual_at = b["index"] + 1, b["abs_end"]
                break
        if insert_index is None:
            raise HTTPException(status_code=500, detail="Could not resolve an insertion point")

    snapped = abs(actual_at - update.start) > 1e-3

    new_scene_id = f"s_ins_{uuid.uuid4().hex[:6]}"
    new_beat = {
        "beat_id": f"{new_scene_id}_b1", "beat_index": 0, "start": 0.0, "end": duration,
        "vo_text": "", "keywords": None, "preferred_media_type": update.source,
        "motion_type": update.motion_type or _DEFAULT_MOTION_TYPE,
        "media": {"videos": {"total_results": 0, "results": [], "error": None}, "images": {"total_results": 0, "results": [], "error": None}},
        "broll_override": override,
    }
    new_scene = {
        "scene_id": new_scene_id, "vo_text": "", "visual_intent": "", "on_screen_text": "",
        "start": 0.0, "end": duration, "duration_seconds": duration, "voiceover": None, "word_segments": [],
        "error": None, "beats": [new_beat], "media": new_beat["media"], "animations": [],
        "requires_animation": False, "scene_animation_density": "low",
    }

    raw_scenes = raw_scenes[:insert_index] + [new_scene] + raw_scenes[insert_index:]

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1
    infographics_list, text_list = _compute_infographics_and_text_lists(raw_scenes, timeline_json)

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes, "timeline_json": timeline_json, "timeline_version": new_version,
            "final_video_url": None, "render_status": "stale_needs_render",
            "infographics_list": infographics_list, "text_list": text_list,
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[insert-broll-gap] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to insert B-roll segment")

    return {
        "video_id": video_id, "new_scene_id": new_scene_id, "requested_start": update.start, "actual_start": actual_at,
        "snapped_to_scene_boundary": snapped, "duration": duration, "end": actual_at + duration,
        "new_total_duration": total_duration + duration, "selected_asset": override,
        "infographics_list": infographics_list, "text_list": text_list,
        "timeline_version": new_version, "timeline": timeline_json, "needs_render": True,
    }


    
@app.delete("/timeline/{video_id}/scene/{scene_id}/content")
async def delete_scene_content(
    video_id: str, scene_id: str,
    content_type: Literal["video", "image", "text", "infographics"],
    beat_id: Optional[str] = None,
):
    if beat_id is None:
        raise HTTPException(status_code=422, detail="beat_id is required — content is now beat-owned, not scene-owned")

    try:
        row = supabase.table("videos").select("raw_scenes, timeline_version").eq("id", video_id).single().execute()
    except Exception as e:
        print(f"[delete-scene-content] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])
    deleted_something = False

    if content_type in ("video", "image"):
        beats = scene.get("beats") or []
        beat_idx = next((i for i, b in enumerate(beats) if b.get("beat_id") == beat_id), None)
        if beat_idx is None:
            raise HTTPException(status_code=404, detail=f"Beat {beat_id} not found in scene {scene_id} (available: {[b.get('beat_id') for b in beats]})")
        beat = dict(beats[beat_idx])
        beat["broll_override"] = None
        beat["preferred_media_type"] = None
        beat["media"] = {"videos": {"total_results": 0, "results": [], "error": None}, "images": {"total_results": 0, "results": [], "error": None}}
        beats[beat_idx] = beat
        scene["beats"] = beats
        deleted_something = True

    elif content_type in ("text", "infographics"):
        animations = scene.get("animations") or []
        remaining = [a for a in animations if a.get("beat_id") != beat_id]
        if len(remaining) != len(animations):
            scene["animations"] = remaining
            deleted_something = True

    if not deleted_something:
        raise HTTPException(status_code=422, detail=f"Beat {beat_id} in scene {scene_id} has no {content_type} content to delete")

    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1

    infographics_list, text_list = _compute_infographics_and_text_lists(raw_scenes, timeline_json)
    broll_list = _compute_broll_list(timeline_json)

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes, "timeline_json": timeline_json, "timeline_version": new_version,
            "final_video_url": None, "render_status": "stale_needs_render",
            "infographics_list": infographics_list, "text_list": text_list, "broll_list": broll_list,
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[delete-scene-content] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete content")

    return {
        "video_id": video_id, "scene_id": scene_id, "content_type": content_type, "beat_id": beat_id,
        "infographics_list": infographics_list, "text_list": text_list, "broll_list": broll_list,
        "timeline_version": new_version, "timeline": timeline_json, "needs_render": True,
    }





















from fastapi import HTTPException, BackgroundTasks, Response, status


RENDER_TMP_ROOT = os.getenv("RENDER_TMP_ROOT", "/tmp/storybit-render")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

REMOTION_PROJECT_DIR = os.getenv("REMOTION_PROJECT_DIR", "")

RENDER_CONCURRENCY = int(os.getenv("RENDER_CONCURRENCY", str(max(os.cpu_count() or 2, 2))))

FFMPEG_X264_PRESET = os.getenv("FFMPEG_X264_PRESET", "veryfast")
FFMPEG_X264_CRF = os.getenv("FFMPEG_X264_CRF", "23")
FFMPEG_X264_FLAGS = [
    "-c:v", "libx264",
    "-preset", FFMPEG_X264_PRESET,
    "-crf", FFMPEG_X264_CRF,
    "-pix_fmt", "yuv420p",
    "-threads", "0",
]

SILENT_AUDIO_SAMPLE_RATE = int(os.getenv("SILENT_AUDIO_SAMPLE_RATE", "48000"))
SILENT_AUDIO_CHANNEL_LAYOUT = os.getenv("SILENT_AUDIO_CHANNEL_LAYOUT", "stereo")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_OUTPUT_DIR = os.environ.get("RENDER_OUTPUT_DIR", os.path.join(BASE_DIR, "rendered_videos"))
os.makedirs(RENDER_OUTPUT_DIR, exist_ok=True)

LANDSCAPE_RESOLUTION = {"width": 1920, "height": 1080}
PORTRAIT_RESOLUTION = {"width": 1080, "height": 1920}

SUPABASE_RENDERED_VIDEOS_BUCKET = os.getenv("SUPABASE_RENDERED_VIDEOS_BUCKET", "rendered-videos")

_REMOTION_COMPOSITION_BY_ANIMATION_TYPE = {
    # full_screen
    "full_screen_title_card": "TitleCard",
    "full_screen_quote_card": "QuoteCard",
    "full_screen_data_viz": "DataVizFullScreen",
    "full_screen_broll": "FullScreenBroll",  # no-op by design — see compositions.tsx
    "full_screen_transition": "FullScreenTransitionFx",
    "full_screen_color_wash": "FullScreenColorWash",
    "full_screen_document_highlight": "FullScreenDocumentHighlight",  # degraded, no real screenshot yet
    # overlay_text
    "stat_counter_overlay": "StatCounterOverlay",
    "bullet_list_reveal": "BulletListReveal",
    "lower_third": "LowerThird",
    "kinetic_caption": "KineticCaption",
    "callout_textbox": "CalloutTextbox",
    # overlay_graphic
    "icon_sequence": "IconSequenceOverlay",
    "icon_pop_in": "IconPopIn",
    "logo_watermark": "LogoWatermark",
    "emoji_reaction": "EmojiReaction",
    "arrow_highlight": "ArrowHighlight",
    "badge_sticker": "BadgeSticker",
    # pip — chrome/frame only, no second video source in the schema yet
    "pip_video": "PipVideoFrame",
    "split_screen": "SplitScreenDivider",
    "multi_panel_grid": "MultiPanelGrid",
    # branding — icon placeholder only, no avatar/mascot asset in the schema yet
    "avatar_overlay": "AvatarOverlayPlaceholder",
    "mascot_animation": "MascotAnimationPlaceholder",
    # transition — accents/no-ops only; true shake/speed/parallax need an
    # FFmpeg pass on the beat clip itself, not a Remotion overlay (see
    # compositions.tsx for why)
    "ken_burns_pan_zoom": "KenBurnsNoOp",
    "parallax_layering": "ParallaxAccent",
    "shake_impact": "ShakeImpactFlash",
    "speed_ramp_indicator": "SpeedRampIndicator",
}

# Concurrent scene renders can each spawn their own `npx remotion render`
# (and therefore its own headless Chrome instance). Uncapped, that's a
# plausible source of the "Was not able to close puppeteer page /
# No target found for targetId" warnings under RENDER_CONCURRENCY > a
# couple — that error was cosmetic (it fired during cleanup of an
# already-failed render), but capping concurrent Remotion invocations
# is worth doing regardless of the composition-id fix below.
REMOTION_MAX_CONCURRENT = int(os.getenv("REMOTION_MAX_CONCURRENT", "2"))
_remotion_semaphore = asyncio.Semaphore(REMOTION_MAX_CONCURRENT)


# =============================================================================
# IMAGE MOTION EFFECTS  (unchanged — already per-beat via `motion_type` on
# each beat's broll timeline track, resolved by `_resolve_beat_motion_type`
# upstream. Nothing here changes with the animations refactor.)
# =============================================================================

def _ease_expr(duration_frames: int) -> str:
    d = max(duration_frames - 1, 1)
    return f"(1-cos(PI*min(on/{d},1)))/2"


def _build_image_motion_filter(
    motion_type: str, duration_frames: int, fps: int, width: int, height: int,
) -> str:
    ease = _ease_expr(duration_frames)
    zoom_base = 1.2
    zoom_amount = 0.3

    if motion_type == "zoom_in":
        z_expr = f"1+{zoom_amount}*({ease})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif motion_type == "zoom_out":
        z_expr = f"{1 + zoom_amount}-{zoom_amount}*({ease})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif motion_type == "pan_left":
        z_expr = f"{zoom_base}"
        x_expr = f"({ease})*(iw-iw/zoom)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif motion_type == "pan_right":
        z_expr = f"{zoom_base}"
        x_expr = f"(1-({ease}))*(iw-iw/zoom)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif motion_type == "tilt_up":
        z_expr = f"{zoom_base}"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = f"(1-({ease}))*(ih-ih/zoom)"
    elif motion_type == "tilt_down":
        z_expr = f"{zoom_base}"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = f"({ease})*(ih-ih/zoom)"
    else:
        print(f"[render] unrecognized motion_type {motion_type!r} — using a subtle default zoom")
        z_expr = f"1+0.05*({ease})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    return (
        f"scale=8000:-1,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
        f"d={duration_frames}:s={width}x{height}:fps={fps}"
    )


def _upload_rendered_video_to_supabase(local_path: str, video_id: str, filename: Optional[str] = None) -> str:
    if filename is None:
        filename = f"final_{uuid.uuid4().hex}.mp4"
    dest_path = f"{video_id}/{filename}"

    with open(local_path, "rb") as f:
        file_bytes = f.read()

    supabase.storage.from_(SUPABASE_RENDERED_VIDEOS_BUCKET).upload(
        path=dest_path,
        file=file_bytes,
        file_options={"content-type": "video/mp4", "upsert": "true"},
    )

    return supabase.storage.from_(SUPABASE_RENDERED_VIDEOS_BUCKET).get_public_url(dest_path)


class RenderVideoRequest(BaseModel):
    force: bool = False
    orientation: Literal["landscape", "portrait"] = "landscape"


RUN_SUBPROCESS_TIMEOUT_SECONDS = int(os.getenv("RUN_SUBPROCESS_TIMEOUT_SECONDS", "300"))


async def _run(cmd: list[str], cwd: Optional[str] = None, timeout: Optional[float] = None) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout or RUN_SUBPROCESS_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(
            f"Command timed out after {timeout or RUN_SUBPROCESS_TIMEOUT_SECONDS}s "
            f"and was killed: {' '.join(cmd)}"
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stderr ---\n{stderr.decode(errors='replace')[-4000:]}"
        )


async def _probe_duration_seconds(path: str) -> float:
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {stderr.decode(errors='replace')}")
    try:
        return float(stdout.decode().strip())
    except ValueError:
        raise RuntimeError(f"ffprobe returned unparsable duration for {path}: {stdout!r}")


async def _probe_video_dimensions(path: str) -> Optional[tuple[int, int]]:
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
        "-of", "json",
        path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        print(f"[render] ffprobe dimension check failed on {path}: {stderr.decode(errors='replace')[-500:]}")
        return None

    try:
        data = json.loads(stdout.decode())
        streams = data.get("streams") or []
        if not streams:
            return None
        stream = streams[0]

        width = int(stream.get("width", 0) or 0)
        height = int(stream.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            return None

        rotation = 0
        tags_rotate = (stream.get("tags") or {}).get("rotate")
        if tags_rotate is not None:
            try:
                rotation = int(float(tags_rotate))
            except (TypeError, ValueError):
                rotation = 0

        for sd in (stream.get("side_data_list") or []):
            if "rotation" in sd:
                try:
                    rotation = int(float(sd["rotation"]))
                except (TypeError, ValueError):
                    pass

        rotation = abs(rotation) % 360
        if rotation in (90, 270):
            width, height = height, width

        return width, height
    except (ValueError, KeyError, IndexError, json.JSONDecodeError):
        return None


async def _download(url: str, dest: str, client: httpx.AsyncClient) -> str:
    resp = await client.get(url, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    return dest


async def _make_color_fallback(
    duration_frames: int, fps: int, width: int, height: int, tmp_dir: str,
    color: str = "0x111827",
) -> str:
    out_path = os.path.join(tmp_dir, "broll_fallback.mp4")
    seconds = duration_frames / fps
    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "lavfi",
        "-i", f"color=c={color}:s={width}x{height}:d={seconds:.3f}:r={fps}",
        *FFMPEG_X264_FLAGS,
        out_path,
    ]
    await _run(cmd)
    return out_path


async def _make_blurpad_fallback(
    local_src: str, source_kind: str, duration_frames: int, fps: int,
    width: int, height: int, tmp_dir: str,
) -> Optional[str]:
    target_seconds = duration_frames / fps
    out_path = os.path.join(tmp_dir, "broll_blurpad.mp4")

    filter_complex = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma=30,eq=brightness=-0.05[bg];"
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps}[outv]"
    )

    if source_kind == "image":
        cmd = [
            FFMPEG_BIN, "-y",
            "-loop", "1", "-i", local_src,
            "-t", f"{target_seconds:.3f}",
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-an",
            *FFMPEG_X264_FLAGS,
            out_path,
        ]
        try:
            await _run(cmd)
            return out_path
        except Exception as e:
            print(f"[render] blur-pad image composite failed: {e}")
            return None

    try:
        source_seconds = await _probe_duration_seconds(local_src)
    except Exception:
        source_seconds = target_seconds

    if source_seconds >= target_seconds:
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", local_src,
            "-t", f"{target_seconds:.3f}",
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-an",
            *FFMPEG_X264_FLAGS,
            out_path,
        ]
    else:
        loops_needed = int(target_seconds // max(source_seconds, 0.1)) + 1
        cmd = [
            FFMPEG_BIN, "-y",
            "-stream_loop", str(loops_needed),
            "-i", local_src,
            "-t", f"{target_seconds:.3f}",
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-an",
            *FFMPEG_X264_FLAGS,
            out_path,
        ]

    try:
        await _run(cmd)
        return out_path
    except Exception as e:
        print(f"[render] blur-pad video composite failed: {e}")
        return None


async def _prepare_broll_clip(
    broll_track: dict, scene_duration_frames: int, fps: int, width: int, height: int,
    tmp_dir: str, client: httpx.AsyncClient,
    background_color_override: Optional[str] = None,
    motion_type: Optional[str] = None,
) -> str:
    if background_color_override:
        return await _make_color_fallback(
            scene_duration_frames, fps, width, height, tmp_dir,
            color=_normalize_ffmpeg_color(background_color_override),
        )

    selected = broll_track.get("selected_asset")

    if not selected:
        print("[render] no b-roll asset selected for this beat — using color fallback")
        return await _make_color_fallback(scene_duration_frames, fps, width, height, tmp_dir)

    source = selected.get("source", "video")
    landscape_url = _resolve_broll_file_url(selected, source)

    if landscape_url:
        target_seconds = scene_duration_frames / fps
        ext = ".mp4" if source == "video" else ".jpg"
        local_src = os.path.join(tmp_dir, f"broll_src{ext}")

        try:
            await _download(landscape_url, local_src, client)
        except Exception as e:
            print(f"[render] broll download failed ({landscape_url}): {e} — trying blur-pad fallback")
            landscape_url = None

        if landscape_url:
            if source == "video":
                dims = await _probe_video_dimensions(local_src)
                if dims is not None and not _is_landscape_dimensions(dims[0], dims[1]):
                    print(
                        f"[render] downloaded broll video DISPLAYS as portrait/square "
                        f"(effective {dims[0]}x{dims[1]} after rotation) despite metadata "
                        f"claiming landscape — using blur-pad composite instead of stretching it"
                    )
                    blurpad = await _make_blurpad_fallback(
                        local_src, "video", scene_duration_frames, fps, width, height, tmp_dir
                    )
                    if blurpad:
                        return blurpad
                    return await _make_color_fallback(scene_duration_frames, fps, width, height, tmp_dir)

            out_path = os.path.join(tmp_dir, "broll_fit.mp4")

            if source == "image":
                resolved_motion = motion_type if motion_type in _VALID_MOTION_TYPES else _DEFAULT_MOTION_TYPE
                motion_filter = _build_image_motion_filter(
                    resolved_motion, duration_frames=scene_duration_frames,
                    fps=fps, width=width, height=height,
                )
                cmd = [
                    FFMPEG_BIN, "-y",
                    "-loop", "1", "-i", local_src,
                    "-t", f"{target_seconds:.3f}",
                    "-vf", motion_filter,
                    "-an",
                    *FFMPEG_X264_FLAGS,
                    out_path,
                ]
                await _run(cmd)
                return out_path

            scale_crop = (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},setsar=1,fps={fps}"
            )

            source_seconds = await _probe_duration_seconds(local_src)
            if source_seconds >= target_seconds:
                cmd = [
                    FFMPEG_BIN, "-y",
                    "-i", local_src,
                    "-t", f"{target_seconds:.3f}",
                    "-vf", scale_crop, "-an",
                    *FFMPEG_X264_FLAGS,
                    out_path,
                ]
            else:
                loops_needed = int(target_seconds // source_seconds) + 1
                cmd = [
                    FFMPEG_BIN, "-y",
                    "-stream_loop", str(loops_needed),
                    "-i", local_src,
                    "-t", f"{target_seconds:.3f}",
                    "-vf", scale_crop, "-an",
                    *FFMPEG_X264_FLAGS,
                    out_path,
                ]
            await _run(cmd)
            return out_path

    any_url = _resolve_broll_file_url_any_orientation(selected, source)
    if not any_url:
        print(
            f"[render] no downloadable file at all for asset "
            f"{selected.get('asset_id') or selected.get('id')} — using color fallback"
        )
        return await _make_color_fallback(scene_duration_frames, fps, width, height, tmp_dir)

    ext = ".mp4" if source == "video" else ".jpg"
    local_src = os.path.join(tmp_dir, f"broll_anyorient{ext}")
    try:
        await _download(any_url, local_src, client)
    except Exception as e:
        print(f"[render] any-orientation broll download failed ({any_url}): {e} — using color fallback")
        return await _make_color_fallback(scene_duration_frames, fps, width, height, tmp_dir)

    blurpad = await _make_blurpad_fallback(local_src, source, scene_duration_frames, fps, width, height, tmp_dir)
    if blurpad:
        return blurpad

    print("[render] blur-pad composite failed — using color fallback as last resort")
    return await _make_color_fallback(scene_duration_frames, fps, width, height, tmp_dir)


async def _make_silent_audio(duration_frames: int, fps: int, tmp_dir: str) -> str:
    out_path = os.path.join(tmp_dir, "silence.aac")
    seconds = max(duration_frames / fps, 1 / fps)
    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout={SILENT_AUDIO_CHANNEL_LAYOUT}:sample_rate={SILENT_AUDIO_SAMPLE_RATE}",
        "-t", f"{seconds:.3f}",
        "-c:a", "aac",
        out_path,
    ]
    await _run(cmd)
    return out_path


async def render_infographic_via_remotion(
    composition_id: str, props: dict, duration_frames: int, fps: int,
    width: int, height: int, tmp_dir: str,
) -> Optional[str]:
    """Renders one animation as a transparent ProRes4444 clip via Remotion.
    `composition_id` is the animation_type (e.g. "icon_pop_in",
    "full_screen_quote_card") — the Remotion project is expected to expose
    one composition per animation_type in ANIMATION_TAXONOMY."""
    if not REMOTION_PROJECT_DIR:
        print(
            f"[render] REMOTION_PROJECT_DIR not configured — skipping "
            f"animation '{composition_id}'. Set up a Remotion project and "
            f"point REMOTION_PROJECT_DIR at it to enable overlays."
        )
        return None

    out_path = os.path.join(tmp_dir, f"anim_{uuid.uuid4().hex}.mov")
    props_path = os.path.join(tmp_dir, f"props_{uuid.uuid4().hex}.json")
    with open(props_path, "w") as f:
        json.dump(props, f)

    def _build_cmd(frame_range: Optional[str]) -> list[str]:
        cmd = ["npx", "remotion", "render", composition_id, out_path, f"--props={props_path}"]
        if frame_range:
            cmd.append(f"--frames={frame_range}")
        cmd += [
            f"--fps={fps}", f"--width={width}", f"--height={height}",
            "--codec=prores", "--prores-profile=4444", "--pixel-format=yuva444p10le",
        ]
        return cmd

    async with _remotion_semaphore:
        try:
            await _run(_build_cmd(f"0-{duration_frames - 1}"), cwd=REMOTION_PROJECT_DIR)
        except RuntimeError as e:
            msg = str(e)
            m = re.search(
                r"durationInFrames.*?evaluated to be (\d+).*?not inbetween 0-(\d+)",
                msg, re.IGNORECASE | re.DOTALL,
            )
            if not m:
                print(f"[render] Remotion render failed for '{composition_id}': {e}")
                return None

            max_frame = int(m.group(2))
            print(
                f"[render] '{composition_id}' has a fixed durationInFrames "
                f"— retrying within its native bounds (0-{max_frame})"
            )
            try:
                await _run(_build_cmd(f"0-{max_frame}"), cwd=REMOTION_PROJECT_DIR)
            except Exception as e2:
                print(f"[render] retry also failed for '{composition_id}': {e2}")
                return None

    return out_path


def _frames_to_ass_time(frame: int, fps: int) -> str:
    total_seconds = frame / fps
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = total_seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _build_ass_from_words(
    words: list[dict], scene_start_frame: int, fps: int, width: int, height: int,
    style: Optional[dict] = None,
) -> str:
    style = style or {}
    font_size = style.get("font_size") or 72
    primary_color = _hex_to_ass_color(style.get("text_color") or "#FFFFFF", alpha_hex="00")
    outline_color = _hex_to_ass_color(style.get("outline_color") or "#000000", alpha_hex="00")
    animation_type = style.get("animation_type") or "kinetic_caption"

    vertical_position = (style.get("vertical_position") or "bottom").lower()
    horizontal_position = (style.get("horizontal_position") or "center").lower()
    try:
        margin_v_percent = float(style.get("margin_bottom_percent", 3))
    except (TypeError, ValueError):
        margin_v_percent = 3.0
    try:
        margin_h_percent = float(style.get("margin_horizontal_percent", 0))
    except (TypeError, ValueError):
        margin_h_percent = 0.0

    _ALIGNMENT_GRID = {
        ("top", "left"): 7, ("top", "center"): 8, ("top", "right"): 9,
        ("middle", "left"): 4, ("middle", "center"): 5, ("middle", "right"): 6,
        ("bottom", "left"): 1, ("bottom", "center"): 2, ("bottom", "right"): 3,
    }
    alignment = _ALIGNMENT_GRID.get((vertical_position, horizontal_position), 2)

    margin_v = 0 if vertical_position == "middle" else max(round(height * margin_v_percent / 100), 0)
    default_side_margin = 60
    margin_h = (
        default_side_margin if margin_h_percent <= 0
        else max(round(width * margin_h_percent / 100), 0)
    )
    margin_l = margin_h
    margin_r = margin_h

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Caption,Arial Black,{font_size},{primary_color},{outline_color},&H80000000,1,4,0,{alignment},{margin_l},{margin_r},{margin_v}

[Events]
Format: Layer, Start, End, Style, Text
"""
    if animation_type == "static_line" and words:
        start_frame = words[0]["startFrame"] - scene_start_frame
        end_frame = words[-1]["endFrame"] - scene_start_frame
        text = " ".join(w["word"] for w in words)
        if start_frame < 0:
            start_frame = 0
        if end_frame > start_frame:
            start_ts = _frames_to_ass_time(start_frame, fps)
            end_ts = _frames_to_ass_time(end_frame, fps)
            return header + f"Dialogue: 0,{start_ts},{end_ts},Caption,{text}\n"
        return header

    lines = []
    chunk: list[dict] = []
    CHUNK_SIZE = 4
    for w in words:
        chunk.append(w)
        if len(chunk) >= CHUNK_SIZE:
            lines.append(chunk)
            chunk = []
    if chunk:
        lines.append(chunk)

    events = []
    for line_words in lines:
        start_frame = line_words[0]["startFrame"] - scene_start_frame
        end_frame = line_words[-1]["endFrame"] - scene_start_frame
        if start_frame < 0 or end_frame <= start_frame:
            continue
        text = " ".join(w["word"] for w in line_words)
        start_ts = _frames_to_ass_time(start_frame, fps)
        end_ts = _frames_to_ass_time(end_frame, fps)
        events.append(f"Dialogue: 0,{start_ts},{end_ts},Caption,{text}")

    return header + "\n".join(events) + "\n"


async def _burn_captions(
    input_path: str, words: list[dict], scene_start_frame: int, fps: int,
    width: int, height: int, tmp_dir: str, style: Optional[dict] = None,
) -> str:
    if not words:
        return input_path

    ass_content = _build_ass_from_words(words, scene_start_frame, fps, width, height, style=style)
    ass_path = os.path.join(tmp_dir, "captions.ass")
    with open(ass_path, "w") as f:
        f.write(ass_content)

    out_path = os.path.join(tmp_dir, "with_captions.mp4")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        "-vf", "ass=captions.ass",
        *FFMPEG_X264_FLAGS,
        "-c:a", "copy",
        out_path,
    ]
    await _run(cmd, cwd=tmp_dir)
    return out_path


def _escape_drawtext(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\u2019")
        .replace("%", "\\%")
        .replace("\n", "\\n")
    )


# =============================================================================
# BEAT ANIMATION RENDERING  (rewritten for the new per-beat animations
# schema — see the module docstring at the top of this file for the full
# rationale.)
# =============================================================================

def _scale_geometry_px(geometry_px: dict, out_width: int, out_height: int) -> dict:
    """
    geometry_px is always expressed on a fixed 1920x1080 (16:9) canvas
    (ANIMATION_CANVAS_WIDTH/HEIGHT, from the edit-video module). Scales
    uniformly by width (contain-fit) and centers vertically when the
    actual output is a different aspect ratio (e.g. 1080x1920 portrait) —
    this keeps proportions correct instead of stretching text/icons.
    """
    geometry_px = geometry_px or {"x": 0, "y": 0, "width": 200, "height": 100}
    scale = out_width / ANIMATION_CANVAS_WIDTH
    scaled_canvas_height = ANIMATION_CANVAS_HEIGHT * scale
    y_offset = (out_height - scaled_canvas_height) / 2

    return {
        "x": geometry_px.get("x", 0) * scale,
        "y": geometry_px.get("y", 0) * scale + y_offset,
        "width": geometry_px.get("width", 0) * scale,
        "height": geometry_px.get("height", 0) * scale,
        "scale": scale,
    }


def _display_text_to_string(display_text: Any) -> str:
    if isinstance(display_text, list):
        return "\n".join(str(d) for d in display_text if d)
    if isinstance(display_text, str):
        return display_text
    return ""


def _font_size_for_geometry(geo: dict, text: str) -> int:
    """Loose implementation of the Animation Planner's own sizing rule of
    thumb (~14-18 chars per 100px of box width) — solves for a font size
    that keeps the longest line inside the (already-scaled) box."""
    lines = text.split("\n") or [text]
    longest = max((len(l) for l in lines), default=1) or 1
    chars_per_100px = 16
    width_based = max(int((geo["width"] / max(longest, 1)) * (100 / chars_per_100px) * 0.6), 18)
    height_based = int(geo["height"] / max(len(lines), 1) * 0.7) if geo["height"] else width_based
    return max(18, min(width_based, height_based, 120))

def _build_icon_overlay_drawtext(
    animation: dict, out_width: int, out_height: int, clip_duration_frames: int, fps: int,
) -> Optional[str]:
    """FFmpeg fallback for overlay_graphic/branding animations — draws the
    icon's emoji glyph (+ optional label) at geometry_px, with fade in/out
    and support for single / sequence / cluster / pair icon_layout."""
    icon_name = animation.get("icon_name")
    icons: list[str] = []
    if isinstance(icon_name, str) and icon_name:
        icons = [icon_name]
    elif isinstance(icon_name, list):
        icons = [i for i in icon_name if isinstance(i, str) and i]

    label = _display_text_to_string(animation.get("display_text"))
    if not icons and not label:
        return None

    geo = _scale_geometry_px(animation.get("geometry_px") or {}, out_width, out_height)

    duration_seconds = max(clip_duration_frames / fps, 0.2)
    fade_in = min(0.35, duration_seconds / 4)
    fade_out = min(0.35, duration_seconds / 4)
    alpha_expr = (
        f"if(lt(t,{fade_in}),t/{fade_in},"
        f"if(gt(t,{duration_seconds - fade_out}),({duration_seconds}-t)/{fade_out},1))"
    )

    icon_font_size = max(28, min(int(geo["height"] * 0.55), 140))
    layout = animation.get("icon_layout")
    filters = []

    if len(icons) <= 1:
        glyph = _icon_glyph(icons[0]) if icons else ""
        text = f"{glyph}  {label}".strip() if (glyph and label) else (glyph or label)
        safe_text = _escape_drawtext(text)
        cx = geo["x"] + geo["width"] / 2
        cy = geo["y"] + geo["height"] / 2
        filters.append(
            f"drawtext=text='{safe_text}':fontcolor=white:fontsize={icon_font_size}:"
            f"box=1:boxcolor=black@0.5:boxborderw=14:"
            f"x={cx:.1f}-text_w/2:y={cy:.1f}-text_h/2:alpha='{alpha_expr}'"
        )
    else:
        n = min(len(icons), 4)
        if layout == "sequence":
            slice_seconds = duration_seconds / n
            x0 = geo["x"] + geo["width"] / 2
            y0 = geo["y"] + geo["height"] / 2
            for i, icon in enumerate(icons[:n]):
                glyph = _icon_glyph(icon)
                safe_text = _escape_drawtext(glyph)
                seg_start = i * slice_seconds
                seg_end = duration_seconds if i == n - 1 else (i + 1) * slice_seconds
                local_fade_in = min(0.2, slice_seconds / 4)
                seg_alpha = (
                    f"if(lt(t,{seg_start}),0,"
                    f"if(lt(t,{seg_start + local_fade_in}),(t-{seg_start})/{local_fade_in},"
                    f"if(lt(t,{seg_end}),1,0)))"
                )
                filters.append(
                    f"drawtext=text='{safe_text}':fontcolor=white:fontsize={icon_font_size}:"
                    f"box=1:boxcolor=black@0.5:boxborderw=14:"
                    f"x={x0:.1f}-text_w/2:y={y0:.1f}-text_h/2:alpha='{seg_alpha}'"
                )
        else:
            spacing = geo["width"] / n
            y0 = geo["y"] + geo["height"] / 2
            for i, icon in enumerate(icons[:n]):
                glyph = _icon_glyph(icon)
                safe_text = _escape_drawtext(glyph)
                cx = geo["x"] + spacing * (i + 0.5)
                filters.append(
                    f"drawtext=text='{safe_text}':fontcolor=white:fontsize={icon_font_size}:"
                    f"box=1:boxcolor=black@0.5:boxborderw=10:"
                    f"x={cx:.1f}-text_w/2:y={y0:.1f}-text_h/2:alpha='{alpha_expr}'"
                )
        if label:
            safe_label = _escape_drawtext(label)
            label_font_size = max(20, min(int(geo["height"] * 0.28), 56))
            lx = geo["x"] + geo["width"] / 2
            ly = geo["y"] + geo["height"] * 0.82
            filters.append(
                f"drawtext=text='{safe_label}':fontcolor=white:fontsize={label_font_size}:"
                f"box=1:boxcolor=black@0.5:boxborderw=10:"
                f"x={lx:.1f}-text_w/2:y={ly:.1f}-text_h/2:alpha='{alpha_expr}'"
            )

    return ",".join(filters)


    
def _build_beat_animation_drawtext(
    animation: dict, out_width: int, out_height: int, clip_duration_frames: int, fps: int,
) -> Optional[str]:
    """
    FFmpeg fallback for animation categories that are pure text —
    `overlay_text`, and `full_screen` animations that carry `display_text`
    or `highlight_target_text` with nothing to actually screenshot.
    `overlay_graphic` / `pip` / `branding` (icons, PiP frames, avatars)
    have no honest FFmpeg equivalent — callers should not route those here.
    """
    text = _display_text_to_string(animation.get("display_text"))
    if not text and animation.get("highlight_target_text"):
        # No real screenshot asset to composite — degrade to showing the
        # quoted text itself rather than dropping the beat's emphasis.
        text = animation["highlight_target_text"]
    if not text:
        return None

    geo = _scale_geometry_px(animation.get("geometry_px") or {}, out_width, out_height)
    font_size = _font_size_for_geometry(geo, text)

    duration_seconds = max(clip_duration_frames / fps, 0.2)
    fade_in = min(0.4, duration_seconds / 4)
    fade_out = min(0.4, duration_seconds / 4)
    alpha_expr = (
        f"if(lt(t,{fade_in}),t/{fade_in},"
        f"if(gt(t,{duration_seconds - fade_out}),({duration_seconds}-t)/{fade_out},1))"
    )

    safe_text = _escape_drawtext(text)
    x = f"{geo['x']:.1f}"
    y = f"{geo['y']:.1f}"

    return (
        f"drawtext=text='{safe_text}':fontcolor=white:fontsize={font_size}:"
        f"box=1:boxcolor=black@0.55:boxborderw=16:line_spacing=8:"
        f"x={x}:y={y}:alpha='{alpha_expr}'"
    )


def _build_remotion_props(animation_type: str, animation: dict, width: int, height: int) -> dict:
    """
    Your first 7 compositions (TitleCard, QuoteCard, DataVizFullScreen,
    StatCounterOverlay, BulletListReveal, IconSequenceOverlay, IconPopIn)
    each expect their OWN specific field names (title/subtitle,
    quote/attribution, label/caption, value/label, items, icons) — not
    the generic displayText/iconName/colorHint shape the animation track
    carries. Sending the generic shape to those meant they were either
    rendering with their defaultProps (ignoring your beat's actual text)
    or failing zod validation outright. This maps animation_type to the
    right prop shape per composition.

    Every OTHER composition (everything in compositions/Extra.tsx) shares
    one generic schema (baseAnimSchema) and accepts the raw shape
    directly, so those fall through to the generic branch unchanged.
    """
    text = _display_text_to_string(animation.get("display_text"))
    lines = [l for l in text.split("\n") if l.strip()] if text else []

    icons = animation.get("icon_name")
    if isinstance(icons, str):
        icons = [icons]
    elif not isinstance(icons, list):
        icons = []

    if animation_type == "full_screen_title_card":
        return {"title": lines[0] if lines else text, "subtitle": lines[1] if len(lines) > 1 else ""}

    if animation_type == "full_screen_quote_card":
        quote = lines[0] if lines else (animation.get("highlight_target_text") or text)
        return {"quote": quote, "attribution": lines[1] if len(lines) > 1 else ""}

    if animation_type == "full_screen_data_viz":
        return {"label": lines[0] if lines else text, "caption": lines[1] if len(lines) > 1 else ""}

    if animation_type == "stat_counter_overlay":
        return {"value": lines[0] if lines else text, "label": lines[1] if len(lines) > 1 else ""}

    if animation_type == "bullet_list_reveal":
        items = lines if lines else ([text] if text else ["", ""])
        return {"title": "", "items": items[:6] or ["", ""]}

    if animation_type == "icon_sequence":
        return {"icons": icons or ["sparkles"], "label": text}

    if animation_type == "icon_pop_in":
        return {"icons": (icons[:1] or ["sparkles"]), "label": text}

    # Everything else (compositions/Extra.tsx) — generic shape.
    return {
        "displayText": animation.get("display_text"),
        "colorHint": animation.get("color_hint"),
        "geometryPx": _scale_geometry_px(animation.get("geometry_px") or {}, width, height),
        "iconName": animation.get("icon_name"),
        "iconLayout": animation.get("icon_layout"),
        "highlightTargetText": animation.get("highlight_target_text"),
        "motion": animation.get("motion"),
    }


async def _apply_beat_animation(
    beat_clip_path: str, animation: dict, width: int, height: int, fps: int, tmp_dir: str,
) -> str:
    """
    Applies ONE animation (a timeline "animation" track dict, already
    scoped to a single beat) onto that beat's own clip. Returns the
    animated clip path, or the original `beat_clip_path` unchanged if
    nothing could be rendered for it.
    """
    category = animation.get("category")
    animation_type = animation.get("animation_type")
    duration_frames = animation.get("duration_frames") or fps * 2

    # A separate async Remotion pipeline may already have pre-rendered
    # this animation and populated `asset_url` on the timeline track —
    # use it directly instead of invoking Remotion synchronously again.
    asset_url = animation.get("asset_url")
    overlay_clip = None

    if asset_url:
        try:
            ext = ".mov" if asset_url.lower().endswith((".mov", ".mp4")) else ".png"
            local_asset = os.path.join(tmp_dir, f"preRendered_{uuid.uuid4().hex}{ext}")
            async with httpx.AsyncClient() as client:
                await _download(asset_url, local_asset, client)
            overlay_clip = local_asset
        except Exception as e:
            print(f"[render] failed to fetch pre-rendered asset for '{animation_type}': {e} — falling back")

    composition_id = _REMOTION_COMPOSITION_BY_ANIMATION_TYPE.get(animation_type)

    if overlay_clip is None and animation.get("render_engine_hint") == "remotion" and REMOTION_PROJECT_DIR:
        if composition_id:
            props = _build_remotion_props(animation_type, animation, width, height)
            overlay_clip = await render_infographic_via_remotion(
                composition_id=composition_id, props=props,
                duration_frames=duration_frames, fps=fps, width=width, height=height, tmp_dir=tmp_dir,
            )
        else:
            # No composition built for this animation_type yet — don't
            # even spawn `npx remotion render`, it can only fail. Fall
            # straight through to the text/skip logic below.
            print(
                f"[render] no Remotion composition mapped for animation_type "
                f"'{animation_type}' yet (see _REMOTION_COMPOSITION_BY_ANIMATION_TYPE) "
                f"— trying the FFmpeg text fallback instead"
            )

    if overlay_clip:
        composited = os.path.join(tmp_dir, f"anim_composited_{uuid.uuid4().hex}.mp4")
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", beat_clip_path,
            "-i", overlay_clip,
            "-filter_complex", "[1:v]format=yuva420p[fg];[0:v][fg]overlay=0:0:eof_action=pass:format=auto",
            "-r", str(fps),
            *FFMPEG_X264_FLAGS,
            "-an",
            composited,
        ]
        try:
            await _run(cmd)
            return composited
        except Exception as e:
            print(f"[render] compositing rendered animation '{animation_type}' failed: {e} — falling back")

    text_eligible = category == "overlay_text" or (
        category == "full_screen" and (animation.get("display_text") or animation.get("highlight_target_text"))
    )
    if text_eligible:
        drawtext = _build_beat_animation_drawtext(animation, width, height, duration_frames, fps)
        if drawtext:
            out_path = os.path.join(tmp_dir, f"anim_text_{uuid.uuid4().hex}.mp4")
            try:
                await _run([
                    FFMPEG_BIN, "-y", "-i", beat_clip_path, "-vf", drawtext,
                    *FFMPEG_X264_FLAGS, "-an", out_path,
                ])
                return out_path
            except Exception as e:
                print(f"[render] FFmpeg text-overlay fallback for '{animation_type}' failed: {e}")
        return beat_clip_path

    # --- ICON FIX: overlay_graphic/branding used to just fall through to
    # "return beat_clip_path" below with no FFmpeg rendering at all.
    icon_eligible = category in ("overlay_graphic", "branding")
    if icon_eligible:
        icon_drawtext = _build_icon_overlay_drawtext(animation, width, height, duration_frames, fps)
        if icon_drawtext:
            out_path = os.path.join(tmp_dir, f"anim_icon_{uuid.uuid4().hex}.mp4")
            try:
                await _run([
                    FFMPEG_BIN, "-y", "-i", beat_clip_path, "-vf", icon_drawtext,
                    *FFMPEG_X264_FLAGS, "-an", out_path,
                ])
                return out_path
            except Exception as e:
                print(f"[render] FFmpeg icon-overlay fallback for '{animation_type}' failed: {e}")
        else:
            print(f"[render] icon animation '{animation_type}' had no icon_name/display_text to draw — skipping")
        return beat_clip_path

    print(
        f"[render] animation '{animation_type}' (category={category}) needs Remotion "
        f"(pip/transition content has no FFmpeg equivalent) and none "
        f"was available — this beat renders without it"
    )
    return beat_clip_path

async def _lock_clip_to_frame_count(
    input_path: str, duration_frames: int, fps: int, width: int, height: int, tmp_dir: str,
) -> str:
    out_path = os.path.join(tmp_dir, "locked.mp4")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        "-vf", f"fps={fps},tpad=stop_mode=clone:stop=100",
        "-frames:v", str(duration_frames),
        *FFMPEG_X264_FLAGS,
        "-an",
        out_path,
    ]
    await _run(cmd)
    return out_path


async def _render_scene(
    scene: dict, fps: int, width: int, height: int, work_root: str,
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore,
    timeline_tracks_by_scene: Optional[dict] = None,
    caption_tracks_by_scene: Optional[dict] = None,
    animation_tracks_by_scene_beat: Optional[dict] = None,
) -> str:
    """
    `animation_tracks_by_scene_beat` is {scene_id: {beat_id: track}} — a
    scene can carry several animated beats, so this MUST be keyed by both
    scene and beat, not scene alone (see module docstring, point 1).
    """
    async with semaphore:
        scene_id = scene.get("scene_id", uuid.uuid4().hex)
        tmp_dir = os.path.join(work_root, f"scene_{scene_id}")
        os.makedirs(tmp_dir, exist_ok=True)

        trim = scene.get("trim") or {}
        start_sec = scene.get("start")
        if start_sec is None:
            start_sec = trim.get("start", 0.0)

        end_sec = scene.get("end")
        if end_sec is None:
            end_sec = trim.get("end", 0.0)

        if (start_sec in (None, 0.0)) or (end_sec in (None, 0.0)):
            word_segments = scene.get("word_segments") or []
            timed = [w for w in word_segments if "start" in w and "end" in w]
            if timed:
                if start_sec in (None,):
                    start_sec = timed[0]["start"]
                if end_sec in (None, 0.0):
                    end_sec = timed[-1]["end"]

        duration_frames = max(round((end_sec - start_sec) * fps), fps)
        target_seconds = duration_frames / fps

        scene_beat_tracks = (timeline_tracks_by_scene or {}).get(scene_id) or []
        beat_animations = (animation_tracks_by_scene_beat or {}).get(scene_id) or {}

        if scene_beat_tracks:
            beat_clip_paths = []
            for i, beat_track in enumerate(scene_beat_tracks):
                b_start_sec = beat_track.get("beat_start_sec")
                b_end_sec = beat_track.get("beat_end_sec")
                if b_start_sec is None or b_end_sec is None:
                    b_start_sec, b_end_sec = start_sec, end_sec

                beat_duration_frames = max(round((b_end_sec - b_start_sec) * fps), 1)

                beat_selected = beat_track.get("selected_asset")
                beat_broll_track = {"selected_asset": None}
                if beat_selected:
                    beat_broll_track = {"selected_asset": dict(beat_selected)}

                beat_bg_override = beat_track.get("background_color") or scene.get("background_color")
                beat_motion_type = beat_track.get("motion_type")

                beat_tmp_dir = os.path.join(tmp_dir, f"beat_{i}")
                os.makedirs(beat_tmp_dir, exist_ok=True)

                beat_clip = await _prepare_broll_clip(
                    beat_broll_track, beat_duration_frames, fps, width, height, beat_tmp_dir, client,
                    background_color_override=beat_bg_override,
                    motion_type=beat_motion_type,
                )

                # NEW: apply THIS beat's own animation (if any) onto its
                # own clip before concatenation — this is what actually
                # lets a scene carry more than one animated moment.
                this_beat_animation = beat_animations.get(beat_track.get("beat_id"))
                if this_beat_animation:
                    beat_clip = await _apply_beat_animation(
                        beat_clip, this_beat_animation, width, height, fps, beat_tmp_dir,
                    )

                beat_clip_paths.append(beat_clip)

            if len(beat_clip_paths) == 1:
                base_clip = beat_clip_paths[0]
            else:
                base_clip = await _concat_scenes(beat_clip_paths, tmp_dir, fps=fps)

            base_clip = await _lock_clip_to_frame_count(base_clip, duration_frames, fps, width, height, tmp_dir)
        else:
            # Legacy fallback: no beat tracks at all for this scene (older
            # timeline_json predating the beats refactor). No beat-level
            # animation to apply here since there's no beat to key it by.
            selected, source = None, None
            media = scene.get("media") or {}
            video_candidates = (media.get("videos") or {}).get("results") or []
            image_candidates = (media.get("images") or {}).get("results") or []
            if video_candidates:
                selected, source = dict(video_candidates[0]), "video"
            elif image_candidates:
                selected, source = dict(image_candidates[0]), "image"

            broll_track = {"selected_asset": None}
            if selected:
                broll_track = {"selected_asset": {**selected, "source": source}}

            background_color_override = scene.get("background_color")

            base_clip = await _prepare_broll_clip(
                broll_track, duration_frames, fps, width, height, tmp_dir, client,
                background_color_override=background_color_override,
                motion_type=None,
            )

        current = base_clip

        timeline_caption = (caption_tracks_by_scene or {}).get(scene_id)
        caption_style = (timeline_caption or {}).get("style") or scene.get("caption_style")

        words = scene.get("word_segments") or []
        words = [
            w for w in words
            if "start" in w and "end" in w
            and w["start"] >= start_sec and w["end"] <= end_sec
        ]
        frame_words = [
            {
                "word": w.get("word", ""),
                "startFrame": max(round((w["start"] - start_sec) * fps), 0),
                "endFrame": max(round((w["end"] - start_sec) * fps), 0),
            }
            for w in words
        ]
        current = await _burn_captions(
            current, frame_words, 0, fps, width, height, tmp_dir, style=caption_style
        )

        voiceover = scene.get("voiceover")
        final_scene_path = os.path.join(work_root, f"scene_{scene_id}_final.mp4")

        if voiceover and voiceover.get("url"):
            audio_path_raw = os.path.join(tmp_dir, "audio_raw.mp3")
            await _download(voiceover["url"], audio_path_raw, client)

            audio_path = os.path.join(tmp_dir, "audio_trimmed.m4a")
            trim_cmd = [
                FFMPEG_BIN, "-y",
                "-i", audio_path_raw,
                "-ss", f"{start_sec:.3f}",
                "-to", f"{end_sec:.3f}",
                "-c:a", "aac",
                audio_path,
            ]
            await _run(trim_cmd)
        else:
            print(f"[render] scene {scene_id} has no voiceover ({scene.get('error')}) — muxing silent audio instead")
            audio_path = await _make_silent_audio(duration_frames, fps, tmp_dir)

        locked_audio_path = os.path.join(tmp_dir, "audio_locked.m4a")
        lock_cmd = [
            FFMPEG_BIN, "-y",
            "-i", audio_path,
            "-af", f"apad=whole_dur={target_seconds:.3f}",
            "-t", f"{target_seconds:.3f}",
            "-c:a", "aac",
            locked_audio_path,
        ]
        await _run(lock_cmd)

        cmd = [
            FFMPEG_BIN, "-y",
            "-i", current,
            "-i", locked_audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-frames:v", str(duration_frames),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            final_scene_path,
        ]
        try:
            await _run(cmd)
        except Exception as e:
            print(f"[render] stream-copy mux failed for scene {scene_id}, falling back to re-encode: {e}")
            cmd = [
                FFMPEG_BIN, "-y",
                "-i", current,
                "-i", locked_audio_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-r", str(fps),
                "-vsync", "cfr",
                "-frames:v", str(duration_frames),
                *FFMPEG_X264_FLAGS,
                "-c:a", "aac",
                "-shortest",
                final_scene_path,
            ]
            await _run(cmd)

        return final_scene_path


async def _concat_scenes(scene_paths: list[str], work_root: str, fps: int = TIMELINE_FPS) -> str:
    list_path = os.path.join(work_root, "concat_list.txt")
    with open(list_path, "w") as f:
        for p in scene_paths:
            f.write(f"file '{p}'\n")

    out_path = os.path.join(work_root, "final_output.mp4")

    cmd = [FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path]
    try:
        await _run(cmd)
    except Exception as e:
        print(f"[render] concat stream-copy failed, falling back to re-encode: {e}")
        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-r", str(fps),
            "-vsync", "cfr",
            *FFMPEG_X264_FLAGS,
            "-c:a", "aac",
            out_path,
        ]
        await _run(cmd)

    return out_path


async def _run_render_job(video_id: str, timeline: dict, scenes: list, orientation: str) -> Optional[str]:
    no_voice_scene_ids = [
        s.get("scene_id") for s in scenes
        if not (s.get("voiceover") and s.get("voiceover", {}).get("url"))
    ]
    if no_voice_scene_ids:
        print(f"[render] video {video_id}: scenes with NO voiceover (will render silent): {no_voice_scene_ids}")

    fps = timeline.get("fps", 30)
    resolution = PORTRAIT_RESOLUTION if orientation == "portrait" else LANDSCAPE_RESOLUTION

    stored_resolution = timeline.get("resolution")
    if stored_resolution and (
        stored_resolution.get("width") != resolution["width"]
        or stored_resolution.get("height") != resolution["height"]
    ):
        print(
            f"[render] video {video_id}: overriding stored timeline resolution "
            f"{stored_resolution} -> {resolution} to match requested orientation='{orientation}'"
        )

    width, height = resolution["width"], resolution["height"]

    # timeline_tracks_by_scene: scene_id -> [broll tracks] (one per beat).
    # caption_tracks_by_scene: scene_id -> single caption_word track.
    # animation_tracks_by_scene_beat: scene_id -> {beat_id: animation
    #   track} — NOT a single track per scene, since a scene can have
    #   several animated beats (see module docstring, point 1). There is
    #   no "infographic" track type any more — the timeline only ever
    #   emits "audio" / "caption_word" / "broll" / "animation".
    timeline_tracks_by_scene = {}
    caption_tracks_by_scene = {}
    animation_tracks_by_scene_beat = {}
    for track in timeline.get("tracks", []):
        if track.get("type") == "broll" and track.get("scene_id"):
            timeline_tracks_by_scene.setdefault(track["scene_id"], []).append(track)
        elif track.get("type") == "caption_word" and track.get("scene_id"):
            caption_tracks_by_scene[track["scene_id"]] = track
        elif track.get("type") == "animation" and track.get("scene_id"):
            animation_tracks_by_scene_beat.setdefault(track["scene_id"], {})[track.get("beat_id")] = track

    for beat_tracks in timeline_tracks_by_scene.values():
        beat_tracks.sort(key=lambda t: t.get("startFrame", 0))

    try:
        supabase.table("videos").update({"render_status": "rendering"}).eq("id", video_id).execute()
    except Exception as e:
        print(f"[render] failed to set render_status=rendering for {video_id}: {e}")

    work_root = os.path.join(RENDER_TMP_ROOT, video_id)
    os.makedirs(work_root, exist_ok=True)
    semaphore = asyncio.Semaphore(RENDER_CONCURRENCY)

    try:
        async with httpx.AsyncClient() as client:
            async def _render_one(scene: dict):
                scene_id = scene.get("scene_id")
                try:
                    path = await _render_scene(
                        scene, fps, width, height, work_root, client, semaphore,
                        timeline_tracks_by_scene=timeline_tracks_by_scene,
                        caption_tracks_by_scene=caption_tracks_by_scene,
                        animation_tracks_by_scene_beat=animation_tracks_by_scene_beat,
                    )
                    return scene_id, path, None
                except Exception as e:
                    print(f"[render] scene {scene_id} failed: {e}")
                    return scene_id, None, scene_id

            results = await asyncio.gather(*(_render_one(scene) for scene in scenes))

            scene_paths_by_id = {sid: path for sid, path, _ in results if path}
            failed_scenes = [sid for _, path, sid in results if path is None]
            scene_paths = [
                scene_paths_by_id[s.get("scene_id")]
                for s in scenes
                if s.get("scene_id") in scene_paths_by_id
            ]

            if not scene_paths:
                print(f"[render] video {video_id}: all scenes failed to render")
                try:
                    supabase.table("videos").update({
                        "render_status": "failed", "failed_render_scene_ids": failed_scenes,
                    }).eq("id", video_id).execute()
                except Exception as e:
                    print(f"[render] failed to persist all-scenes-failed status for {video_id}: {e}")
                raise HTTPException(status_code=500, detail="All scenes failed to render")

            final_path = await _concat_scenes(scene_paths, work_root, fps=fps)

        try:
            final_duration = await _probe_duration_seconds(final_path)
            print(f"[render] video {video_id}: final output duration = {final_duration:.3f}s")
        except Exception as e:
            print(f"[render] could not probe final output duration: {e}")

        try:
            public_url = _upload_rendered_video_to_supabase(final_path, video_id)
            render_status = "completed" if not failed_scenes else "completed_with_errors"
        except Exception as e:
            print(f"[render] video {video_id}: Supabase upload failed: {e}")
            video_output_dir = os.path.join(RENDER_OUTPUT_DIR, video_id)
            os.makedirs(video_output_dir, exist_ok=True)
            dest_path = os.path.join(video_output_dir, "final.mp4")
            shutil.copy2(final_path, dest_path)
            public_url = None
            render_status = "completed_upload_failed"

        try:
            supabase.table("videos").update({
                "final_video_url": public_url,
                "render_status": render_status,
                "failed_render_scene_ids": failed_scenes,
            }).eq("id", video_id).execute()
        except Exception as e:
            print(f"[render] failed to persist final status for {video_id}: {e}")

        print(
            f"[render] video {video_id}: done, status={render_status}, "
            f"failed_scenes={failed_scenes}, no_voiceover_scenes={no_voice_scene_ids}"
        )

        if public_url is None:
            raise HTTPException(status_code=500, detail="Render completed but upload to storage failed")

        return public_url

    except HTTPException:
        try:
            supabase.table("videos").update({"render_status": "failed"}).eq("id", video_id).execute()
        except Exception:
            pass
        raise
    except Exception as e:
        print(f"[render] render failed for {video_id}: {e}")
        try:
            supabase.table("videos").update({"render_status": "failed"}).eq("id", video_id).execute()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Render failed: {e}")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


@app.post("/render/{video_id}")
async def render_video(video_id: str, request: RenderVideoRequest = RenderVideoRequest()):
    try:
        row = (
            supabase.table("videos")
            .select("timeline_json, raw_scenes, final_video_url")
            .eq("id", video_id)
            .single()
            .execute()
        )
    except Exception as e:
        print(f"[render] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    if row.data.get("final_video_url") and not request.force:
        return {"final_video_url": row.data["final_video_url"]}

    timeline = row.data.get("timeline_json") or {}
    scenes = row.data.get("raw_scenes") or []
    if not scenes:
        raise HTTPException(status_code=400, detail="No scenes to render for this video")

    final_video_url = await _run_render_job(video_id, timeline, scenes, request.orientation)

    return {"final_video_url": final_video_url}