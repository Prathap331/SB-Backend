
from fastapi import HTTPException
import os
import asyncio
from pydantic import BaseModel
import uuid
from typing import Optional,Literal,Any
from supabase import create_client
import json
import httpx
import re
import shutil
from contextlib import asynccontextmanager
from fastapi import HTTPException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import datetime


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


FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

supabase_url_env = os.getenv("SUPABASE_URL")
supabase_key_env = os.getenv("SUPABASE_KEY")
CAPTION_WORDS_PER_LINE = int(os.getenv("CAPTION_WORDS_PER_LINE", "10"))

_MOTION_TYPE_LIST = ["zoom_in", "zoom_out", "pan_left", "pan_right", "tilt_up", "tilt_down"]
_VALID_MOTION_TYPES = set(_MOTION_TYPE_LIST)
_DEFAULT_MOTION_TYPE = "zoom_in"


TIMELINE_WIDTH = int(os.getenv("TIMELINE_WIDTH", "1920"))
TIMELINE_HEIGHT = int(os.getenv("TIMELINE_HEIGHT", "1080"))

supabase = create_client(supabase_url_env, supabase_key_env)

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


CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
ANIMATION_CANVAS_WIDTH = CANVAS_WIDTH
ANIMATION_CANVAS_HEIGHT = CANVAS_HEIGHT


TIMELINE_FPS = int(os.getenv("TIMELINE_FPS", "30"))
SILENT_AUDIO_SAMPLE_RATE = int(os.getenv("SILENT_AUDIO_SAMPLE_RATE", "48000"))
SILENT_AUDIO_CHANNEL_LAYOUT = os.getenv("SILENT_AUDIO_CHANNEL_LAYOUT", "stereo")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_OUTPUT_DIR = os.environ.get("RENDER_OUTPUT_DIR", os.path.join(BASE_DIR, "rendered_videos"))
os.makedirs(RENDER_OUTPUT_DIR, exist_ok=True)
DEFAULT_CAPTION_FONT = os.getenv("DEFAULT_CAPTION_FONT", "Roboto")

LANDSCAPE_RESOLUTION = {"width": 1920, "height": 1080}
PORTRAIT_RESOLUTION = {"width": 1080, "height": 1920}

SUPABASE_RENDERED_VIDEOS_BUCKET = os.getenv("SUPABASE_RENDERED_VIDEOS_BUCKET", "rendered-videos")

_REMOTION_COMPOSITION_BY_ANIMATION_TYPE = {
    "full_screen_title_card": "TitleCard",
    "full_screen_quote_card": "QuoteCard",
    "full_screen_data_viz": "DataVizFullScreen",
    "full_screen_broll": "FullScreenBroll",  # no-op by design — see compositions.tsx
    "full_screen_transition": "FullScreenTransitionFx",
    "full_screen_color_wash": "FullScreenColorWash",
    "full_screen_document_highlight": "FullScreenDocumentHighlight", 
    "stat_counter_overlay": "StatCounterOverlay",
    "bullet_list_reveal": "BulletListReveal",
    "lower_third": "LowerThird",
    "kinetic_caption": "KineticCaption",
    "callout_textbox": "CalloutTextbox",
    "icon_sequence": "IconSequenceOverlay",
    "icon_pop_in": "IconPopIn",
    "logo_watermark": "LogoWatermark",
    "emoji_reaction": "EmojiReaction",
    "arrow_highlight": "ArrowHighlight",
    "badge_sticker": "BadgeSticker",
    "pip_video": "PipVideoFrame",
    "split_screen": "SplitScreenDivider",
    "multi_panel_grid": "MultiPanelGrid",
    "avatar_overlay": "AvatarOverlayPlaceholder",
    "mascot_animation": "MascotAnimationPlaceholder",
    "ken_burns_pan_zoom": "KenBurnsNoOp",
    "parallax_layering": "ParallaxAccent",
    "shake_impact": "ShakeImpactFlash",
    "speed_ramp_indicator": "SpeedRampIndicator",
}

REMOTION_MAX_CONCURRENT = int(os.getenv("REMOTION_MAX_CONCURRENT", "2"))
_remotion_semaphore = asyncio.Semaphore(REMOTION_MAX_CONCURRENT)



def _ease_expr(duration_frames: int) -> str:
    d = max(duration_frames - 1, 1)
    return f"(1-cos(PI*min(on/{d},1)))/2"

def _is_landscape_dimensions(width, height) -> bool:
    try:
        w = float(width)
        h = float(height)
    except (TypeError, ValueError):
        return False
    return w > 0 and h > 0 and w >= h * 1.2


def _normalize_ffmpeg_color(hex_color: str) -> str:
    h = (hex_color or "").strip().lstrip("#")
    if len(h) != 6:
        h = "111827"
    return f"0x{h}"


def _looks_like_playable_media_url(url: Optional[str]) -> bool:
    if not url:
        return False
    if "videos.pexels.com" in url or "images.pexels.com" in url:
        return True
    if re.search(r"\.(mp4|mov|webm|jpg|jpeg|png|webp)(\?|$)", url, re.IGNORECASE):
        return True
    return False



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

    rendered = False
    async with _remotion_semaphore:
        try:
            await _run(_build_cmd(f"0-{duration_frames - 1}"), cwd=REMOTION_PROJECT_DIR)
            rendered = True
        except RuntimeError as e:
            msg = str(e)
            m = re.search(
                r"durationInFrames.*?evaluated to be (\d+).*?not inbetween 0-(\d+)",
                msg, re.IGNORECASE | re.DOTALL,
            )
            max_frame = int(m.group(2)) if m else None
            if max_frame is None:
                # Couldn't parse a specific bound from the error — try a
                # single generic retry with no --frames constraint at all
                # (let the composition use its own default length) rather
                # than giving up outright. If this also fails, there's
                # nothing more to try.
                print(f"[render] '{composition_id}' render failed and error format wasn't recognized ({e}) — retrying with the composition's own default duration")
                try:
                    await _run(_build_cmd(None), cwd=REMOTION_PROJECT_DIR)
                    rendered = True
                except Exception as e2:
                    print(f"[render] '{composition_id}': fallback retry also failed: {e2}")
                    return None
            else:
                print(f"[render] '{composition_id}' has a fixed durationInFrames — retrying within its native bounds (0-{max_frame})")
                try:
                    await _run(_build_cmd(f"0-{max_frame}"), cwd=REMOTION_PROJECT_DIR)
                    rendered = True
                except Exception as e2:
                    print(f"[render] '{composition_id}': retry also failed: {e2}")
                    return None

    if not rendered:
        return None

    # Unconditional check: does the actual rendered file have as many
    # frames as requested? If not — for ANY reason, error-message-parsed
    # or not — freeze-pad its last frame out to duration_frames so the
    # overlay stays visible for its full intended duration rather than
    # vanishing early (the render pipeline composites this clip starting
    # at frame 0 of wherever it's placed, so a short clip directly means
    # early disappearance, independent of how correct the upstream
    # anchor_start_sec/anchor_end_sec computation was).
    try:
        actual_seconds = await _probe_duration_seconds(out_path)
        actual_frames = max(1, round(actual_seconds * fps))
    except Exception as e:
        print(f"[render] '{composition_id}': couldn't verify rendered clip length ({e}) — using it as-is")
        return out_path

    if actual_frames < duration_frames:
        padded_path = os.path.join(tmp_dir, f"anim_padded_{uuid.uuid4().hex}.mov")
        pad_cmd = [
            FFMPEG_BIN, "-y", "-i", out_path,
            "-vf", f"tpad=stop_mode=clone:stop={duration_frames - actual_frames}",
            "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
            padded_path,
        ]
        try:
            await _run(pad_cmd)
            print(
                f"[render] '{composition_id}': rendered clip was {actual_frames}/{duration_frames} "
                f"frames — padded by freezing its final frame so it stays visible for the "
                f"full intended duration instead of vanishing early"
            )
            return padded_path
        except Exception as e3:
            print(
                f"[render] '{composition_id}': failed to pad truncated clip, "
                f"leaving it short at {actual_frames}/{duration_frames} frames: {e3}"
            )

    return out_path


def _frames_to_ass_time(frame: int, fps: int) -> str:
    total_seconds = frame / fps
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = total_seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _hex_to_ass_color(hex_color: str, alpha_hex: str = "00") -> str:
    h = (hex_color or "").strip().lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha_hex}{b}{g}{r}"




def _build_ass_from_words(
    words: list[dict], scene_start_frame: int, fps: int, width: int, height: int,
    style: Optional[dict] = None,
) -> str:
    style = style or {}
    font_size = style.get("font_size") or 72
    font_family = style.get("font_family") or DEFAULT_CAPTION_FONT
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
Style: Caption,{font_family},{font_size},{primary_color},{outline_color},&H80000000,1,4,0,{alignment},{margin_l},{margin_r},{margin_v}

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
    CHUNK_SIZE = style.get("words_per_line") or CAPTION_WORDS_PER_LINE
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


def _scale_geometry_px(geometry_px: dict, out_width: int, out_height: int) -> dict:
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
    lines = text.split("\n") or [text]
    longest = max((len(l) for l in lines), default=1) or 1
    chars_per_100px = 16
    width_based = int((geo["width"] / max(longest, 1)) * (100 / chars_per_100px) * 0.6)
    height_based = int(geo["height"] / max(len(lines), 1) * 0.7) if geo["height"] else width_based
    return max(20, min(width_based, height_based, 25))

def _ffmpeg_box_color(background_color_hint: Optional[str], alpha: float, default: str = "black") -> str:
    if background_color_hint:
        h = background_color_hint.strip().lstrip("#")
        if len(h) == 6:
            return f"0x{h}@{alpha}"
    return f"{default}@{alpha}"


def _build_icon_overlay_drawtext(
    animation: dict, out_width: int, out_height: int, clip_duration_frames: int, fps: int,
    offset_seconds: float = 0.0,
) -> Optional[str]:

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
    o = offset_seconds
    alpha_expr = (
        f"if(lt(t,{o}),0,"
        f"if(lt(t,{o + fade_in}),(t-{o})/{fade_in},"
        f"if(gt(t,{o + duration_seconds - fade_out}),({o + duration_seconds}-t)/{fade_out},1)))"
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
            f"box=1:boxcolor={_ffmpeg_box_color(animation.get('background_color_hint'), 0.5)}:boxborderw=14:"
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
                seg_start = o + i * slice_seconds
                seg_end = o + (duration_seconds if i == n - 1 else (i + 1) * slice_seconds)
                local_fade_in = min(0.2, slice_seconds / 4)
                seg_alpha = (
                    f"if(lt(t,{seg_start}),0,"
                    f"if(lt(t,{seg_start + local_fade_in}),(t-{seg_start})/{local_fade_in},"
                    f"if(lt(t,{seg_end}),1,0)))"
                )
                filters.append(
                    f"drawtext=text='{safe_text}':fontcolor=white:fontsize={icon_font_size}:"
                    f"box=1:boxcolor={_ffmpeg_box_color(animation.get('background_color_hint'), 0.5)}:boxborderw=14:"
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
                    f"box=1:boxcolor={_ffmpeg_box_color(animation.get('background_color_hint'), 0.5)}:boxborderw=10:"
                    f"x={cx:.1f}-text_w/2:y={y0:.1f}-text_h/2:alpha='{alpha_expr}'"
                )
        if label:
            safe_label = _escape_drawtext(label)
            label_font_size = max(20, min(int(geo["height"] * 0.28), 56))
            lx = geo["x"] + geo["width"] / 2
            ly = geo["y"] + geo["height"] * 0.82
            filters.append(
                f"drawtext=text='{safe_label}':fontcolor=white:fontsize={label_font_size}:"
                f"box=1:boxcolor={_ffmpeg_box_color(animation.get('background_color_hint'), 0.5)}:boxborderw=10:"
                f"x={lx:.1f}-text_w/2:y={ly:.1f}-text_h/2:alpha='{alpha_expr}'"
            )

    return ",".join(filters)


    
def _build_beat_animation_drawtext(
    animation: dict, out_width: int, out_height: int, clip_duration_frames: int, fps: int,
    offset_seconds: float = 0.0,
) -> Optional[str]:
    text = _display_text_to_string(animation.get("display_text"))
    if not text and animation.get("highlight_target_text"):
        # No real screenshot asset to composite — degrade to showing the
        # quoted text itself rather than dropping the beat's emphasis.
        text = animation["highlight_target_text"]
    if not text:
        return None

    geo = _scale_geometry_px(animation.get("geometry_px") or {}, out_width, out_height)
    font_size = animation.get("font_size") or _font_size_for_geometry(geo, text)

    duration_seconds = max(clip_duration_frames / fps, 0.2)
    fade_in = min(0.4, duration_seconds / 4)
    fade_out = min(0.4, duration_seconds / 4)
    o = offset_seconds
    alpha_expr = (
        f"if(lt(t,{o}),0,"
        f"if(lt(t,{o + fade_in}),(t-{o})/{fade_in},"
        f"if(gt(t,{o + duration_seconds - fade_out}),({o + duration_seconds}-t)/{fade_out},1)))"
    )

    safe_text = _escape_drawtext(text)
    x = f"{geo['x']:.1f}"
    y = f"{geo['y']:.1f}"

    return (
        f"drawtext=text='{safe_text}':fontcolor=white:fontsize={font_size}:"
        f"box=1:boxcolor={_ffmpeg_box_color(animation.get('background_color_hint'), 0.55)}:boxborderw=16:line_spacing=8:"
        f"x={x}:y={y}:alpha='{alpha_expr}'"
    )


def _build_remotion_props(animation_type: str, animation: dict, width: int, height: int) -> dict:

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
        return {
            "icons": icons or ["sparkles"], "label": text,
            "colorHint": animation.get("color_hint"),
            "backgroundColorHint": animation.get("background_color_hint"),
            "geometryPx": _scale_geometry_px(animation.get("geometry_px") or {}, width, height),
        }

    if animation_type == "icon_pop_in":
        return {
            "icons": (icons[:1] or ["sparkles"]), "label": text,
            "colorHint": animation.get("color_hint"),
            "backgroundColorHint": animation.get("background_color_hint"),
            "geometryPx": _scale_geometry_px(animation.get("geometry_px") or {}, width, height),
        }

    # Everything else (compositions/Extra.tsx) — generic shape.
    return {
        "displayText": animation.get("display_text"),
        "colorHint": animation.get("color_hint"),
        "backgroundColorHint": animation.get("background_color_hint"),
        "fontSize": animation.get("font_size"),
        "geometryPx": _scale_geometry_px(animation.get("geometry_px") or {}, width, height),
        "iconName": animation.get("icon_name"),
        "iconLayout": animation.get("icon_layout"),
        "highlightTargetText": animation.get("highlight_target_text"),
        "motion": animation.get("motion"),
    }


async def _apply_beat_animation(
    beat_clip_path: str, animation: dict, width: int, height: int, fps: int, tmp_dir: str,
    offset_seconds: float = 0.0,
) -> str:
    category = animation.get("category")
    animation_type = animation.get("animation_type")
    duration_frames = animation.get("duration_frames") or fps * 2
    offset_seconds = max(0.0, offset_seconds)

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
        # -itsoffset on the overlay INPUT shifts its presentation
        # timestamps forward by offset_seconds, so it has no frames to
        # show until that point in the merged timeline — before then,
        # overlay just passes the base beat clip through unchanged. This
        # is what actually delays the animation to its real anchor time
        # instead of always starting at the beat's own frame 0.
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", beat_clip_path,
            "-itsoffset", f"{offset_seconds:.3f}", "-i", overlay_clip,
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
        drawtext = _build_beat_animation_drawtext(animation, width, height, duration_frames, fps, offset_seconds)
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
        icon_drawtext = _build_icon_overlay_drawtext(animation, width, height, duration_frames, fps, offset_seconds)
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

                track_start_frame = beat_track.get("startFrame")
                track_end_frame = beat_track.get("endFrame")
                if track_start_frame is not None and track_end_frame is not None:
                    beat_duration_frames = max(track_end_frame - track_start_frame, 1)
                else:
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

                relevant_animations = []
                for anim in beat_animations.values():
                    a_start_frame = anim.get("startFrame")
                    a_end_frame = anim.get("endFrame")
                    if a_start_frame is None or a_end_frame is None:
                        continue
                    if a_end_frame <= track_start_frame or a_start_frame >= track_end_frame:
                        continue  # no overlap with this beat at all
                    relevant_animations.append((a_start_frame, anim))
                relevant_animations.sort(key=lambda pair: pair[0])

                for a_start_frame, anim in relevant_animations:
                    a_end_frame = anim.get("endFrame")
                    local_start_frame = max(0, a_start_frame - track_start_frame)
                    local_end_frame = min(beat_duration_frames, a_end_frame - track_start_frame)
                    if local_end_frame <= local_start_frame:
                        continue
                    offset_seconds = local_start_frame / fps
                    local_anim = dict(anim)
                    local_anim["duration_frames"] = local_end_frame - local_start_frame
                    beat_clip = await _apply_beat_animation(
                        beat_clip, local_anim, width, height, fps, beat_tmp_dir,
                        offset_seconds=offset_seconds,
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




def _seconds_to_frames(seconds: float, fps: int = TIMELINE_FPS) -> int:
    return max(round((seconds or 0.0) * fps), 0)


DEFAULT_CAPTION_STYLE = {
    "vertical_position": "bottom",
    "margin_bottom_percent": 3,
}


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

    preferred = beat.get("preferred_media_type")
    if preferred == "image":
        if best_image:
            return best_image, "image"
        if best_video:
            return best_video, "video"
        return None, None
    if preferred == "video":
        if best_video:
            return best_video, "video"
        if best_image:
            return best_image, "image"
        return None, None

    if best_video:
        return best_video, "video"
    if best_image:
        return best_image, "image"
    return None, None


def _resolve_beat_motion_type(beat: dict) -> str:
    override = beat.get("broll_override") or {}
    motion = override.get("motion_type")
    if motion in _VALID_MOTION_TYPES:
        return motion
    motion = beat.get("motion_type")
    if motion in _VALID_MOTION_TYPES:
        return motion
    return _DEFAULT_MOTION_TYPE

_PHRASE_MATCH_PUNCT_RE = re.compile(r"[^\w\s]")

def _normalize_phrase_word(w: str) -> str:
    return _PHRASE_MATCH_PUNCT_RE.sub("", w or "").lower().strip()


def _find_phrase_span_sec(
    target_text: Optional[str], timed_words: list, search_start: float, search_end: float,
    hint_start_sec: Optional[float] = None,
) -> Optional[tuple]:
    if not target_text or not timed_words:
        return None
    target_words = [_normalize_phrase_word(w) for w in target_text.split()]
    target_words = [w for w in target_words if w]
    if not target_words:
        return None

    window = [
        w for w in timed_words
        if "start" in w and "end" in w and w["start"] >= search_start - 0.05 and w["start"] < search_end + 0.05
    ]
    window_norm = [_normalize_phrase_word(w.get("word", "")) for w in window]
    n = len(window_norm)
    if n == 0:
        return None

    n_target = len(target_words)
    min_run = min(n_target, 3)
    min_ratio = 0.6 if n_target >= 5 else 1.0  # short phrases still need a near-exact hit

    candidates = []  # every qualifying (score, offset) pair, not just the best
    best_score = None
    for offset in range(-n_target, n):
        matches = 0
        run = 0
        best_run = 0
        for t in range(n_target):
            wi = t + offset
            if 0 <= wi < n and window_norm[wi] == target_words[t]:
                matches += 1
                run += 1
                best_run = max(best_run, run)
            else:
                run = 0
        if matches == 0:
            continue
        ratio = matches / n_target
        if best_run < min_run and ratio < min_ratio:
            continue
        score = (best_run, matches)
        candidates.append((score, offset))
        if best_score is None or score >= best_score:
            best_score = score

    if not candidates:
        return None

    def _offset_to_start_sec(offset: int) -> float:
        idx = max(0, min(offset, n - 1))
        return window[idx]["start"]

    if hint_start_sec is not None:
        strong_candidates = [c for c in candidates if c[0][1] >= best_score[1] - 1]
        best_offset = min(strong_candidates, key=lambda c: abs(_offset_to_start_sec(c[1]) - hint_start_sec))[1]
    else:
        best_offset = None
        best_kept_score = None
        for score, offset in candidates:
            if best_kept_score is None or score >= best_kept_score:
                best_kept_score = score
                best_offset = offset

    start_idx = max(0, min(best_offset, n - 1))
    end_idx = max(0, min(n_target - 1 + best_offset, n - 1))
    if end_idx < start_idx:
        end_idx = start_idx
    return window[start_idx]["start"], window[end_idx]["end"]



_SAFE_MARGIN = 72
_CAPTION_BAND_HEIGHT = round(CANVAS_HEIGHT * 0.15)
_CAPTION_BAND_TOP_Y = CANVAS_HEIGHT - _CAPTION_BAND_HEIGHT
CAPTION_SAFE_ZONE_Y = _CAPTION_BAND_TOP_Y



def _content_aware_max_box_size(display_text: Any) -> tuple:
    
    text = _display_text_to_string(display_text) if isinstance(display_text, (str, list)) else ""
    if not text:
        return (400, 110)
    lines = [l for l in text.split("\n") if l.strip()] or [text]
    longest_line = max((len(l) for l in lines), default=0)
    num_lines = max(len(lines), 1)
    max_width = min(700, max(220, longest_line * 13 + 100))
    max_height = min(260, max(80, num_lines * 46 + 50))
    return max_width, max_height


def _validate_geometry_px(
    raw: Any, category: str, allow_manual_placement: bool = False, display_text: Any = None,
) -> dict:
    if category in ("full_screen", "transition"):
        return {"x": 0, "y": 0, "width": ANIMATION_CANVAS_WIDTH, "height": ANIMATION_CANVAS_HEIGHT}

    default = {"width": 520, "height": 160}
    if not isinstance(raw, dict):
        geo = {"x": 0, "y": 0, **default}
    else:
        try:
            geo = {
                "x": int(raw.get("x", 0)) if allow_manual_placement else 0,
                "y": int(raw.get("y", 0)) if allow_manual_placement else 0,
                "width": int(raw.get("width", default["width"])),
                "height": int(raw.get("height", default["height"])),
            }
        except (TypeError, ValueError):
            geo = {"x": 0, "y": 0, **default}

    max_width = max(40, ANIMATION_CANVAS_WIDTH - 2 * _SAFE_MARGIN)
    max_height = max(40, ANIMATION_CANVAS_HEIGHT - 2 * _SAFE_MARGIN)
    if category == "overlay_text" and display_text is not None:
        content_max_w, content_max_h = _content_aware_max_box_size(display_text)
        max_width = min(max_width, content_max_w)
        max_height = min(max_height, content_max_h)
    geo["width"] = max(40, min(geo["width"], max_width))
    geo["height"] = max(40, min(geo["height"], max_height))

    if allow_manual_placement and isinstance(raw, dict) and ("x" in raw or "y" in raw):
        geo["x"] = max(_SAFE_MARGIN, min(geo["x"], ANIMATION_CANVAS_WIDTH - _SAFE_MARGIN - geo["width"]))
        geo["y"] = max(_SAFE_MARGIN, min(geo["y"], ANIMATION_CANVAS_HEIGHT - _SAFE_MARGIN - geo["height"]))
        if category in ("overlay_text", "overlay_graphic") and geo["y"] + geo["height"] > CAPTION_SAFE_ZONE_Y:
            if geo["height"] < CAPTION_SAFE_ZONE_Y - _SAFE_MARGIN:
                geo["y"] = CAPTION_SAFE_ZONE_Y - geo["height"]
            else:
                geo["height"] = CAPTION_SAFE_ZONE_Y - _SAFE_MARGIN - 4
                geo["y"] = _SAFE_MARGIN
        return geo

    geo["x"] = (ANIMATION_CANVAS_WIDTH - geo["width"]) // 2

    if category in ("overlay_text", "overlay_graphic"):
        centered_y = (ANIMATION_CANVAS_HEIGHT - geo["height"]) // 2
        if centered_y + geo["height"] > CAPTION_SAFE_ZONE_Y:
            if geo["height"] < CAPTION_SAFE_ZONE_Y - _SAFE_MARGIN:
                centered_y = CAPTION_SAFE_ZONE_Y - geo["height"]
            else:
                geo["height"] = CAPTION_SAFE_ZONE_Y - _SAFE_MARGIN - 4
                centered_y = _SAFE_MARGIN
        geo["y"] = centered_y
    else:
        geo["y"] = (ANIMATION_CANVAS_HEIGHT - geo["height"]) // 2

    return geo



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
        timed_words_sec = [w for w in word_segments if "start" in w and "end" in w]
        words = []
        for w in word_segments:
            if "start" not in w or "end" not in w:
                continue
            w_start_frame = scene_start_frame + _seconds_to_frames(w["start"] - start_sec, fps)
            w_end_frame = scene_start_frame + _seconds_to_frames(w["end"] - start_sec, fps)
            # Defensive clamp: WhisperX timestamps can round to a frame or
            # two past the scene's own computed end — never let a caption
            # word render past the scene/voiceover boundary.
            w_start_frame = max(scene_start_frame, min(w_start_frame, scene_end_frame))
            w_end_frame = max(w_start_frame, min(w_end_frame, scene_end_frame))
            words.append({"word": w.get("word", ""), "startFrame": w_start_frame, "endFrame": w_end_frame})
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

        scene_animation_tracks = []
        for animation in (scene.get("animations") or []):
            beat_id = animation.get("beat_id")
            rng = beat_frame_ranges.get(beat_id)
            if not rng:
                continue
            b_start_frame, b_end_frame = rng

            anim_start_frame = b_start_frame
            matched_end_frame = None
            target_text = animation.get("highlight_target_text")
            if not target_text:
                candidate_text = _display_text_to_string(animation.get("display_text"))
                if candidate_text:
                    target_text = candidate_text

            model_anchor_start = animation.get("anchor_start_sec")
            model_anchor_end = animation.get("anchor_end_sec")

            heuristic_span = _find_phrase_span_sec(
                target_text, timed_words_sec, start_sec, end_sec, hint_start_sec=model_anchor_start,
            )

            use_model_anchor = model_anchor_start is not None and heuristic_span is None
            if model_anchor_start is not None and heuristic_span is not None:
                heuristic_start_sec = heuristic_span[0]
                if abs(model_anchor_start - heuristic_start_sec) > 0.5:
                    print(
                        f"[edit-video] scene {scene_id} beat {beat_id}: Animation Planner's "
                        f"anchor_start_sec ({model_anchor_start:.2f}s) disagrees with the "
                        f"heuristic transcript match ({heuristic_start_sec:.2f}s) — using the "
                        f"heuristic match (deterministic, grounded in the real transcript, "
                        f"and has proven more reliable than trusting the model's own anchor)"
                    )

            if use_model_anchor:
                candidate_start_frame = scene_start_frame + _seconds_to_frames(model_anchor_start - start_sec, fps)
                anim_start_frame = max(scene_start_frame, min(candidate_start_frame, scene_end_frame))
            elif heuristic_span is not None:
                matched_start_sec, _ = heuristic_span
                candidate_start_frame = scene_start_frame + _seconds_to_frames(matched_start_sec - start_sec, fps)
                anim_start_frame = max(scene_start_frame, min(candidate_start_frame, scene_end_frame))

            model_end_frame = min(anim_start_frame + animation.get("duration_frames", 90), scene_end_frame)
            if heuristic_span is not None:
                _, heuristic_end_sec = heuristic_span
                candidate_end_frame = scene_start_frame + _seconds_to_frames(heuristic_end_sec - start_sec, fps)
                heuristic_end_frame = max(anim_start_frame, min(candidate_end_frame, scene_end_frame))
                hold_buffer_frames = round(0.8 * fps)
                speech_end_frame = min(heuristic_end_frame + hold_buffer_frames, scene_end_frame)

                max_reasonable_extra_frames = round(3.0 * fps)
                max_end_frame = min(speech_end_frame + max_reasonable_extra_frames, scene_end_frame)

                candidate_ends = [speech_end_frame, model_end_frame]
                if use_model_anchor and model_anchor_end is not None:
                    candidate_end_frame = scene_start_frame + _seconds_to_frames(model_anchor_end - start_sec, fps)
                    candidate_ends.append(max(anim_start_frame, min(candidate_end_frame, scene_end_frame)))

                anim_end_frame = max(speech_end_frame, min(max(candidate_ends), max_end_frame))
            else:
                anim_end_frame = model_end_frame
                if use_model_anchor and model_anchor_end is not None:
                    candidate_end_frame = scene_start_frame + _seconds_to_frames(model_anchor_end - start_sec, fps)
                    model_anchor_end_frame = max(anim_start_frame, min(candidate_end_frame, scene_end_frame))
                    anim_end_frame = max(anim_end_frame, model_anchor_end_frame)
            anim_start_frame = max(scene_start_frame, min(anim_start_frame, scene_end_frame))
            anim_end_frame = max(anim_start_frame, min(anim_end_frame, scene_end_frame))

            safe_geometry_px = _validate_geometry_px(
                animation.get("geometry_px"), animation.get("category"),
                allow_manual_placement=bool(animation.get("manually_placed")),
                display_text=animation.get("display_text"),
            )

            scene_animation_tracks.append({
                "track_id": f"anim_{scene_id}_{beat_id}",
                "scene_id": scene_id, "beat_id": beat_id, "type": "animation",
                "layer": animation.get("z_index_layer", "foreground"),
                "animation_type": animation.get("animation_type"), "category": animation.get("category"),
                "placement": animation.get("placement"), "geometry_px": safe_geometry_px,
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

        scene_animation_tracks.sort(key=lambda t: t["startFrame"])
        for i in range(len(scene_animation_tracks) - 1):
            cur = scene_animation_tracks[i]
            nxt = scene_animation_tracks[i + 1]
            if cur["endFrame"] > nxt["startFrame"]:
                cur["endFrame"] = max(cur["startFrame"], nxt["startFrame"])
                cur["end_sec"] = cur["endFrame"] / fps
                cur["duration_frames"] = cur["endFrame"] - cur["startFrame"]
        tracks.extend(scene_animation_tracks)

        cumulative_frames = scene_end_frame

    return {
        "fps": fps, "total_frames": cumulative_frames,
        "resolution": {"width": TIMELINE_WIDTH, "height": TIMELINE_HEIGHT},
        "tracks": tracks,
    }



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


def _sum_batches(batches: list[dict]) -> int:
    return sum(int(b.get("remaining", 0)) for b in batches)


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

@app.post("/render/{video_id}")
async def render_video(video_id: str, request: RenderVideoRequest = RenderVideoRequest()):
    try:
        row = (
            supabase.table("videos")
            .select("raw_scenes, final_video_url, user_id")
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

    scenes = row.data.get("raw_scenes") or []
    if not scenes:
        raise HTTPException(status_code=400, detail="No scenes to render for this video")

    user_id = row.data.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Video has no associated user")

    # --- Deduct 50 credits for this render, same batch logic as /unlock ---
    RENDER_CREDIT_COST = 50
    try:
        profile_res = supabase.table('user_profiles') \
            .select('id, credit_batches') \
            .eq('id', user_id) \
            .maybe_single() \
            .execute()

        if not profile_res.data:
            raise HTTPException(status_code=404, detail="user profile not found")

        batches = profile_res.data.get('credit_batches') or []
        now = datetime.datetime.now(datetime.timezone.utc)
        active_batches = _expire_stale_batches(batches, now)

        updated_batches, deducted = _deduct_from_batches(active_batches, RENDER_CREDIT_COST)
        if deducted == 0:
            raise HTTPException(status_code=402, detail="credits not sufficient")

        new_total = _sum_batches(updated_batches)

        supabase.table('user_profiles').update({
            'credit_batches': updated_batches,
            'credits_remaining': new_total,
        }).eq('id', user_id).execute()

    except HTTPException:
        raise
    except Exception as e:
        print(f"[render] failed to deduct credits for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="failed to deduct credits")

    timeline = build_timeline_from_scenes(scenes)
    try:
        infographics_list, text_list = _compute_infographics_and_text_lists(scenes, timeline)
        broll_list = _compute_broll_list(timeline)
        supabase.table("videos").update({
            "timeline_json": timeline, "raw_scenes": scenes,
            "infographics_list": infographics_list, "text_list": text_list, "broll_list": broll_list,
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[render] failed to persist freshly rebuilt timeline for {video_id} (rendering with it anyway): {e}")

    final_video_url = await _run_render_job(video_id, timeline, scenes, request.orientation)

    return {"final_video_url": final_video_url}