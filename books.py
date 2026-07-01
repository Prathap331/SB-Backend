
import os, re, sys, time, shutil, subprocess, html as html_mod
import urllib.request
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, unquote

import requests
import pandas as pd
import fitz
from sentence_transformers import SentenceTransformer
from supabase import create_client

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

try:
    import libtorrent as lt
    LIBTORRENT_AVAILABLE = True
    print("[torrent] libtorrent loaded OK")
except ImportError:
    lt = None
    LIBTORRENT_AVAILABLE = False
    print("[torrent] libtorrent not installed — torrent fallback disabled")
    print("          Ubuntu/Debian : sudo apt install python3-libtorrent")
    print("          pip           : pip install python-libtorrent")

CSV_FILE            = "english_books.csv"
OUTPUT_DIR          = os.path.abspath("downloaded_books")
NUM_BOOKS           = 15

PAGE_LOAD_TIMEOUT   = 60
SLOW_PAGE_WAIT      = 30
DELAY_BETWEEN_BOOKS = 10

BASE_URL            = "https://annas-archive.pk"
CHUNK_SIZE          = 500

MAX_RETRIES         = 3
RETRY_BASE_DELAY    = 5

DOWNLOAD_MAX_WAIT   = 300
DOWNLOAD_POLL       = 1

TORRENT_TIMEOUT     = 300
TORRENT_POLL        = 5

_sb_url = os.getenv("SUPABASE_URL")
_sb_key = os.getenv("SUPABASE_KEY")
if not _sb_url or not _sb_key:
    sys.exit("[ERROR] SUPABASE_URL and SUPABASE_KEY env vars must be set.")
supabase = create_client(_sb_url, _sb_key)

print("Loading embedding model…")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

_http = requests.Session()
_http.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL + "/",
})



def make_driver():
    opts = Options()
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,900")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    prefs = {
        "download.default_directory": OUTPUT_DIR,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
        "download.open_pdf_in_system_reader": False,
        "profile.default_content_settings.popups": 0,
        "profile.default_content_setting_values.automatic_downloads": 1,
    }
    opts.add_experimental_option("prefs", prefs)
    drv = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts,
    )
    drv.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    drv.implicitly_wait(10)
    drv.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )
    drv.execute_cdp_cmd(
        "Page.setDownloadBehavior",
        {"behavior": "allow", "downloadPath": OUTPUT_DIR},
    )
    return drv



def wait_for_chrome_download(timeout=DOWNLOAD_MAX_WAIT, poll=DOWNLOAD_POLL):
    print(f"  [dl] Waiting up to {timeout}s for Chrome download to finish…")
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll)
        files     = os.listdir(OUTPUT_DIR)
        in_flight = [f for f in files if f.endswith(".crdownload") or f.endswith(".part")]
        complete  = [
            f for f in files
            if not f.endswith(".crdownload") and not f.endswith(".part")
            and os.path.isfile(os.path.join(OUTPUT_DIR, f))
        ]
        if complete and not in_flight:
            newest = max(complete, key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)))
            fpath  = os.path.join(OUTPUT_DIR, newest)
            print(f"  [dl] Download complete: {newest}")
            return fpath
    print("  [!] Download timed out")
    return None



def safe_get(driver, url, retries=MAX_RETRIES):
    for attempt in range(1, retries + 1):
        try:
            driver.get(url)
            return True
        except TimeoutException:
            print(f"  [!] Page load timed out ({attempt}/{retries}): {url[:80]}")
        except WebDriverException as exc:
            print(f"  [!] WebDriver error ({attempt}/{retries}): {exc.msg[:100]}")
        if attempt < retries:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"  Retrying in {delay}s…")
            time.sleep(delay)
    print(f"  [✗] Giving up: {url[:80]}")
    return False



def download_direct(url, title_hint="book"):
    print(f"  [http] Downloading: {url[:90]}")
    try:
        head      = _http.head(url, allow_redirects=True, timeout=30)
        final_url = head.url
        fname     = None
        cd        = head.headers.get("Content-Disposition", "")
        cd_match  = re.search(
            r"filename\*?=(?:UTF-8'')?[\"']?([^\"';\r\n]+)", cd, re.IGNORECASE
        )
        if cd_match:
            fname = unquote(cd_match.group(1)).strip().strip('"').strip("'")
        if not fname:
            fname = unquote(urlparse(final_url).path.split("/")[-1]) or "book"
        fname = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", fname)
        if not os.path.splitext(fname)[1]:
            fname += ".pdf"
        dest = os.path.join(OUTPUT_DIR, fname)
        with _http.get(final_url, stream=True, timeout=180) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            done  = 0
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            print(
                                f"  [http] {done/total*100:5.1f}%  "
                                f"({done//1024} KB / {total//1024} KB)",
                                end="\r",
                            )
        print(f"\n  [http] Saved {done//1024} KB → {dest}")
        return dest
    except requests.RequestException as exc:
        print(f"  [!] HTTP download failed: {exc}")
        return None
    except Exception as exc:
        print(f"  [!] Unexpected download error: {exc}")
        return None



def _fb2_extract_text(fb2_path: str) -> str:
    """
    Parse FB2 XML and return plain text.
    FB2 body paragraphs live in <body><section><p> elements.
    Namespace-aware — handles both bare and namespaced FB2 files.
    """
    try:
        tree = ET.parse(fb2_path)
        root = tree.getroot()
    except ET.ParseError:
        with open(fb2_path, "rb") as fh:
            raw = fh.read()
        raw = re.sub(rb'<\?xml[^?]*\?>', b'', raw, count=1)
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as e:
            print(f"  [!] FB2 XML parse failed: {e}")
            return ""

    ns_match = re.match(r'\{[^}]+\}', root.tag)
    ns       = ns_match.group(0) if ns_match else ""

    lines = []
    for body in root.iter(f"{ns}body"):
        for elem in body.iter():
            tag = elem.tag.replace(ns, "")
            if tag in ("p", "v"):                     
                text = (elem.text or "").strip()
                if text:
                    lines.append(html_mod.unescape(text))
            elif tag in ("title", "subtitle", "epigraph"):
                text = "".join(t for t in elem.itertext()).strip()
                if text:
                    lines.append("\n" + html_mod.unescape(text).upper() + "\n")
            elif tag == "empty-line":
                lines.append("")
    return "\n".join(lines)


def _fb2_to_pdf_native(fb2_path: str, pdf_path: str) -> bool:
    """Convert FB2 → PDF using reportlab without Calibre."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.enums import TA_JUSTIFY
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        print("  [!] reportlab not installed — pip install reportlab")
        return False

    print("  [fb2→pdf] Extracting text from FB2…")
    text = _fb2_extract_text(fb2_path)
    if not text.strip():
        print("  [!] No text extracted from FB2")
        return False

    print(f"  [fb2→pdf] Building PDF ({len(text):,} chars)…")

    font_name = "Helvetica"
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        if os.path.exists(candidate):
            try:
                pdfmetrics.registerFont(TTFont("BookFont", candidate))
                font_name = "BookFont"
                print(f"  [fb2→pdf] Using font: {candidate}")
            except Exception:
                pass
            break

    styles    = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "BookBody",
        fontName=font_name,
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
    )
    head_style = ParagraphStyle(
        "BookHead",
        fontName=font_name,
        fontSize=13,
        leading=18,
        spaceBefore=12,
        spaceAfter=6,
        textColor="#222222",
    )

    doc   = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm,  bottomMargin=2.5*cm,
    )
    story = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        if line == line.upper() and len(line) < 120 and len(line) > 2:
            safe = html_mod.escape(line)
            story.append(Paragraph(safe, head_style))
        else:
            safe = html_mod.escape(line)
            story.append(Paragraph(safe, body_style))

    try:
        doc.build(story)
        print(f"  [fb2→pdf] PDF saved: {pdf_path}")
        return True
    except Exception as exc:
        print(f"  [!] reportlab build failed: {exc}")
        return False


def convert_fb2_to_pdf(fb2_path: str) -> str | None:
    """
    Convert FB2 to PDF.
    Strategy:
      1. Calibre ebook-convert  (best quality)
      2. Native reportlab       (no external deps)
    Returns PDF path on success, None on failure.
    """
    pdf_path = os.path.splitext(fb2_path)[0] + ".pdf"
    print(f"  Converting FB2 → PDF: {os.path.basename(fb2_path)}")

    try:
        result = subprocess.run(
            ["ebook-convert", fb2_path, pdf_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
        )
        if result.returncode == 0 and os.path.exists(pdf_path):
            try: os.remove(fb2_path)
            except Exception: pass
            print(f"  [✓] Calibre conversion done: {os.path.basename(pdf_path)}")
            return pdf_path
        print(f"  [!] Calibre exited {result.returncode} — trying native fallback…")
    except FileNotFoundError:
        print("  [~] Calibre not found — using native FB2→PDF fallback…")
    except subprocess.TimeoutExpired:
        print("  [!] Calibre timed out — trying native fallback…")
    except Exception as exc:
        print(f"  [!] Calibre error: {exc} — trying native fallback…")

    if _fb2_to_pdf_native(fb2_path, pdf_path):
        try: os.remove(fb2_path)
        except Exception: pass
        return pdf_path

    print("  [✗] All FB2→PDF methods failed")
    try: os.remove(fb2_path)
    except Exception: pass
    return None



def handle_downloaded_file(fpath):
    if not fpath or not os.path.exists(fpath):
        print(f"  [!] File missing: {fpath}")
        return None
    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".pdf":
        print(f"  [✓] PDF ready: {fpath}")
        return fpath
    elif ext == ".fb2":
        print("  [~] FB2 — converting to PDF…")
        return convert_fb2_to_pdf(fpath)
    elif ext in (".epub", ".mobi", ".azw3", ".djvu"):
        pdf_path = os.path.splitext(fpath)[0] + ".pdf"
        print(f"  [~] {ext.upper()} — trying Calibre conversion…")
        try:
            result = subprocess.run(
                ["ebook-convert", fpath, pdf_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180,
            )
            if result.returncode == 0 and os.path.exists(pdf_path):
                try: os.remove(fpath)
                except Exception: pass
                print(f"  [✓] Converted: {os.path.basename(pdf_path)}")
                return pdf_path
        except FileNotFoundError:
            print(f"  [!] Calibre not found — cannot convert {ext}")
        except Exception as exc:
            print(f"  [!] Conversion error: {exc}")
        try: os.remove(fpath)
        except Exception: pass
        return None
    else:
        print(f"  [!] Unsupported format ({ext}) — deleting")
        try: os.remove(fpath)
        except Exception: pass
        return None


def flatten_torrent_file_to_output(path):
    fname = os.path.basename(path)
    dst   = os.path.join(OUTPUT_DIR, fname)
    if os.path.abspath(path) == os.path.abspath(dst):
        return dst
    print(f"  [torrent] Moving to books folder: {fname}")
    shutil.move(path, dst)
    parent = os.path.dirname(path)
    while os.path.abspath(parent) != os.path.abspath(OUTPUT_DIR):
        try:
            os.rmdir(parent)
            parent = os.path.dirname(parent)
        except Exception:
            break
    return dst



def extract_text_from_pdf(pdf_path):
    doc  = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text.strip()


def chunk_text(text, chunk_size=CHUNK_SIZE):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk: chunks.append(chunk)
            break
        while end > start and text[end] not in (" ", "\n", "\t"):
            end -= 1
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        start = end
    return chunks


def save_chunks_to_db(chunks, title, source_url):
    print(f"  Embedding {len(chunks)} chunks…")
    embeddings = embedding_model.encode(chunks, batch_size=32, show_progress_bar=False)
    records = [
        {
            "content": chunk,
            "embedding": emb.tolist(),
            "source_title": title,
            "source_url": source_url,
            "source_type": "web_scrape",
            "metadata": {"chunk_index": idx, "total_chunks": len(chunks), "chunk_size": len(chunk)},
            "category": "book",
            "topic": title,
        }
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    for i in range(0, len(records), 100):
        batch = records[i:i+100]
        supabase.table("rag_libgen").insert(batch).execute()
        print(f"    Inserted batch {i//100 + 1} ({len(batch)} rows)")
    print(f"  [✓] Saved: {title}")


def process_pdf(pdf_path, title, source_url):
    print(f"  Processing: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("  [!] No text extracted — skipping DB save")
        return
    print(f"  Extracted {len(text):,} chars")
    chunks = chunk_text(text)
    print(f"  {len(chunks)} chunks of ~{CHUNK_SIZE} chars")
    save_chunks_to_db(chunks, title, source_url)



def _scan_page(driver):
    try:
        html = driver.page_source
    except Exception:
        return None, None, None
    direct_url = None
    for pat in [r'https://[^"\'<>\s]+\.pdf', r'https://[^"\'<>\s]+\.fb2']:
        m = re.search(pat, html)
        if m:
            direct_url = m.group(0)
            break
    magnet = None
    m = re.search(r'magnet:\?[^"\'<>\s]+', html)
    if m:
        magnet = m.group(0)
    torrent_url = None
    m = re.search(r'https://[^"\'<>\s]+\.torrent', html)
    if m:
        torrent_url = m.group(0)
    return direct_url, magnet, torrent_url


def get_md5_page_links(driver, md5):
    url = f"{BASE_URL}/md5/{md5}"
    print(f"  Opening MD5 page: {url}")
    if not safe_get(driver, url):
        return None, None, None, None
    time.sleep(5)
    slow_url = None
    for link in driver.find_elements(By.TAG_NAME, "a"):
        try:
            href = link.get_attribute("href") or ""
            if "/slow_download/" in href:
                slow_url = href
                break
        except Exception:
            continue
    direct_url, magnet, torrent_url = _scan_page(driver)
    return slow_url, direct_url, magnet, torrent_url


def scrape_slow_page(driver, slow_url):
    print("  Opening slow-download page…")
    if not safe_get(driver, slow_url):
        return None, None, None
    slow_domain = urlparse(slow_url).netloc
    file_exts   = re.compile(r'\.(pdf|fb2|epub|djvu|mobi|azw3)', re.IGNORECASE)

    def _harvest():
        direct_url  = None
        magnet      = None
        torrent_url = None
        for a in driver.find_elements(By.TAG_NAME, "a"):
            try:
                href = (a.get_attribute("href") or "").strip()
                if not href:
                    continue
                if not direct_url and file_exts.search(href):
                    direct_url = href
                if not magnet and href.startswith("magnet:"):
                    magnet = href
                if not torrent_url and href.endswith(".torrent"):
                    torrent_url = href
            except Exception:
                pass
        return direct_url, magnet, torrent_url

    print(f"  Waiting up to {SLOW_PAGE_WAIT}s for download links to appear…")
    deadline = time.time() + SLOW_PAGE_WAIT
    while time.time() < deadline:
        time.sleep(2)
        d, m, t = _harvest()
        if d or m or t:
            print(f"  [slow] Link found: direct={d and d[:80]} magnet={bool(m)} torrent={bool(t)}")
            return d, m, t

    print("  [slow] No links yet — trying button clicks…")
    click_keywords = {"download", "get", "slow", "click", "partner", "libgen"}
    for btn in driver.find_elements(By.TAG_NAME, "button"):
        try:
            text = (btn.text or "").lower()
            if any(w in text for w in click_keywords):
                print(f"  [slow] Clicking button: '{btn.text.strip()[:60]}'")
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(3)
                d, m, t = _harvest()
                if d or m or t:
                    return d, m, t
        except Exception:
            pass

    for a in driver.find_elements(By.TAG_NAME, "a"):
        try:
            href        = (a.get_attribute("href") or "")
            link_domain = urlparse(href).netloc
            if slow_domain in link_domain or not link_domain:
                continue
            text = (a.text or "").lower()
            if any(w in text or w in href.lower() for w in click_keywords):
                print(f"  [slow] Clicking external link: '{a.text.strip()[:60]}' → {href[:80]}")
                driver.execute_script("arguments[0].click();", a)
                time.sleep(3)
                d, m, t = _harvest()
                if d or m or t:
                    return d, m, t
        except Exception:
            pass

    print("  [slow] No links via DOM — falling back to page source scan…")
    return _scan_page(driver)



def download_via_torrent(source, title_hint="book"):
    if not LIBTORRENT_AVAILABLE:
        print("  [!] libtorrent unavailable — skipping")
        return None
    print(f"  [torrent] Starting: {title_hint}")
    settings = {
        "listen_interfaces": "0.0.0.0:6881,[::]:6881",
        "alert_mask": lt.alert.category_t.all_categories,
    }
    ses    = lt.session(settings)
    handle = None
    try:
        if source.startswith("magnet:"):
            print(f"  [torrent] Magnet: {source[:80]}…")
            params           = lt.parse_magnet_uri(source)
            params.save_path = OUTPUT_DIR
            handle           = ses.add_torrent(params)
        else:
            tmp = os.path.join(OUTPUT_DIR, "_temp.torrent")
            print(f"  [torrent] Fetching .torrent file…")
            try:
                urllib.request.urlretrieve(source, tmp)
            except Exception as exc:
                print(f"  [!] Could not fetch .torrent: {exc}")
                return None
            ti = lt.torrent_info(tmp)
            try: os.remove(tmp)
            except Exception: pass
            params           = lt.add_torrent_params()
            params.ti        = ti
            params.save_path = OUTPUT_DIR
            handle           = ses.add_torrent(params)

        print("  [torrent] Waiting for metadata…", end="", flush=True)
        for _ in range(60):
            if handle.has_metadata():
                break
            time.sleep(1)
            print(".", end="", flush=True)
        else:
            print("\n  [!] Metadata timeout")
            ses.remove_torrent(handle)
            return None
        print(" OK")

        info       = handle.get_torrent_info()
        priorities = []
        targets    = []
        for idx in range(info.num_files()):
            fp = info.files().file_path(idx).lower()
            if fp.endswith(".pdf") or fp.endswith(".fb2"):
                priorities.append(lt.download_priority_t.default_priority)
                targets.append(idx)
                print(f"  [torrent] Queued: {info.files().file_path(idx)}")
            else:
                priorities.append(lt.download_priority_t.dont_download)

        if not targets:
            print("  [!] No PDF/FB2 in torrent")
            ses.remove_torrent(handle)
            return None

        handle.prioritize_files(priorities)
        elapsed = 0
        while elapsed < TORRENT_TIMEOUT:
            s = handle.status()
            print(
                f"  [torrent] {s.progress*100:5.1f}%  "
                f"{s.download_rate/1024:6.0f} KB/s  peers:{s.num_peers}  "
                f"{elapsed}s/{TORRENT_TIMEOUT}s   ",
                end="\r",
            )
            if s.state in (
                lt.torrent_status.states.seeding,
                lt.torrent_status.states.finished,
            ):
                print("\n  [torrent] Complete!")
                break
            time.sleep(TORRENT_POLL)
            elapsed += TORRENT_POLL
        else:
            print(f"\n  [!] Torrent timed out after {TORRENT_TIMEOUT}s")
            ses.remove_torrent(handle)
            return None

        ses.remove_torrent(handle)
        for idx in targets:
            rel      = info.files().file_path(idx)
            abs_path = os.path.join(OUTPUT_DIR, rel)
            if os.path.exists(abs_path):
                return flatten_torrent_file_to_output(abs_path)

        print("  [!] Torrent done but file not on disk")
        return None
    except Exception as exc:
        print(f"  [!] Torrent error: {exc}")
        if handle:
            try: ses.remove_torrent(handle)
            except Exception: pass
        return None



def try_download(direct_url, magnet, torrent_url, title):
    final_pdf  = None
    source_ref = None
    torrent_src = magnet or torrent_url
    if torrent_src:
        print("  [→] Trying torrent/magnet…")
        dl = download_via_torrent(torrent_src, title_hint=title)
        if dl:
            final_pdf  = handle_downloaded_file(dl)
            source_ref = torrent_src
    if not final_pdf and direct_url:
        print("  [→] Trying direct HTTP download…")
        dl = download_direct(direct_url, title_hint=title)
        if dl:
            final_pdf  = handle_downloaded_file(dl)
            source_ref = direct_url
    return final_pdf, source_ref



def main():
    df   = pd.read_csv(CSV_FILE)
    rows = df.head(NUM_BOOKS)
    print(f"Processing {len(rows)} book(s).")
    print(f"Books folder: {OUTPUT_DIR}\n")

    print("Launching Chrome…")
    try:
        driver = make_driver()
    except Exception as exc:
        sys.exit(f"[FATAL] Could not launch Chrome: {exc}")

    try:
        for i, row in rows.iterrows():
            title = str(row.get("title", "book"))[:80]
            md5   = str(row["md5_reported"]).strip()

            print("=" * 70)
            print(f"[{i+1}/{len(rows)}] {title}")
            print(f"  MD5: {md5}\n")

            try:
                slow_url, direct_url, magnet, torrent_url = get_md5_page_links(driver, md5)
            except Exception as exc:
                print(f"  [!] MD5 page error: {exc} — skipping")
                continue

            print(f"  Torrent : {torrent_url or '—'}")
            print(f"  Magnet  : {(magnet or '')[:80] or '—'}")
            print(f"  Direct  : {direct_url or '—'}")
            print(f"  Slow    : {slow_url or '—'}\n")

            final_pdf, source_ref = try_download(direct_url, magnet, torrent_url, title)

            if not final_pdf and slow_url:
                print("  [→] Nothing on MD5 page — opening slow-download page…")
                try:
                    sd_direct, sd_magnet, sd_torrent = scrape_slow_page(driver, slow_url)
                except Exception as exc:
                    print(f"  [!] Slow page error: {exc}")
                    sd_direct = sd_magnet = sd_torrent = None

                print(f"  Slow-page Direct  : {sd_direct or '—'}")
                print(f"  Slow-page Magnet  : {(sd_magnet or '')[:80] or '—'}")
                print(f"  Slow-page Torrent : {sd_torrent or '—'}\n")

                if sd_direct or sd_magnet or sd_torrent:
                    final_pdf, source_ref = try_download(sd_direct, sd_magnet, sd_torrent, title)

            if not final_pdf:
                print("  [!] All methods failed — skipping this book")
                continue

            try:
                process_pdf(final_pdf, title, source_ref or "unknown")
            except Exception as exc:
                print(f"  [!] DB error: {exc}")

            print(f"\n  Cooling down {DELAY_BETWEEN_BOOKS}s…\n")
            time.sleep(DELAY_BETWEEN_BOOKS)

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    print("=" * 70)
    print("DONE")


import psutil

def show_stats():
    ram  = psutil.virtual_memory()
    disk = shutil.disk_usage("/")
    print(f"  RAM  — used: {ram.used/1e9:.1f} GB / {ram.total/1e9:.1f} GB  ({ram.percent}%)")
    print(f"  Disk — used: {disk.used/1e9:.1f} GB / {disk.total/1e9:.1f} GB  ({disk.used/disk.total*100:.1f}%)")


if __name__ == "__main__":
    show_stats()
    main()