"""
Image download and validation helpers.
"""

import hashlib
import threading
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from config import MAX_WHITE_RATIO, MIN_SHORT_SIDE, REQUEST_PROXIES

# Reuse a single session and let the OS or TUN handle routing by default.
_SESSION = requests.Session()
if REQUEST_PROXIES:
    _SESSION.proxies.update(REQUEST_PROXIES)
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})


def is_valid_image(data: bytes) -> bool:
    if len(data) < 20_000:
        return False
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
        w, h = img.size
        if min(w, h) < MIN_SHORT_SIDE:
            return False

        thumb = img.resize((32, 32))
        pixels = list(thumb.getdata())
        n = len(pixels)

        white = sum(1 for r, g, b in pixels if r > 225 and g > 225 and b > 225)
        if white / n > MAX_WHITE_RATIO:
            return False

        dark = sum(1 for r, g, b in pixels if r < 15 and g < 15 and b < 15)
        if dark / n > 0.88:
            return False

        import statistics

        r_vals = [p[0] for p in pixels]
        if statistics.stdev(r_vals) < 8:
            return False

        return True
    except Exception:
        return False


def img_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def download_one(url: str, save_path: Path,
                 seen_hashes: set, lock: threading.Lock) -> bool:
    try:
        r = _SESSION.get(url, timeout=20)
        if r.status_code != 200:
            return False

        data = r.content
        if not is_valid_image(data):
            return False

        h = img_hash(data)
        with lock:
            if h in seen_hashes:
                return False
            seen_hashes.add(h)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(data)
        return True
    except Exception:
        return False
