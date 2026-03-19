"""
纯爬虫数据源模块 —— 走 Clash 代理访问国外网站
来源：
  1. Bing Image Search  <- 主力，覆盖最广
  2. Houzz              <- 室内设计图，质量最高
  3. Flickr             <- 开放图库，室内摄影多
  4. Dezeen             <- 建筑/室内设计媒体
"""

from __future__ import annotations

import re
import time
import random
import json
import requests
from html import unescape
from typing import Optional, List
from urllib.parse import quote_plus
from config import REQUEST_PROXIES

# ── Clash 代理 ────────────────────────────────────────────────────────

_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


def _dedupe(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _extract_bing_media_urls(html_text: str) -> List[str]:
    urls = []
    for pat in [
        r'"murl"\s*:\s*"(https?://[^"]+?\.(?:jpg|jpeg|png|webp))"',
        r'murl&quot;:&quot;(https?://[^&]+)',
        r'mediaurl=(https%3a%2f%2f[^"&]+)',
    ]:
        for raw in re.findall(pat, html_text):
            url = unescape(raw)
            if url.startswith("https%3a%2f%2f"):
                url = requests.utils.unquote(url)
            if re.search(r"\.(?:jpg|jpeg|png|webp)(?:$|\?)", url, re.IGNORECASE):
                urls.append(url)
    return _dedupe(urls)


def _headers(referer: str = "") -> dict:
    h = {
        "User-Agent": random.choice(_UA_POOL),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }
    if referer:
        h["Referer"] = referer
    return h


def _get(url: str, referer: str = "", timeout: int = 20) -> Optional[requests.Response]:
    try:
        kwargs = {
            "headers": _headers(referer),
            "timeout": timeout,
        }
        if REQUEST_PROXIES:
            kwargs["proxies"] = REQUEST_PROXIES
        r = requests.get(url, **kwargs)
        if r.status_code == 200:
            return r
    except Exception:
        pass
    return None


# =====================================================================
#  1. Bing Image Search
# =====================================================================

def fetch_bing(query: str, target: int = 120) -> List[str]:
    urls = []
    first = 1

    while len(urls) < target:
        search_url = (
            "https://www.bing.com/images/search?q=" + quote_plus(query) +
            "&form=HDRSC2&first=" + str(first) +
            "&tsc=ImageHoverTitle&qft=+filterui:imagesize-large+filterui:photo-photo"
        )
        r = _get(search_url, referer="https://www.bing.com/")
        if not r:
            break

        found = _extract_bing_media_urls(r.text)
        if not found:
            break

        for u in found:
            if any(x in u.lower() for x in ["thumb", "small", "icon", "60x", "80x", "100x"]):
                continue
            urls.append(u)

        if len(found) < 20:
            break
        first += 28
        time.sleep(random.uniform(1.0, 2.0))

    return _dedupe(urls)[:target]


# =====================================================================
#  2. Houzz
# =====================================================================

def fetch_houzz(query: str, target: int = 80) -> List[str]:
    urls = set()
    slug = query.strip().lower().replace(" ", "-")

    endpoints = [
        "https://www.houzz.com/photos/query/" + slug,
        "https://www.houzz.com/ideabooks/query/" + slug,
    ]

    for ep in endpoints:
        if len(urls) >= target:
            break
        r = _get(ep, referer="https://www.houzz.com/")
        if not r:
            time.sleep(2)
            continue

        for pat in [
            r'"url"\s*:\s*"(https://st\.hzcdn\.com/simgs/[^"]+)"',
            r'data-src="(https://st\.hzcdn\.com/simgs/[^"]+)"',
            r'src="(https://st\.hzcdn\.com/simgs/[^"]+)"',
        ]:
            for raw in re.findall(pat, r.text):
                hd = re.sub(r'_\d+_\d+\.(jpg|jpeg)', r'_1000_666.\1', raw)
                urls.add(hd)

        time.sleep(random.uniform(2.0, 3.5))

    if len(urls) < target:
        fallback = fetch_bing(f"site:houzz.com {query}", target=max(target * 2, 40))
        for raw in fallback:
            if "hzcdn.com" not in raw:
                continue
            hd = re.sub(r'_\d+_\d+\.(jpg|jpeg)', r'_1000_666.\1', raw)
            urls.add(hd)
            if len(urls) >= target:
                break

    return list(urls)[:target]


# =====================================================================
#  3. Flickr
# =====================================================================

def fetch_flickr(query: str, target: int = 80) -> List[str]:
    urls = []
    page = 1

    while len(urls) < target and page <= 5:
        api_url = (
            "https://www.flickr.com/search/?text=" + quote_plus(query) +
            "&media=photos&page=" + str(page) +
            "&sort=relevance&license=4%2C5%2C6%2C9%2C10&view_all=1"
        )
        r = _get(api_url, referer="https://www.flickr.com/")
        if not r:
            break

        match = re.search(
            r'"photos"\s*:\s*\{[^}]*"photo"\s*:\s*(\[.*?\])',
            r.text, re.DOTALL
        )
        if match:
            try:
                photos = json.loads(match.group(1))
                for p in photos:
                    farm   = p.get("farm", "")
                    server = p.get("server", "")
                    pid    = p.get("id", "")
                    secret = p.get("secret", "")
                    if farm and server and pid and secret:
                        urls.append(
                            "https://farm" + str(farm) + ".staticflickr.com/" +
                            str(server) + "/" + str(pid) + "_" + str(secret) + "_b.jpg"
                        )
            except Exception:
                pass

        if not urls:
            found = re.findall(
                r'(?:src=|displayUrl\\":|url\\":)"?(?:https?:)?(//live\.staticflickr\.com/\d+/\d+_[a-z0-9]+_[a-z]\.jpg)',
                r.text
            )
            for raw in found:
                url = "https:" + raw
                # Try to upgrade smaller inline sizes to a larger Flickr variant.
                url = re.sub(r'_[a-z]\.jpg$', '_b.jpg', url)
                urls.append(url)

        page += 1
        time.sleep(random.uniform(1.5, 2.5))

    return _dedupe(urls)[:target]


# =====================================================================
#  4. Dezeen
# =====================================================================

def fetch_dezeen(query: str, target: int = 40) -> List[str]:
    urls = []
    page = 1

    while len(urls) < target and page <= 4:
        search_url = (
            "https://www.dezeen.com/?s=" + quote_plus(query) +
            "&page=" + str(page)
        )
        r = _get(search_url, referer="https://www.dezeen.com/")
        if not r:
            break

        found = re.findall(
            r'https://static\.dezeen\.com/uploads/\d{4}/\d{2}/[^"\'<>\s]+?\.(?:jpg|jpeg|webp)',
            r.text
        )
        for raw in found:
            hd = re.sub(r'-\d+x\d+\.', '-1704x958.', raw)
            urls.append(hd)

        page += 1
        time.sleep(random.uniform(1.5, 2.5))

    seen = set()
    result = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result[:target]


# =====================================================================
#  统一接口
# =====================================================================

def fetch_all_sources(
    query: str,
    bing_n: int = 120,
    houzz_n: int = 80,
    flickr_n: int = 60,
    dezeen_n: int = 40,
) -> List[str]:
    all_urls = []

    print("    Bing...", end=" ", flush=True)
    b = fetch_bing(query, target=bing_n)
    print(str(len(b)) + "张", end="  ", flush=True)
    all_urls.extend(b)

    print("Houzz...", end=" ", flush=True)
    h = fetch_houzz(query, target=houzz_n)
    print(str(len(h)) + "张", end="  ", flush=True)
    all_urls.extend(h)

    print("Flickr...", end=" ", flush=True)
    f = fetch_flickr(query, target=flickr_n)
    print(str(len(f)) + "张", end="  ", flush=True)
    all_urls.extend(f)

    print("Dezeen...", end=" ", flush=True)
    d = fetch_dezeen(query, target=dezeen_n)
    print(str(len(d)) + "张")
    all_urls.extend(d)

    seen = set()
    result = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result
