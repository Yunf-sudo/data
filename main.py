"""
Main collection entrypoint for the FengShui image dataset.

Run:
    python main.py
"""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import CATEGORIES, DOWNLOAD_WORKERS, OUTPUT_DIR, TARGET_PER_CLASS
from crawlers import fetch_all_sources
from downloader import download_one, img_hash

STATE_FILE_NAME = "_collector_state.json"
MAX_ATTEMPTS = 5
EXTRA_SUFFIXES = [
    " interior photo",
    " real home",
    " interior design",
    " home decor",
    " inside view",
    " modern interior",
    " interior scene",
    " living space",
]

_PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def print_bar(label: str, count: int, target: int = TARGET_PER_CLASS) -> None:
    filled = min(count, target) * 30 // target
    bar = "#" * filled + "-" * (30 - filled)
    if count >= target:
        flag = "OK"
    elif count >= target * 0.6:
        flag = ".."
    else:
        flag = "!!"
    log(f"  {flag} {label:<22} [{bar}] {count:>5}/{target}")


def state_path() -> Path:
    return Path(OUTPUT_DIR) / STATE_FILE_NAME


def load_state() -> dict:
    path = state_path()
    if not path.exists():
        return {"version": 1, "active_label": None, "categories": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log(f"[warn] Failed to read {path.name}, starting with a fresh state.")
        return {"version": 1, "active_label": None, "categories": {}}


def save_state(state: dict) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = time.time()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def image_files_in(dir_path: Path):
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        yield from dir_path.glob(pattern)


def count_existing_images(dir_path: Path) -> int:
    return sum(1 for _ in image_files_in(dir_path))


def load_existing_hashes() -> set[str]:
    seen: set[str] = set()
    total_files = 0

    for label in CATEGORIES:
        src_dir = Path(OUTPUT_DIR) / label
        if not src_dir.exists():
            continue
        for img_path in image_files_in(src_dir):
            total_files += 1
            try:
                seen.add(img_hash(img_path.read_bytes()))
            except Exception:
                continue

    if total_files:
        log(f"[resume] Loaded {len(seen)} hashes from {total_files} existing images.")
    return seen


def get_category_state(state: dict, label: str, cfg: dict) -> dict:
    cat = state["categories"].setdefault(
        label,
        {
            "attempt": 1,
            "query_index": 0,
            "queries": list(cfg["queries"]),
            "attempt_gained": 0,
            "expanded_until": 2,
            "completed": False,
            "count": 0,
        },
    )
    cat.setdefault("attempt", 1)
    cat.setdefault("query_index", 0)
    cat.setdefault("queries", list(cfg["queries"]))
    cat.setdefault("attempt_gained", 0)
    cat.setdefault("expanded_until", 2)
    cat.setdefault("completed", False)
    cat.setdefault("count", 0)
    return cat


def maybe_expand_queries(cat_state: dict, cfg: dict) -> None:
    attempt = max(1, int(cat_state.get("attempt", 1)))
    expanded_until = int(cat_state.get("expanded_until", 2))
    queries = list(cat_state.get("queries", cfg["queries"]))

    while expanded_until < attempt and attempt > 2:
        for base_query in cfg["queries"]:
            queries.append(base_query + random.choice(EXTRA_SUFFIXES))
        queries = list(dict.fromkeys(queries))
        expanded_until += 1

    cat_state["queries"] = queries
    cat_state["expanded_until"] = expanded_until


def collect_category(
    label: str,
    cfg: dict,
    seen_hashes: set[str],
    hash_lock: threading.Lock,
    state: dict,
) -> int:
    save_dir = Path(OUTPUT_DIR) / label
    save_dir.mkdir(parents=True, exist_ok=True)

    cat_state = get_category_state(state, label, cfg)
    state["active_label"] = label

    count = count_existing_images(save_dir)
    cat_state["count"] = count
    if count >= TARGET_PER_CLASS:
        cat_state["completed"] = True
        cat_state["query_index"] = 0
        save_state(state)
        log(f"[skip] {label}: already has {count} images.")
        return count

    if cat_state.get("completed") and count < TARGET_PER_CLASS:
        cat_state["completed"] = False

    maybe_expand_queries(cat_state, cfg)
    save_state(state)

    log("\n" + "=" * 60)
    log(
        f"[collect] [{cfg['idx']:02d}] {label} "
        f"(existing {count}/{TARGET_PER_CLASS}, "
        f"resume attempt {cat_state['attempt']}, query {cat_state['query_index'] + 1})"
    )
    log("=" * 60)

    while count < TARGET_PER_CLASS:
        maybe_expand_queries(cat_state, cfg)
        queries = cat_state["queries"]

        if cat_state["query_index"] >= len(queries):
            gained = int(cat_state.get("attempt_gained", 0))
            if gained == 0:
                log(f"[retry] {label}: attempt {cat_state['attempt']} found no new images.")
                if cat_state["attempt"] >= MAX_ATTEMPTS:
                    break
                save_state(state)
                time.sleep(20)

            cat_state["attempt"] += 1
            cat_state["query_index"] = 0
            cat_state["attempt_gained"] = 0
            maybe_expand_queries(cat_state, cfg)
            save_state(state)
            continue

        attempt = cat_state["attempt"]
        query_index = cat_state["query_index"]
        query = queries[query_index]
        log(
            f"[query] {label} | attempt {attempt} | "
            f"{query_index + 1}/{len(queries)} | {query}"
        )

        candidates = fetch_all_sources(query)
        random.shuffle(candidates)

        gained_this_query = 0
        futures = {}
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            for i, url in enumerate(candidates):
                if count + len(futures) >= int(TARGET_PER_CLASS * 1.05):
                    break
                ts = int(time.time() * 1000)
                file_path = save_dir / f"{label}_{cfg['idx']:02d}_{ts}_{i}.jpg"
                future = executor.submit(download_one, url, file_path, seen_hashes, hash_lock)
                futures[future] = file_path

            for future in as_completed(futures):
                if future.result():
                    count += 1
                    gained_this_query += 1
                    log(f"    [ok] {label}: {count}/{TARGET_PER_CLASS}")
                    if count >= TARGET_PER_CLASS:
                        break

        cat_state["count"] = count
        cat_state["attempt_gained"] = int(cat_state.get("attempt_gained", 0)) + gained_this_query
        cat_state["query_index"] += 1
        save_state(state)

        if count >= TARGET_PER_CLASS:
            break

        time.sleep(random.uniform(1.0, 2.5))

    cat_state["count"] = count
    cat_state["completed"] = count >= TARGET_PER_CLASS
    if cat_state["completed"]:
        cat_state["query_index"] = 0
    save_state(state)

    log(f"[done] {label}: {count} images")
    return count


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    state = load_state()
    if state.get("active_label"):
        log(f"[resume] Last active category: {state['active_label']}")

    seen_hashes = load_existing_hashes()
    hash_lock = threading.Lock()
    results: dict[str, int] = {}

    log("=" * 60)
    log("FengShui dataset collector")
    log(f"Target: {len(CATEGORIES)} classes x {TARGET_PER_CLASS} images")
    log(f"Output: {Path(OUTPUT_DIR).absolute()}")
    log("=" * 60)

    try:
        for label, cfg in CATEGORIES.items():
            results[label] = collect_category(label, cfg, seen_hashes, hash_lock, state)
    except KeyboardInterrupt:
        save_state(state)
        log("\n[stop] Interrupted. Progress has been saved and can be resumed.")
        raise

    state["active_label"] = None
    save_state(state)

    log("\n" + "=" * 60)
    log("Collection summary")
    log("=" * 60)

    total = 0
    short = []
    for label in CATEGORIES:
        count = results.get(label, count_existing_images(Path(OUTPUT_DIR) / label))
        total += count
        print_bar(label, count)
        if count < TARGET_PER_CLASS * 0.7:
            short.append((label, count))

    log(f"\nTotal images: {total}")
    log(f"Target images: {TARGET_PER_CLASS * len(CATEGORIES)}")

    if short:
        log("\nClasses still below 70% of target:")
        for label, count in short:
            log(f"  - {label}: {count}")

    log(f"\nSaved to: {Path(OUTPUT_DIR).absolute()}")
    log("Next step: run auto_label.py to generate RT-DETR labels.")


if __name__ == "__main__":
    main()
