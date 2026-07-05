from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

_BILIBILI_BVID_RE = re.compile(r"/video/(BV[\w]+)", re.IGNORECASE)
_BILIBILI_PART_ID_RE = re.compile(r"_p(\d+)$", re.IGNORECASE)


def _host(source_url: str) -> str:
    return (urlparse(source_url).hostname or "").lower()


def is_douyin_host(host: str) -> bool:
    host = host.lower().removeprefix("www.")
    return host == "douyin.com" or host.endswith(".douyin.com")


def is_bilibili_host(host: str) -> bool:
    host = host.lower().removeprefix("www.")
    if host in ("b23.tv", "bili2233.cn"):
        return True
    return host == "bilibili.com" or host.endswith(".bilibili.com")


def is_supported_source_host(host: str) -> bool:
    return is_douyin_host(host) or is_bilibili_host(host)


def source_platform_label(source_url: str) -> str:
    host = _host(source_url)
    if is_douyin_host(host):
        return "Douyin"
    if is_bilibili_host(host):
        return "Bilibili"
    return "video"


def normalize_source_url(source_url: str) -> str:
    parsed = urlparse(source_url.strip())
    modal_id = parse_qs(parsed.query).get("modal_id", [None])[0]
    if is_douyin_host(parsed.netloc) and modal_id and modal_id.isdigit():
        return f"https://www.douyin.com/video/{modal_id}"
    return source_url.strip()


def is_douyin_user_profile_url(source_url: str) -> bool:
    if not is_douyin_host(_host(source_url)):
        return False
    return (urlparse(source_url).path or "").lower().startswith("/user/")


def extract_bilibili_bvid(url: str) -> str | None:
    match = _BILIBILI_BVID_RE.search(urlparse(url).path or "")
    return match.group(1) if match else None


def _bilibili_page_from_url(url: str) -> int | None:
    raw = parse_qs(urlparse(url).query).get("p", [None])[-1]
    if raw is None:
        return None
    try:
        page = int(raw)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _bilibili_page_from_entry(entry: dict, page_index: int | None) -> int | None:
    for candidate in (
        _bilibili_page_from_url(entry.get("url") or ""),
        _bilibili_page_from_url(entry.get("webpage_url") or ""),
        entry.get("playlist_index"),
        page_index,
    ):
        if candidate is None:
            continue
        try:
            page = int(candidate)
        except (TypeError, ValueError):
            continue
        if page > 0:
            return page

    entry_id = entry.get("id")
    if isinstance(entry_id, str):
        match = _BILIBILI_PART_ID_RE.search(entry_id)
        if match:
            return int(match.group(1))
    return None


def ensure_bilibili_part_url(
    url: str,
    source_url: str,
    *,
    entry: dict | None = None,
    page_index: int | None = None,
) -> str:
    """Ensure a Bilibili anthology URL includes ?p=N so yt-dlp downloads one part."""
    page = _bilibili_page_from_entry(entry or {}, page_index)
    if page is None:
        page = _bilibili_page_from_url(url)
    if page is None:
        return url

    parsed = urlparse(url)
    if is_bilibili_host(parsed.netloc) and _bilibili_page_from_url(url) == page:
        return url

    bvid = extract_bilibili_bvid(url) or extract_bilibili_bvid(source_url)
    if not bvid:
        return url
    return f"https://www.bilibili.com/video/{bvid}?p={page}"


def fallback_playlist_video_url(entry: dict, source_url: str, *, page_index: int | None = None) -> str:
    host = _host(source_url)
    for key in ("url", "webpage_url"):
        raw = entry.get(key)
        if not raw:
            continue
        if is_bilibili_host(host) or is_bilibili_host(_host(raw)):
            return ensure_bilibili_part_url(raw, source_url, entry=entry, page_index=page_index)
        return raw

    entry_id = entry.get("id")
    if is_bilibili_host(host):
        page = _bilibili_page_from_entry(entry, page_index)
        bvid = extract_bilibili_bvid(source_url)
        if not bvid and isinstance(entry_id, str) and entry_id.upper().startswith("BV"):
            bvid = entry_id.split("_", 1)[0]
        if bvid and page:
            return f"https://www.bilibili.com/video/{bvid}?p={page}"
        if bvid:
            return f"https://www.bilibili.com/video/{bvid}"

    if not entry_id:
        return source_url
    if is_douyin_host(host):
        return f"https://www.douyin.com/video/{entry_id}"
    if is_bilibili_host(host):
        return f"https://www.bilibili.com/video/{entry_id}"
    return source_url
