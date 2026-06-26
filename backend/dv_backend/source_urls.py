from __future__ import annotations

from urllib.parse import parse_qs, urlparse


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


def normalize_source_url(source_url: str) -> str:
    parsed = urlparse(source_url)
    modal_id = parse_qs(parsed.query).get("modal_id", [None])[0]
    if is_douyin_host(parsed.netloc) and modal_id and modal_id.isdigit():
        return f"https://www.douyin.com/video/{modal_id}"
    return source_url


def is_douyin_user_profile_url(source_url: str) -> bool:
    if not is_douyin_host(_host(source_url)):
        return False
    return (urlparse(source_url).path or "").lower().startswith("/user/")


def fallback_playlist_video_url(entry: dict, source_url: str) -> str:
    if entry.get("url"):
        return entry["url"]
    if entry.get("webpage_url"):
        return entry["webpage_url"]
    entry_id = entry.get("id")
    if not entry_id:
        return source_url
    host = _host(source_url)
    if is_douyin_host(host):
        return f"https://www.douyin.com/video/{entry_id}"
    if is_bilibili_host(host):
        return f"https://www.bilibili.com/video/{entry_id}"
    return source_url
