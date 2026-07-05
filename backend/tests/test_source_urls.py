import pytest

from dv_backend.source_urls import (
    ensure_bilibili_part_url,
    extract_bilibili_bvid,
    fallback_playlist_video_url,
    is_bilibili_host,
    is_douyin_host,
    is_supported_source_host,
    normalize_source_url,
)

_BV_URL = "https://www.bilibili.com/video/BV1MEJw6qE8b/"


@pytest.mark.parametrize(
    "host,expected",
    [
        ("www.bilibili.com", True),
        ("space.bilibili.com", True),
        ("m.bilibili.com", True),
        ("b23.tv", True),
        ("bili2233.cn", True),
        ("www.douyin.com", True),
        ("v.douyin.com", True),
        ("example.com", False),
        ("youtube.com", False),
    ],
)
def test_is_supported_source_host(host: str, expected: bool) -> None:
    assert is_supported_source_host(host) is expected


def test_normalize_bilibili_url_is_unchanged() -> None:
    assert normalize_source_url(_BV_URL) == _BV_URL


def test_fallback_playlist_video_url_prefers_entry_url() -> None:
    entry = {"id": "BV123", "url": "https://www.bilibili.com/video/BV123?p=2"}
    assert fallback_playlist_video_url(entry, _BV_URL) == entry["url"]


def test_fallback_playlist_video_url_builds_bilibili_part_from_index() -> None:
    entry = {"ie_key": "BiliBili"}
    assert (
        fallback_playlist_video_url(entry, _BV_URL, page_index=3)
        == "https://www.bilibili.com/video/BV1MEJw6qE8b?p=3"
    )


def test_fallback_playlist_video_url_uses_playlist_index() -> None:
    entry = {"playlist_index": 2}
    assert (
        fallback_playlist_video_url(entry, _BV_URL)
        == "https://www.bilibili.com/video/BV1MEJw6qE8b?p=2"
    )


def test_fallback_playlist_video_url_parses_part_from_id() -> None:
    entry = {"id": "BV1MEJw6qE8b_p4"}
    assert (
        fallback_playlist_video_url(entry, _BV_URL)
        == "https://www.bilibili.com/video/BV1MEJw6qE8b?p=4"
    )


def test_ensure_bilibili_part_url_adds_missing_p() -> None:
    assert (
        ensure_bilibili_part_url("https://www.bilibili.com/video/BV1MEJw6qE8b", _BV_URL, page_index=2)
        == "https://www.bilibili.com/video/BV1MEJw6qE8b?p=2"
    )


def test_extract_bilibili_bvid() -> None:
    assert extract_bilibili_bvid(_BV_URL) == "BV1MEJw6qE8b"


def test_host_classifiers() -> None:
    assert is_bilibili_host("b23.tv")
    assert not is_bilibili_host("douyin.com")
    assert is_douyin_host("www.douyin.com")


def test_normalize_douyin_modal_id() -> None:
    url = "https://www.douyin.com/jingxuan?modal_id=1234567890"
    assert normalize_source_url(url) == "https://www.douyin.com/video/1234567890"
