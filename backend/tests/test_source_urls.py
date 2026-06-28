import pytest

from dv_backend.source_urls import (
    fallback_playlist_video_url,
    is_bilibili_host,
    is_douyin_host,
    is_douyin_user_profile_url,
    is_supported_source_host,
    normalize_source_url,
)


@pytest.mark.parametrize(
    "host,expected",
    [
        ("www.douyin.com", True),
        ("v.douyin.com", True),
        ("www.bilibili.com", True),
        ("space.bilibili.com", True),
        ("m.bilibili.com", True),
        ("b23.tv", True),
        ("bili2233.cn", True),
        ("example.com", False),
        ("youtube.com", False),
    ],
)
def test_is_supported_source_host(host: str, expected: bool) -> None:
    assert is_supported_source_host(host) is expected


def test_normalize_douyin_jingxuan_modal_url() -> None:
    assert normalize_source_url(
        "https://www.douyin.com/jingxuan?modal_id=7639476837437699301"
    ) == "https://www.douyin.com/video/7639476837437699301"


def test_normalize_bilibili_url_is_unchanged() -> None:
    url = "https://www.bilibili.com/video/BV1MEJw6qE8b/"
    assert normalize_source_url(url) == url


def test_is_douyin_user_profile_url() -> None:
    url = (
        "https://www.douyin.com/user/"
        "MS4wLjABAAAAOnRpvxiasUeDLCX4WG94yZ3LA6ogPP7MJ6rNzi7bFy8m6QrRR9orTshL80q-1cUc"
    )
    assert is_douyin_user_profile_url(url) is True
    assert is_douyin_user_profile_url("https://www.douyin.com/video/123") is False
    assert is_douyin_user_profile_url("https://www.bilibili.com/video/BV123") is False


def test_fallback_playlist_video_url_prefers_entry_url() -> None:
    entry = {"id": "BV123", "url": "https://www.bilibili.com/video/BV123"}
    assert fallback_playlist_video_url(entry, "https://www.bilibili.com/video/BV123") == entry["url"]


def test_fallback_playlist_video_url_builds_bilibili_url() -> None:
    entry = {"id": "BV1MEJw6qE8b"}
    assert (
        fallback_playlist_video_url(entry, "https://www.bilibili.com/video/BV1MEJw6qE8b/")
        == "https://www.bilibili.com/video/BV1MEJw6qE8b"
    )


def test_host_classifiers() -> None:
    assert is_douyin_host("v.douyin.com")
    assert not is_douyin_host("bilibili.com")
    assert is_bilibili_host("b23.tv")
    assert not is_bilibili_host("douyin.com")
