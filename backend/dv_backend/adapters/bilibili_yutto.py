from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

import httpx

from yutto.api.bangumi import get_bangumi_list
from yutto.api.ugc_video import get_ugc_video_list
from yutto.exceptions import NoAccessPermissionError, NotFoundError
from yutto.extractor.bangumi_batch import BangumiBatchExtractor
from yutto.extractor.ugc_video_batch import UgcVideoBatchExtractor
from yutto.utils.fetcher import Fetcher, FetcherContext, create_client

from ..errors import AppError
from ..models import ErrorInfo

_SHORTCUT_EXTRACTORS = (
    UgcVideoBatchExtractor(),
    BangumiBatchExtractor(),
)


def _video_dict(
    *,
    video_id: str,
    title: str,
    url: str,
    duration: int | None = None,
    thumbnail: str | None = None,
) -> dict:
    return {
        "id": video_id,
        "title": title,
        "url": url,
        "duration": duration,
        "thumbnail": thumbnail,
    }


async def _resolve_url(source_url: str, *, limit: int) -> dict:
    ctx = FetcherContext()
    url = source_url.strip()
    for extractor in _SHORTCUT_EXTRACTORS:
        matched, url = extractor.resolve_shortcut(url)
        if matched:
            break

    async with create_client(
        cookies=ctx.cookies,
        trust_env=ctx.trust_env,
        proxy=ctx.proxy,
    ) as client:
        try:
            url = await Fetcher.get_redirected_url(ctx, client, url)
        except httpx.InvalidURL as exc:
            raise AppError(
                422,
                ErrorInfo(
                    code="INVALID_SOURCE_URL",
                    message="The Bilibili URL is invalid.",
                    action="Paste a valid bilibili.com video or bangumi link.",
                    detail=str(exc),
                ),
            ) from exc
        except httpx.UnsupportedProtocol as exc:
            raise AppError(
                422,
                ErrorInfo(
                    code="INVALID_SOURCE_URL",
                    message="The Bilibili URL uses an unsupported protocol.",
                    action="Use an https://www.bilibili.com link.",
                    detail=str(exc),
                ),
            ) from exc

        ugc_extractor = UgcVideoBatchExtractor()
        if ugc_extractor.match(url):
            try:
                ugc_list = await get_ugc_video_list(ctx, client, ugc_extractor.avid)
            except (NotFoundError, NoAccessPermissionError) as exc:
                raise AppError(
                    404,
                    ErrorInfo(
                        code="BILIBILI_RESOLVE_FAILED",
                        message="Failed to resolve Bilibili video metadata.",
                        action="Check that the video is public and the URL is correct.",
                        detail=exc.message,
                    ),
                ) from exc

            bvid = str(ugc_list["avid"].as_bvid())
            base_url = f"https://www.bilibili.com/video/{bvid}"
            series_title = ugc_list["title"]
            videos = []
            for page in ugc_list["pages"][:limit]:
                page_title = page["name"] or f"P{page['id']}"
                title = page_title if len(ugc_list["pages"]) == 1 else f"{series_title} - {page_title}"
                thumb = page.get("metadata", {}).get("thumb") if isinstance(page.get("metadata"), dict) else None
                videos.append(
                    _video_dict(
                        video_id=f"{bvid}-p{page['id']}",
                        title=title,
                        url=f"{base_url}?p={page['id']}",
                        thumbnail=thumb,
                    )
                )
            return {"is_playlist": len(videos) > 1, "videos": videos}

        bangumi_extractor = BangumiBatchExtractor()
        if bangumi_extractor.match(url):
            await bangumi_extractor._parse_ids(ctx, client)
            bangumi_list = await get_bangumi_list(ctx, client, bangumi_extractor.season_id)
            pages = [item for item in bangumi_list["pages"] if not item["is_section"]][:limit]
            videos = []
            for item in pages:
                episode_id = str(item["episode_id"])
                thumb = item.get("metadata", {}).get("thumb") if isinstance(item.get("metadata"), dict) else None
                videos.append(
                    _video_dict(
                        video_id=episode_id,
                        title=f"{bangumi_list['title']} - {item['name']}",
                        url=f"https://www.bilibili.com/bangumi/play/ep{episode_id}",
                        thumbnail=thumb,
                    )
                )
            return {"is_playlist": len(videos) > 1, "videos": videos}

    raise AppError(
        422,
        ErrorInfo(
            code="INVALID_SOURCE_URL",
            message="Only Bilibili video and bangumi links are supported.",
            action="Paste a bilibili.com/video, bilibili.com/bangumi, or b23.tv share link.",
        ),
    )


def resolve_bilibili_url(source_url: str, *, limit: int = 20) -> dict:
    return asyncio.run(_resolve_url(source_url, limit=limit))


def build_yutto_download_command(video_url: str, download_dir: Path) -> list[str]:
    download_dir.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable,
        "-m",
        "yutto",
        "download",
        "-d",
        str(download_dir),
        "--output-format",
        "mp4",
        "--no-danmaku",
        "--no-subtitle",
        "--video-only",
        "--no-color",
        "--no-progress",
        video_url,
    ]


def finalize_yutto_download(download_dir: Path, output_mp4: Path) -> None:
    candidates = sorted(download_dir.glob("**/*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise AppError(
            500,
            ErrorInfo(
                code="DOWNLOAD_FAILED",
                message="yutto finished without producing an MP4 file.",
                action="Retry the job or verify the Bilibili link is still available.",
            ),
        )
    source = candidates[0]
    if output_mp4.exists():
        output_mp4.unlink()
    shutil.move(str(source), str(output_mp4))
    shutil.rmtree(download_dir, ignore_errors=True)


def ensure_yutto_available() -> None:
    try:
        import yutto  # noqa: F401
    except ImportError as exc:
        raise AppError(
            500,
            ErrorInfo(
                code="YUTTO_NOT_AVAILABLE",
                message="The yutto downloader package is not installed.",
                action="Reinstall the backend environment or rebuild the portable runtime.",
                detail=str(exc),
            ),
        ) from exc
