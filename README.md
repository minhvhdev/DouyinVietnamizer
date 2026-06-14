# Douyin Vietnamizer Portable Edition

Windows-first desktop application for producing Vietnamese dubbed Douyin videos.

## Current status

The default CPU pipeline is implemented end to end:

1. Resolve and download a Douyin video.
2. Extract audio, detect speech, and transcribe Chinese with whisper.cpp.
3. Translate to Vietnamese with the free Google Translate adapter.
4. Synthesize Vietnamese speech with `edge-tts`.
5. Repair timing, mix audio, render `dubbed.mp4`, and produce JSON/HTML QC reports.

The real Douyin URL
`https://www.douyin.com/jingxuan?modal_id=7639476837437699301` has passed all
twelve steps with 90 translated and dubbed segments.

## Development

Requirements: Node.js 20+ and Python 3.11+.

Development also needs FFmpeg and yt-dlp on `PATH`, plus whisper.cpp CPU and a
multilingual model under:

```text
vendor/whisper.cpp/cpu/whisper-cli.exe
vendor/whisper.cpp/models/ggml-base.bin
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/dev.ps1
```

Run verification:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test.ps1
```

Development state defaults to `%LOCALAPPDATA%\DouyinVietnamizer`. Set `DV_DATA_DIR` to override it.

## Vendor runtime

`vendor/manifest.json` declares FFmpeg, yt-dlp, whisper.cpp CPU, optional whisper.cpp Vulkan, and optional Piper. Customer and packaged execution only accept executables bundled under `vendor/`.

Development may explicitly allow tools installed on `%PATH%` by setting `DV_ALLOW_PATH_TOOLS=1`; `scripts/dev.ps1` sets this flag for the development backend. PATH-resolved tools always produce a runtime warning so they cannot be mistaken for a complete customer build.

The Runtime panel shows storage, SQLite, manifest, and executable probe results. A missing required CPU tool blocks new-job creation. Optional Vulkan or Piper failures only produce warnings.

## Privacy and limitations

- Douyin URLs and downloaded media are processed locally, but Google Translate
  receives transcript text and Edge TTS receives translated Vietnamese text.
- Browser cookies are optional and are only passed to yt-dlp when selected.
- Douyin may change its site or require authentication, which can break a URL.
- Qwen3-ASR, Vulkan acceleration, and a final customer installer remain optional
  release work; the verified baseline is whisper.cpp CPU.
