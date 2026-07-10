# Settings Feature Inventory

Auto-saved on change. Default tab: **Lồng tiếng (tts)**.

## Shell
| Element | Notes |
|---------|-------|
| Title + subtitle | Cài đặt |
| Readiness chips | Dịch, Lồng tiếng, Engine label |
| Tab bar (5) | download, translation, audio, tts, subtitles |

## 1. Tải video (`download`)
- cookies_file (text)
- Cập nhật yt-dlp (button) + notice

## 2. Dịch thuật (`translation`)
- translation_backend: google_free | gemini | openai
- gemini_translation_model (if gemini)
- openai: base URL, API key, model select + refresh
- Disclosure: Gemini API key pool (add/remove/label keys)

## 3. Âm thanh (`audio`)
- vad_engine: silero | silencedetect
- vad_false_positive_filter_enabled, vad_energy_filter_enabled
- vad_energy_min_vocal_ratio (range, if energy filter)
- Disclosure: Silero / FFmpeg / Sparse ASR advanced fields

## 4. Lồng tiếng (`tts`)
- Engine picker: omnivoice, edge_tts, google_tts, gemini_tts
- Per-engine config panels (conditional)
- Preview sidebar: textarea + Nghe thử + audio player
- Disclosure: Khớp thời lượng nâng cao (5 fields)

## 5. Phụ đề (`subtitles`)
- subtitles_enabled + font size, position, colors, opacity, padding, margin

## Health evaluation (nav dots)
- translation: gemini/openai key missing → attention
- tts: google/gemini key, omnivoice ref_text → attention
- subtitles: disabled → neutral
