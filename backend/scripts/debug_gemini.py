"""One-off Gemini API debugger. Run: uv run python scripts/debug_gemini.py"""
import json
import os
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path


def load_key() -> str:
    data_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "DouyinVietnamizer"
    db = sqlite3.connect(data_dir / "app.db")
    db.row_factory = sqlite3.Row
    rows = {
        r["key"]: json.loads(r["value"])
        for r in db.execute("SELECT key, value FROM settings")
    }
    for item in rows.get("gemini_api_keys", []):
        if isinstance(item, dict) and item.get("key"):
            return str(item["key"])
    raise SystemExit("No gemini API key in settings")


def call(api_key: str, model: str, payload: dict) -> tuple[int, str | None, str | None]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode())
            text = "".join(
                part.get("text", "")
                for part in data["candidates"][0]["content"]["parts"]
            )
            return resp.status, text[:300], None
    except urllib.error.HTTPError as exc:
        return exc.code, None, exc.read().decode("utf-8", errors="replace")[:800]


def main() -> None:
    key = load_key()
    masked = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
    print(f"Using key: {masked}\n")

    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": key},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        models = json.loads(resp.read().decode())["models"]

    flash3 = [
        m["name"]
        for m in models
        if "flash" in m["name"].lower() and "3" in m["name"]
    ]
    print("Flash-3 models visible to this API key:")
    for name in flash3:
        print(f"  {name}")
    print()

    simple = {"contents": [{"parts": [{"text": "Say hi in Vietnamese, one word only"}]}]}
    json_payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            'Translate this JSON array from zh-CN to vi. '
                            'Return only a JSON array of translated strings in the same order.\n'
                            '["你好"]'
                        )
                    }
                ],
            }
        ],
        "generationConfig": {"responseMimeType": "application/json"},
    }

    for model in [
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
    ]:
        print(f"=== {model} ===")
        for label, payload in [("simple", simple), ("json_mode", json_payload)]:
            code, text, err = call(key, model, payload)
            print(f"  [{label}] HTTP {code}")
            if text:
                print(f"    text: {text!r}")
            if err:
                try:
                    parsed = json.loads(err)
                    print(f"    error: {json.dumps(parsed, ensure_ascii=False)}")
                except json.JSONDecodeError:
                    print(f"    error: {err}")
        print()


if __name__ == "__main__":
    main()
