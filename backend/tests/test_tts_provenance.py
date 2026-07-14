from dv_backend.tts_provenance import (
    classify_clause_seam,
    detect_clause_seams,
    validate_segments_tts_provenance,
)
from dv_backend.text_sanitation import sanitize_spoken_text, punctuation_artifact_issues


def test_detect_verb_complement_seam() -> None:
    segments = [
        {"index": 71, "tts_spoken_text": "anh em suýt chạy gãy. . .", "text": "差点没把腿"},
        {"index": 72, "tts_spoken_text": "chân. Nhớ đãi một bữa.", "text": "腿打断兄弟记得一顿饭菜"},
    ]
    seams = detect_clause_seams(segments)
    assert seams
    assert seams[0]["left_index"] == 71
    assert "vi_verb_complement_split" in seams[0]["reasons"] or "cn_token_continuation" in seams[0]["reasons"]


def test_cn_token_complete_clauses_are_soft() -> None:
    seam = {
        "reasons": ["cn_token_continuation"],
        "left_text": "Muốn sống thì mau đặt bội đao xuống! Nghe chưa?",
        "right_text": "Dám làm thì đừng cầu xin tha mạng.",
    }
    assert classify_clause_seam(seam) == "soft"


def test_ellipsis_continuation_is_hard() -> None:
    seam = {
        "reasons": ["vi_ellipsis_continuation"],
        "left_text": "Vì sao lúc ra tay. . .",
        "right_text": "trên mặt ngài vẫn treo nụ cười đáng ghét ấy?",
    }
    assert classify_clause_seam(seam) == "hard"


def test_sanitize_orphan_dot_space() -> None:
    assert " . " not in sanitize_spoken_text("Quý. . tới thôn")
    assert "dot_space_dot" in punctuation_artifact_issues("Quý. . tới")


def test_provenance_rejects_missing_path() -> None:
    report = validate_segments_tts_provenance(
        [{"index": 36, "tts_spoken_text": "...một trăm chín mươi chín năm."}]
    )
    assert report["passed"] is False
    assert report["blocking"][0]["issues"] == ["missing_tts_path"]
