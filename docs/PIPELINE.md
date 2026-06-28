# DouyinVietnamizer — Pipeline Workflow

Tài liệu này tổng hợp pipeline chạy job hiện tại của dự án, dựa trên source code trong
`backend/dv_backend/` (commit hiện tại trên `main`).

## 1. Tổng quan thành phần

| Thành phần                | File / Hàm                                              | Vai trò                                                                                          |
|---------------------------|---------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| FastAPI app               | `backend/dv_backend/api.py` — `create_app()`            | Khởi tạo config, database, settings, JobService, JobRunner; mount các endpoint `/api/jobs/*`.     |
| JobService                | `backend/dv_backend/jobs.py`                            | Tạo / import / rerun / redub / delete job; đồng bộ `job_steps` với `PIPELINE_STEPS`.              |
| JobRunner                 | `backend/dv_backend/runner.py` — `JobRunner._run_job`   | Chạy từng step trong thread riêng; cập nhật DB; xử lý cancel / lỗi.                              |
| Pipeline steps            | `backend/dv_backend/pipeline.py` — `*_step()`           | Thực thi logic của từng step; ghi checkpoint JSON sau khi xong.                                   |
| Checkpoint storage        | `backend/dv_backend/checkpoints.py`                     | Định nghĩa `PIPELINE_STEPS` + helpers `load_checkpoint` / `save_checkpoint` (atomic write).      |
| Cancellation registry     | `runner.py` — `running_processes`, `cancelled_jobs`      | Track Popen theo `job_id` để kill khi user cancel.                                                |
| Subprocess wrapper        | `pipeline.py` — `run_subprocess_with_cancel`            | Chạy lệnh ngoài (yt-dlp, ffmpeg), respect cancel + timeout.                                      |

## 2. Danh sách 12 bước của pipeline

`PIPELINE_STEPS` (`backend/dv_backend/checkpoints.py:7-20`):

```
resolve → download → extract_audio → vad → asr →
normalize_segments → translate → tts →
duration_repair → mix → render → qc
```

| # | Step                | Hàm (`pipeline.py`)                | I/O chính                                                                                                                  | Checkpoint key                  |
|---|---------------------|------------------------------------|----------------------------------------------------------------------------------------------------------------------------|---------------------------------|
| 1 | resolve             | `resolve_step` (line 250)          | `yt-dlp --dump-single-json` → liệt kê video (playlist hoặc single). Nếu chỉ 1 video, auto-select.                          | `resolve`                       |
| 2 | download            | `download_step` (line 343)         | `yt-dlp` tải `artifacts/original.mp4`. Throw `NO_VIDEO_SELECTED` (trạng thái `waiting_for_selection`) nếu playlist chưa chọn. | `download`                      |
| 3 | extract_audio       | `extract_audio_step` (line 402)    | ffmpeg tách 48kHz stereo `original_48k.wav` + 16kHz mono `audio_16k.wav`.                                                  | `extract_audio`                 |
| 4 | vad                 | `vad_step` (line 468)              | ffmpeg `silencedetect=-30dB:d=0.5` trên 16k → `speech_regions`, `total_duration`.                                          | `vad`                           |
| 5 | asr                 | `asr_step` (line 560)              | Qwen3-ASR (vendor `dv_vendor`) trên `audio_16k.wav`, `language="Chinese"`, `include_alignment=True`. Lưu `segments` + `aligned_units`. | `asr` (schema_version = 2)      |
| 6 | normalize_segments  | `normalize_segments_step` (line 689)| `_split_long_asr_segments_with_vad` (>20s) → cắt overlap, gán `duration_budget`, `index`.                                    | `normalize_segments`            |
| 7 | translate           | `translate_step` (line 838)        | `GoogleFreeTranslator` hoặc `GeminiKeyPool` (chọn qua setting `translation_backend`) → dịch `text` → `translation` cho từng segment + `title_vi`. | `translate`                     |
| 8 | tts                 | `tts_step` (line 1055)             | `create_tts_adapter` (VoxCPM2) synthesize từng segment → `tts_raw_*.wav` → convert 48k/2ch → `tts_*.wav`. Lưu `tts_duration`. | `tts`                           |
| 9 | duration_repair     | `duration_repair_step` (line 1115) | Nếu TTS dài hơn `duration_budget`: (a) OpenAI chat `llm_shorten` (nếu có key), (b) `atempo` time-stretch, (c) `apad+atrim` exact. | `duration_repair`               |
| 10| mix                 | `mix_step` (line 1273)             | Nhét WAV TTS đã repair vào khung `original_48k.wav` (48k/2ch) → `narration.wav`; ffmpeg `sidechaincompress` duck BGM → `mixed.wav`. Copy ra `output/vietnamese_narration.wav`. | `mix`                           |
| 11| render              | `render_step` (line 1381)          | ffmpeg `loudnorm` → `normalized.wav`; render `original.mp4` + normalized audio (H.264 superfast/AAC 192k) → `output/dubbed.mp4`. Nếu `subtitles_enabled`, ghi `subtitles.ass` và add `ass` filter. | `render`                        |
| 12| qc                  | `qc_step` (line 1486)              | Tổng hợp `repaired_count`, `shortened_count`, `stretched_count`, `warnings`; ghi `artifacts/qc_report.json` + `qc_report.html`. | `qc` (schema_version = 2)       |

## 3. Vòng đời của một job

### 3.1 Tạo job
- Endpoint: `POST /api/jobs` (`api.py:324`) → `JobService.create(source_url)` (`jobs.py:72`).
  - Validate host (`is_supported_source_host` — Douyin / Bilibili).
  - Insert row `jobs (status='queued')` + N row `job_steps (status='pending')` cho 12 step.
  - Trả về `Job`, đồng thời `runner.start_job(job.id)` chạy thread `_run_job`.

- Import file local: `POST /api/jobs/import` (`api.py:331`) → `JobService.create_imported()` (`jobs.py:99`).
  - Copy file upload vào `artifacts/original.mp4`.
  - Auto-mark `resolve` + `download` là `completed`; `current_step = 'extract_audio'`.
  - File list cố định: `SKIP_STEPS_FOR_IMPORT = ("resolve", "download")`.

### 3.2 Chạy job — `JobRunner._run_job` (`runner.py:70-189`)

Với mỗi `step_name` trong `PIPELINE_STEPS` (lặp tuần tự):

1. `is_cancelled(job_id)` → `break` nếu user cancel.
2. Đọc `job_steps.status` từ DB. Nếu `completed` → `continue` (resume-friendly).
3. Cập nhật `job_steps.status='running'`, set `started_at`, set `jobs.current_step`.
4. Gọi `getattr(pipeline, f"{step_name}_step")(job_id, config, db, runner)`.
5. Khi thành công: `job_steps.status='completed'`, `completed_at` được set.
6. Khi lỗi:
   - `AppError(code='NO_VIDEO_SELECTED')` → reset step về `pending`, `jobs.status='waiting_for_selection'`, dừng (chờ user chọn video rồi gọi `select-video`).
   - `AppError` khác → `job_steps.status='failed'`, `jobs.status='failed'`, ghi `events (level='error')`.
   - `Exception` lạ → `UNEXPECTED_ERROR`, traceback, ghi `events`.

Sau vòng lặp: nếu không cancel → `jobs.status='completed'`, `current_step=NULL`, ghi `events JOB_COMPLETED`.

### 3.3 Cancel job
- Endpoint `POST /api/jobs/{id}/cancel` (`api.py:395`) → `JobRunner.cancel_job()` (`runner.py:38`).
  - Thêm `job_id` vào `cancelled_jobs`, kill `Popen` đang chạy (nếu có).
  - DB: `jobs.status='failed'`, `last_error_code='CANCELLED'`; mọi `job_steps.status='running'` → `failed`.

### 3.4 User chọn video (playlist case)
- Endpoint `POST /api/jobs/{id}/select-video` (`api.py:417`):
  - Đọc checkpoint `resolve`, lấy `videos[index]`, lưu `selected_video` vào checkpoint.
  - Reset `job_steps.download` về `pending`, set `jobs.status='queued'`, update `title`.
  - `runner.start_job(job_id)` chạy lại từ `download`.

### 3.5 Rerun / Redub
- `POST /api/jobs/{id}/rerun` (`api.py:400`) → `JobService.rerun(job_id, keep_steps)` (`jobs.py:225`):
  - `keep_steps` phải là **prefix liên tục** của `PIPELINE_STEPS` (validate `INVALID_RERUN_KEEP_PREFIX`).
  - Reset tất cả step từ vị trí `first_reset_idx` về `pending`, xóa checkpoint file tương ứng.
  - `jobs.status='queued'`, clear error.
  - `runner.start_job` chạy lại.

- `POST /api/jobs/{id}/redub` (`api.py:406`) → `JobService.redub()` (`jobs.py:313`): rerun với `keep_steps = PIPELINE_STEPS[:translate_index+1]` (giữ tới `translate`, rerun `tts` trở đi).

### 3.6 Restart sau khi app chết
- `create_app` gọi `JobService.reconcile_interrupted()` (`jobs.py:212`):
  - Mọi `jobs.status='running'` → `interrupted`; `job_steps.status='running'` → `failed` với `error_code='APP_INTERRUPTED'`.

### 3.7 Delete
- `DELETE /api/jobs/{id}` (`api.py:412`) → `JobService.delete()` (`jobs.py:318`): cấm nếu đang `running`; xóa row + `rmtree` thư mục `jobs/{id}`.

## 4. Subprocess & cancel

`pipeline.run_subprocess_with_cancel(cmd, job_id, runner, timeout)` (`pipeline.py:96-142`):
- Tạo `Popen` với `CREATE_NO_WINDOW` (Windows).
- `runner.register_process(job_id, proc)` để runner có thể kill.
- `proc.communicate(timeout=...)`. Nếu `timeout` → kill + `AppError(PROCESS_TIMEOUT)`.
- Khi return code ≠ 0 → `subprocess.CalledProcessError`; mỗi step bắt riêng và map sang `AppError` có `code` + `action`.

## 5. Trạng thái DB quan trọng

- `jobs.status`: `queued | running | waiting_for_selection | failed | completed | interrupted`.
- `job_steps.status`: `pending | running | completed | failed`.
- `job_steps.position` = index trong `PIPELINE_STEPS` (đồng bộ qua `_sync_pipeline_steps` ở startup, `jobs.py:33`).
- `events`: log error / info theo job.
- `settings` (key/value JSON): config cho ASR model, translation backend, TTS (VoxCPM2 ref audio / clone mode), `subtitles_enabled`, `exact_timing_*`, `openai_api_key`, `gemini_api_keys`, `cookies_browser`, v.v.

## 6. Tóm tắt luồng dữ liệu

```
URL (Douyin/Bilibili)
  └─ resolve ──▶ videos + selected_video
       └─ download ──▶ artifacts/original.mp4
            └─ extract_audio ──▶ original_48k.wav + audio_16k.wav
                 ├─ vad ──▶ speech_regions, total_duration
                 └─ asr ──▶ segments (text zh, timings, aligned_units)
                      └─ normalize_segments ──▶ indexed segments + duration_budget
                           └─ translate ──▶ segments[].translation + title_vi
                                └─ tts (VoxCPM2) ──▶ artifacts/tts/tts_*.wav + tts_duration
                                     └─ duration_repair ──▶ tts_repaired_*.wav + repaired_method
                                          └─ mix ──▶ output/vietnamese_narration.wav + mixed.wav
                                               └─ render ──▶ output/dubbed.mp4 (+ subtitles.ass)
                                                    └─ qc ──▶ artifacts/qc_report.json + .html
```

## 7. Điểm đáng chú ý / có thể tinh gọn

- `_run_job` lặp 12 step nhưng `pipeline` import qua `getattr(pipeline, f"{step_name}_step")` mỗi lần — không đáng kể nhưng có thể cache map `step_name → func` một lần.
- `mix_step` chứa thao tác I/O thuần Python (lặp segment, ghi WAV) xen với ffmpeg. Có thể chuyển phần ghép narration sang ffmpeg filter (concat/delays) để giảm logic Python.
- `qc_step` hiện chỉ thống kê — không chặn job failed khi có warning. Hành vi này là cố ý.
- `create_tts_adapter` được gọi **mỗi segment** trong `_synthesize_segment_tts`; nếu TTS adapter giữ state nặng (model load), nên cache instance giữa các segment.
- `yt_dlp_cookie_args` đọc `settings.cookies_browser` mỗi lần gọi — chấp nhận được nhưng có thể cache ở `SettingsService` nếu cần.
