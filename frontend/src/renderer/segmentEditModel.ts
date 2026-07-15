import type {
  EditableSegmentDto,
  SegmentEditDiff,
  SegmentEditSaveRequest,
} from "../shared/contracts";

export type LocalSegmentRow = {
  client_id: string;
  segment_id: string | null;
  start_ms: number;
  end_ms: number;
  spoken_text: string;
  source_text: string | null;
  origin: "pipeline" | "user" | null;
  source_segment_index: number | null;
};

export type PreviewMode =
  | { mode: "available"; source_segment_index: number }
  | { mode: "needs_tts"; note: string }
  | { mode: "timing_only"; source_segment_index: number; note: string }
  | { mode: "unavailable"; note?: string };

export type EditorStatus = "clean" | "dirty" | "saved_unexported";

let clientIdCounter = 0;

export function createClientId(): string {
  clientIdCounter += 1;
  return `client-${Date.now().toString(36)}-${clientIdCounter}-${Math.random().toString(36).slice(2, 8)}`;
}

export function msToSecondsInput(ms: number): string {
  return (ms / 1000).toFixed(3);
}

export function parseSecondsInputToMs(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) return null;
  return Math.round(parsed * 1000);
}

export function defaultTimingAfter(previousEndMs: number | null): { start_ms: number; end_ms: number } {
  const start_ms = previousEndMs != null && previousEndMs >= 0 ? previousEndMs : 0;
  return { start_ms, end_ms: start_ms + 1000 };
}

export function createLocalRow(partial: {
  start_ms: number;
  end_ms: number;
  spoken_text?: string;
  segment_id?: string | null;
  source_text?: string | null;
  origin?: "pipeline" | "user" | null;
  source_segment_index?: number | null;
  client_id?: string;
}): LocalSegmentRow {
  const segment_id = partial.segment_id ?? null;
  return {
    client_id: partial.client_id ?? (segment_id ? segment_id : createClientId()),
    segment_id,
    start_ms: partial.start_ms,
    end_ms: partial.end_ms,
    spoken_text: partial.spoken_text ?? "",
    source_text: partial.source_text ?? null,
    origin: partial.origin ?? (segment_id ? null : "user"),
    source_segment_index: partial.source_segment_index ?? null,
  };
}

export function draftRowsFromServer(segments: EditableSegmentDto[]): LocalSegmentRow[] {
  return segments.map((seg) =>
    createLocalRow({
      client_id: seg.segment_id,
      segment_id: seg.segment_id,
      start_ms: seg.start_ms,
      end_ms: seg.end_ms,
      spoken_text: seg.spoken_text,
      source_text: seg.source_text,
      origin: seg.origin,
      source_segment_index: seg.source_segment_index,
    }),
  );
}

export type SemanticRow = {
  segment_id: string | null;
  start_ms: number;
  end_ms: number;
  spoken_text: string;
};

export function semanticSnapshot(rows: LocalSegmentRow[]): SemanticRow[] {
  return rows.map((row) => ({
    segment_id: row.segment_id,
    start_ms: row.start_ms,
    end_ms: row.end_ms,
    spoken_text: row.spoken_text,
  }));
}

function semanticFromServer(segments: EditableSegmentDto[]): SemanticRow[] {
  return segments.map((seg) => ({
    segment_id: seg.segment_id,
    start_ms: seg.start_ms,
    end_ms: seg.end_ms,
    spoken_text: seg.spoken_text,
  }));
}

export function isDraftDirty(rows: LocalSegmentRow[], baseline: EditableSegmentDto[]): boolean {
  const local = semanticSnapshot(rows);
  const server = semanticFromServer(baseline);
  if (local.length !== server.length) return true;
  for (let i = 0; i < local.length; i += 1) {
    const a = local[i];
    const b = server[i];
    if (
      a.segment_id !== b.segment_id ||
      a.start_ms !== b.start_ms ||
      a.end_ms !== b.end_ms ||
      a.spoken_text !== b.spoken_text
    ) {
      return true;
    }
  }
  return false;
}

export function isDraftValid(rows: LocalSegmentRow[]): boolean {
  if (rows.length === 0) return false;
  for (const row of rows) {
    if (row.start_ms < 0) return false;
    if (row.end_ms <= row.start_ms) return false;
    if (!row.spoken_text.trim()) return false;
  }
  return true;
}

export function insertRowAfter(rows: LocalSegmentRow[], afterClientId: string | null): LocalSegmentRow[] {
  const next = [...rows];
  let insertAt = next.length;
  let previousEnd: number | null = next.length > 0 ? next[next.length - 1].end_ms : null;

  if (afterClientId != null) {
    const idx = next.findIndex((row) => row.client_id === afterClientId);
    if (idx >= 0) {
      insertAt = idx + 1;
      previousEnd = next[idx].end_ms;
    }
  }

  const timing = defaultTimingAfter(previousEnd);
  next.splice(insertAt, 0, createLocalRow({ ...timing, spoken_text: "" }));
  return next;
}

export function moveRow(rows: LocalSegmentRow[], clientId: string, direction: "up" | "down"): LocalSegmentRow[] {
  const idx = rows.findIndex((row) => row.client_id === clientId);
  if (idx < 0) return rows;
  const target = direction === "up" ? idx - 1 : idx + 1;
  if (target < 0 || target >= rows.length) return rows;
  const next = [...rows];
  const [item] = next.splice(idx, 1);
  next.splice(target, 0, item);
  return next;
}

export function removeRow(rows: LocalSegmentRow[], clientId: string): LocalSegmentRow[] {
  return rows.filter((row) => row.client_id !== clientId);
}

export function rowsToSavePayload(rows: LocalSegmentRow[]): SegmentEditSaveRequest["segments"] {
  return rows.map((row) => ({
    segment_id: row.segment_id,
    start_ms: row.start_ms,
    end_ms: row.end_ms,
    spoken_text: row.spoken_text,
  }));
}

function findBaseline(row: LocalSegmentRow, baseline: EditableSegmentDto[]): EditableSegmentDto | undefined {
  if (!row.segment_id) return undefined;
  return baseline.find((seg) => seg.segment_id === row.segment_id);
}

export function previewModeForRow(row: LocalSegmentRow, baseline: EditableSegmentDto[]): PreviewMode {
  const needsTtsNote = "Âm thanh sẽ được tạo khi Export";
  const timingNote = "Thời gian chưa được áp dụng; đang nghe bản cũ";

  if (!row.segment_id) {
    return { mode: "needs_tts", note: needsTtsNote };
  }

  const original = findBaseline(row, baseline);
  if (!original) {
    return { mode: "needs_tts", note: needsTtsNote };
  }

  const textChanged = row.spoken_text !== original.spoken_text;
  if (textChanged) {
    return { mode: "needs_tts", note: needsTtsNote };
  }

  const timingChanged = row.start_ms !== original.start_ms || row.end_ms !== original.end_ms;
  const sourceIndex = original.source_segment_index;

  if (sourceIndex == null) {
    return { mode: "unavailable", note: needsTtsNote };
  }

  if (timingChanged) {
    return {
      mode: "timing_only",
      source_segment_index: sourceIndex,
      note: timingNote,
    };
  }

  return { mode: "available", source_segment_index: sourceIndex };
}

export function editorStatusLabel(
  rows: LocalSegmentRow[],
  baseline: EditableSegmentDto[],
  diff: SegmentEditDiff,
): EditorStatus {
  if (isDraftDirty(rows, baseline)) return "dirty";
  if (diff.has_changes) return "saved_unexported";
  return "clean";
}

export function canSaveDraft(args: {
  rows: LocalSegmentRow[];
  baseline: EditableSegmentDto[];
  saving: boolean;
  exporting: boolean;
}): boolean {
  if (args.saving || args.exporting) return false;
  if (!isDraftDirty(args.rows, args.baseline)) return false;
  return isDraftValid(args.rows);
}

export function canExportDraft(args: {
  rows: LocalSegmentRow[];
  baseline: EditableSegmentDto[];
  diff: SegmentEditDiff;
  saving: boolean;
  exporting: boolean;
}): boolean {
  if (args.saving || args.exporting) return false;
  if (isDraftDirty(args.rows, args.baseline)) return false;
  return args.diff.has_changes === true;
}

export function reconcileRowsAfterSave(
  _localRows: LocalSegmentRow[],
  savedSegments: EditableSegmentDto[],
): LocalSegmentRow[] {
  return savedSegments.map((seg) =>
    createLocalRow({
      client_id: seg.segment_id,
      segment_id: seg.segment_id,
      start_ms: seg.start_ms,
      end_ms: seg.end_ms,
      spoken_text: seg.spoken_text,
      source_text: seg.source_text,
      origin: seg.origin,
      source_segment_index: seg.source_segment_index,
    }),
  );
}

export function isSegmentEditConflictCode(code: string | undefined | null): boolean {
  if (!code) return false;
  return code === "SEGMENT_EDIT_VERSION_CONFLICT" || code === "plan_version_conflict";
}

export function isSegmentExportInProgressCode(code: string | undefined | null): boolean {
  return code === "segment_export_in_progress";
}

export function statusDisplayText(status: EditorStatus): string {
  if (status === "dirty") return "Có thay đổi chưa lưu";
  if (status === "saved_unexported") return "Đã lưu, chưa xuất lại";
  return "Đã đồng bộ";
}
