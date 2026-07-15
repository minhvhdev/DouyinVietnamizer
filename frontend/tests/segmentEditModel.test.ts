import { describe, expect, test } from "vitest";

import type { SegmentEditPlanResponse, SegmentEditDiff } from "../src/shared/contracts";
import {
  canExportDraft,
  canSaveDraft,
  createLocalRow,
  defaultTimingAfter,
  draftRowsFromServer,
  editorStatusLabel,
  insertRowAfter,
  isDraftDirty,
  isDraftValid,
  moveRow,
  msToSecondsInput,
  parseSecondsInputToMs,
  previewModeForRow,
  reconcileRowsAfterSave,
  removeRow,
  rowsToSavePayload,
  semanticSnapshot,
} from "../src/renderer/segmentEditModel";

const emptyDiff: SegmentEditDiff = {
  has_changes: false,
  structural_changed: false,
  deltas: [],
  requires_tts_segment_ids: [],
  requires_duration_check_segment_ids: [],
  reusable_tts_segment_ids: [],
  deleted_segment_ids: [],
};

const samplePlan: SegmentEditPlanResponse = {
  schema_version: 1,
  plan_version: 2,
  applied_plan_version: 2,
  draft_segments: [
    {
      segment_id: "seg-a",
      start_ms: 0,
      end_ms: 1000,
      spoken_text: "Xin chào",
      source_text: "你好",
      origin: "pipeline",
      source_segment_index: 0,
    },
    {
      segment_id: "seg-b",
      start_ms: 1200,
      end_ms: 2500,
      spoken_text: "Tạm biệt",
      source_text: null,
      origin: "pipeline",
      source_segment_index: 1,
    },
  ],
  diff: emptyDiff,
};

describe("time helpers", () => {
  test("converts ms to seconds with 3 decimals and back", () => {
    expect(msToSecondsInput(1234)).toBe("1.234");
    expect(parseSecondsInputToMs("1.234")).toBe(1234);
    expect(parseSecondsInputToMs("0")).toBe(0);
    expect(parseSecondsInputToMs("")).toBeNull();
    expect(parseSecondsInputToMs("abc")).toBeNull();
  });
});

describe("draftRowsFromServer", () => {
  test("maps server segments to local rows with client_id = segment_id", () => {
    const rows = draftRowsFromServer(samplePlan.draft_segments);
    expect(rows).toHaveLength(2);
    expect(rows[0].client_id).toBe("seg-a");
    expect(rows[0].segment_id).toBe("seg-a");
    expect(rows[0].start_ms).toBe(0);
    expect(rows[0].spoken_text).toBe("Xin chào");
    expect(rows[1].client_id).toBe("seg-b");
  });
});

describe("dirty detection", () => {
  test("is clean when order+ids+timing+text match server snapshot", () => {
    const rows = draftRowsFromServer(samplePlan.draft_segments);
    expect(isDraftDirty(rows, samplePlan.draft_segments)).toBe(false);
  });

  test("detects text, timing, order, and structural changes", () => {
    const baseline = samplePlan.draft_segments;
    const rows = draftRowsFromServer(baseline);

    rows[0].spoken_text = "Đã sửa";
    expect(isDraftDirty(rows, baseline)).toBe(true);

    const timing = draftRowsFromServer(baseline);
    timing[0].end_ms = 1500;
    expect(isDraftDirty(timing, baseline)).toBe(true);

    const reordered = draftRowsFromServer(baseline);
    reordered.reverse();
    expect(isDraftDirty(reordered, baseline)).toBe(true);

    const withNew = insertRowAfter(draftRowsFromServer(baseline), null);
    expect(isDraftDirty(withNew, baseline)).toBe(true);
  });

  test("semanticSnapshot ignores client_id for comparison shape", () => {
    const snap = semanticSnapshot(draftRowsFromServer(samplePlan.draft_segments));
    expect(snap[0]).toEqual({
      segment_id: "seg-a",
      start_ms: 0,
      end_ms: 1000,
      spoken_text: "Xin chào",
    });
  });
});

describe("validation", () => {
  test("rejects negative start, end<=start, and blank text", () => {
    const rows = draftRowsFromServer(samplePlan.draft_segments);
    expect(isDraftValid(rows)).toBe(true);

    rows[0].start_ms = -1;
    expect(isDraftValid(rows)).toBe(false);

    rows[0].start_ms = 0;
    rows[0].end_ms = 0;
    expect(isDraftValid(rows)).toBe(false);

    rows[0].end_ms = 1000;
    rows[0].spoken_text = "   ";
    expect(isDraftValid(rows)).toBe(false);
  });

  test("allows overlapping timings", () => {
    const rows = draftRowsFromServer(samplePlan.draft_segments);
    rows[1].start_ms = 500;
    rows[1].end_ms = 1500;
    expect(isDraftValid(rows)).toBe(true);
  });
});

describe("row mutations", () => {
  test("defaultTimingAfter uses previous end then +1000ms", () => {
    expect(defaultTimingAfter(2500)).toEqual({ start_ms: 2500, end_ms: 3500 });
    expect(defaultTimingAfter(null)).toEqual({ start_ms: 0, end_ms: 1000 });
  });

  test("insertRowAfter appends at end when afterClientId is null", () => {
    const rows = draftRowsFromServer(samplePlan.draft_segments);
    const next = insertRowAfter(rows, null);
    expect(next).toHaveLength(3);
    expect(next[2].segment_id).toBeNull();
    expect(next[2].client_id).toMatch(/^client-/);
    expect(next[2].start_ms).toBe(2500);
    expect(next[2].end_ms).toBe(3500);
    expect(next[2].spoken_text).toBe("");
  });

  test("insertRowAfter inserts after a specific client_id", () => {
    const rows = draftRowsFromServer(samplePlan.draft_segments);
    const next = insertRowAfter(rows, "seg-a");
    expect(next).toHaveLength(3);
    expect(next[1].segment_id).toBeNull();
    expect(next[1].start_ms).toBe(1000);
    expect(next[2].segment_id).toBe("seg-b");
  });

  test("moveRow and removeRow use client_id not index identity", () => {
    const rows = draftRowsFromServer(samplePlan.draft_segments);
    const moved = moveRow(rows, "seg-b", "up");
    expect(moved.map((r) => r.segment_id)).toEqual(["seg-b", "seg-a"]);

    const removed = removeRow(rows, "seg-a");
    expect(removed.map((r) => r.segment_id)).toEqual(["seg-b"]);
  });

  test("createLocalRow never reuses array index as client_id", () => {
    const a = createLocalRow({ start_ms: 0, end_ms: 1000, spoken_text: "" });
    const b = createLocalRow({ start_ms: 0, end_ms: 1000, spoken_text: "" });
    expect(a.client_id).not.toBe(b.client_id);
    expect(a.client_id.startsWith("client-")).toBe(true);
  });
});

describe("save payload", () => {
  test("omits client_id and sends null segment_id for new rows", () => {
    const rows = insertRowAfter(draftRowsFromServer(samplePlan.draft_segments), null);
    rows[2].spoken_text = "Mới";
    const payload = rowsToSavePayload(rows);
    expect(payload).toEqual([
      { segment_id: "seg-a", start_ms: 0, end_ms: 1000, spoken_text: "Xin chào" },
      { segment_id: "seg-b", start_ms: 1200, end_ms: 2500, spoken_text: "Tạm biệt" },
      { segment_id: null, start_ms: 2500, end_ms: 3500, spoken_text: "Mới" },
    ]);
    expect(JSON.stringify(payload)).not.toContain("client_id");
  });
});

describe("preview mode", () => {
  test("unchanged existing row can preview via source_segment_index", () => {
    const rows = draftRowsFromServer(samplePlan.draft_segments);
    expect(previewModeForRow(rows[0], samplePlan.draft_segments)).toEqual({
      mode: "available",
      source_segment_index: 0,
    });
  });

  test("text change or new row disables preview", () => {
    const baseline = samplePlan.draft_segments;
    const changed = draftRowsFromServer(baseline);
    changed[0].spoken_text = "Khác";
    expect(previewModeForRow(changed[0], baseline)).toEqual({
      mode: "needs_tts",
      note: "Âm thanh sẽ được tạo khi Export",
    });

    const added = createLocalRow({ start_ms: 0, end_ms: 1000, spoken_text: "Mới" });
    expect(previewModeForRow(added, baseline).mode).toBe("needs_tts");
  });

  test("timing-only change allows optional preview with note", () => {
    const baseline = samplePlan.draft_segments;
    const rows = draftRowsFromServer(baseline);
    rows[0].end_ms = 1500;
    expect(previewModeForRow(rows[0], baseline)).toEqual({
      mode: "timing_only",
      source_segment_index: 0,
      note: "Thời gian chưa được áp dụng; đang nghe bản cũ",
    });
  });
});

describe("status / save / export gates", () => {
  test("editorStatusLabel reflects clean, dirty, and saved-unexported", () => {
    const rows = draftRowsFromServer(samplePlan.draft_segments);
    expect(editorStatusLabel(rows, samplePlan.draft_segments, emptyDiff)).toBe("clean");

    rows[0].spoken_text = "Sửa";
    expect(editorStatusLabel(rows, samplePlan.draft_segments, emptyDiff)).toBe("dirty");

    const saved = draftRowsFromServer(samplePlan.draft_segments);
    expect(
      editorStatusLabel(saved, samplePlan.draft_segments, { ...emptyDiff, has_changes: true }),
    ).toBe("saved_unexported");
  });

  test("canSave only when dirty and valid and not busy", () => {
    const baseline = samplePlan.draft_segments;
    const rows = draftRowsFromServer(baseline);
    expect(canSaveDraft({ rows, baseline, saving: false, exporting: false })).toBe(false);

    rows[0].spoken_text = "Sửa";
    expect(canSaveDraft({ rows, baseline, saving: false, exporting: false })).toBe(true);
    expect(canSaveDraft({ rows, baseline, saving: true, exporting: false })).toBe(false);

    rows[0].spoken_text = "  ";
    expect(canSaveDraft({ rows, baseline, saving: false, exporting: false })).toBe(false);
  });

  test("canExport requires clean local draft and backend diff.has_changes", () => {
    const baseline = samplePlan.draft_segments;
    const rows = draftRowsFromServer(baseline);
    expect(
      canExportDraft({
        rows,
        baseline,
        diff: { ...emptyDiff, has_changes: true },
        saving: false,
        exporting: false,
      }),
    ).toBe(true);

    rows[0].spoken_text = "dirty";
    expect(
      canExportDraft({
        rows,
        baseline,
        diff: { ...emptyDiff, has_changes: true },
        saving: false,
        exporting: false,
      }),
    ).toBe(false);

    const clean = draftRowsFromServer(baseline);
    expect(
      canExportDraft({
        rows: clean,
        baseline,
        diff: emptyDiff,
        saving: false,
        exporting: false,
      }),
    ).toBe(false);
  });
});

describe("reconcile after save", () => {
  test("maps new server ids onto local temp client rows by position", () => {
    const local = insertRowAfter(draftRowsFromServer(samplePlan.draft_segments), null);
    local[2].spoken_text = "Mới";
    const savedPlan: SegmentEditPlanResponse = {
      ...samplePlan,
      plan_version: 3,
      draft_segments: [
        ...samplePlan.draft_segments,
        {
          segment_id: "seg-new",
          start_ms: 2500,
          end_ms: 3500,
          spoken_text: "Mới",
          source_text: null,
          origin: "user",
          source_segment_index: null,
        },
      ],
      diff: { ...emptyDiff, has_changes: true },
    };
    const reconciled = reconcileRowsAfterSave(local, savedPlan.draft_segments);
    expect(reconciled[2].segment_id).toBe("seg-new");
    expect(reconciled[2].client_id).toBe("seg-new");
  });
});
