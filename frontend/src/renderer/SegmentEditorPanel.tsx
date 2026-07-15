import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp, ListChecks, Plus, RefreshCw, Save, Trash2, Upload, X } from "lucide-react";

import { ApiError } from "../shared/api";
import type {
  EditableSegmentDto,
  JobsApi,
  SegmentEditDiff,
  SegmentEditPlanResponse,
} from "../shared/contracts";
import {
  canExportDraft,
  canSaveDraft,
  draftRowsFromServer,
  editorStatusLabel,
  insertRowAfter,
  isDraftDirty,
  isSegmentEditConflictCode,
  isSegmentExportInProgressCode,
  moveRow,
  msToSecondsInput,
  parseSecondsInputToMs,
  previewModeForRow,
  reconcileRowsAfterSave,
  removeRow,
  rowsToSavePayload,
  statusDisplayText,
  type LocalSegmentRow,
} from "./segmentEditModel";

const EMPTY_DIFF: SegmentEditDiff = {
  has_changes: false,
  structural_changed: false,
  deltas: [],
  requires_tts_segment_ids: [],
  requires_duration_check_segment_ids: [],
  reusable_tts_segment_ids: [],
  deleted_segment_ids: [],
};

type Props = {
  api: JobsApi;
  jobId: string;
  jobStatus: string;
  onDirtyChange?: (dirty: boolean) => void;
};

export function SegmentEditorPanel({ api, jobId, jobStatus, onDirtyChange }: Props) {
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [planVersion, setPlanVersion] = useState(0);
  const [appliedPlanVersion, setAppliedPlanVersion] = useState(0);
  const [baseline, setBaseline] = useState<EditableSegmentDto[]>([]);
  const [rows, setRows] = useState<LocalSegmentRow[]>([]);
  const [diff, setDiff] = useState<SegmentEditDiff>(EMPTY_DIFF);
  const [saving, setSaving] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [conflict, setConflict] = useState(false);
  const [selectedClientId, setSelectedClientId] = useState<string | null>(null);
  const [showChangesModal, setShowChangesModal] = useState(false);
  const exportAwaitingRef = useRef(false);
  const exportCapturedVersionRef = useRef<number | null>(null);
  const prevJobStatusRef = useRef(jobStatus);

  const applyPlan = useCallback((plan: SegmentEditPlanResponse) => {
    setPlanVersion(plan.plan_version);
    setAppliedPlanVersion(plan.applied_plan_version);
    setBaseline(plan.draft_segments);
    setRows(draftRowsFromServer(plan.draft_segments));
    setDiff(plan.diff ?? EMPTY_DIFF);
    setConflict(false);
    setLoadError(null);
  }, []);

  const loadPlan = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const plan = await api.getSegmentEditPlan(jobId);
      applyPlan(plan);
    } catch (cause) {
      setLoadError(cause instanceof Error ? cause.message : "Không tải được plan phân đoạn.");
    } finally {
      setLoading(false);
    }
  }, [api, applyPlan, jobId]);

  useEffect(() => {
    void loadPlan();
    return () => {
      onDirtyChange?.(false);
    };
  }, [jobId, loadPlan, onDirtyChange]);

  const dirty = useMemo(() => isDraftDirty(rows, baseline), [rows, baseline]);

  useEffect(() => {
    onDirtyChange?.(dirty);
  }, [dirty, onDirtyChange]);

  useEffect(() => {
    const prev = prevJobStatusRef.current;
    prevJobStatusRef.current = jobStatus;
    if (!exportAwaitingRef.current) return;
    if (prev !== "completed" && jobStatus === "completed") {
      void (async () => {
        try {
          const plan = await api.getSegmentEditPlan(jobId);
          applyPlan(plan);
          const captured = exportCapturedVersionRef.current;
          exportAwaitingRef.current = false;
          exportCapturedVersionRef.current = null;
          setExporting(false);
          if (
            plan.applied_plan_version === plan.plan_version &&
            !plan.diff.has_changes
          ) {
            setMessage("Xuất phân đoạn thành công.");
          } else if (captured != null && plan.plan_version > captured) {
            setMessage("Đã xuất phiên bản trước; còn thay đổi chưa xuất");
          } else if (plan.diff.has_changes) {
            setMessage("Đã xuất phiên bản trước; còn thay đổi chưa xuất");
          } else {
            setMessage("Xuất phân đoạn thành công.");
          }
        } catch (cause) {
          setExporting(false);
          exportAwaitingRef.current = false;
          setMessage(cause instanceof Error ? cause.message : "Không tải lại plan sau khi xuất.");
        }
      })();
    }
  }, [api, applyPlan, jobId, jobStatus]);

  const status = editorStatusLabel(rows, baseline, diff);
  const saveEnabled = canSaveDraft({ rows, baseline, saving, exporting });
  const exportEnabled = canExportDraft({ rows, baseline, diff, saving, exporting });

  const rerunSegmentIds = useMemo(() => {
    const ids = new Set<string>();
    for (const id of diff.requires_tts_segment_ids) ids.add(id);
    for (const id of diff.requires_duration_check_segment_ids) ids.add(id);
    return ids;
  }, [diff]);
  const rerunCount = rerunSegmentIds.size;

  const changeDetails = useMemo(() => {
    const orderById = new Map<string, number>();
    rows.forEach((row, index) => {
      if (row.segment_id) orderById.set(row.segment_id, index + 1);
    });
    const ttsIds = new Set(diff.requires_tts_segment_ids);
    const durationIds = new Set(diff.requires_duration_check_segment_ids);
    return diff.deltas.map((delta) => {
      const kinds: string[] = [];
      if (delta.added) kinds.push("Thêm mới");
      if (delta.deleted) kinds.push("Xóa");
      if (delta.text_changed) kinds.push("Đổi bản dịch");
      if (delta.timing_changed) kinds.push("Đổi thời gian");
      if (delta.order_changed) kinds.push("Đổi thứ tự");
      const rerun: string[] = [];
      if (ttsIds.has(delta.segment_id)) rerun.push("TTS");
      if (durationIds.has(delta.segment_id)) rerun.push("Kiểm tra thời lượng");
      return {
        segment_id: delta.segment_id,
        order: delta.deleted ? null : orderById.get(delta.segment_id) ?? null,
        kinds,
        rerun,
      };
    });
  }, [diff, rows]);

  async function handleSave() {
    if (!saveEnabled) return;
    setSaving(true);
    setMessage(null);
    try {
      const plan = await api.saveSegmentEditPlan(jobId, {
        expected_plan_version: planVersion,
        segments: rowsToSavePayload(rows),
      });
      const reconciled = reconcileRowsAfterSave(rows, plan.draft_segments);
      setPlanVersion(plan.plan_version);
      setAppliedPlanVersion(plan.applied_plan_version);
      setBaseline(plan.draft_segments);
      setRows(reconciled);
      setDiff(plan.diff ?? EMPTY_DIFF);
      setConflict(false);
      setMessage(plan.diff?.has_changes ? "Đã lưu draft. Chưa xuất lại." : "Đã lưu.");
    } catch (cause) {
      const code = cause instanceof ApiError ? cause.code : undefined;
      if (isSegmentEditConflictCode(code)) {
        setConflict(true);
        setMessage(cause instanceof Error ? cause.message : "Xung đột phiên bản.");
      } else {
        setMessage(cause instanceof Error ? cause.message : "Lưu thất bại.");
      }
    } finally {
      setSaving(false);
    }
  }

  async function handleExport() {
    if (dirty) {
      setMessage("Lưu thay đổi trước khi xuất");
      return;
    }
    if (!exportEnabled) return;
    setExporting(true);
    setMessage(null);
    try {
      const result = await api.exportSegmentDraft(jobId, {
        expected_plan_version: planVersion,
      });
      if (result.status === "unchanged") {
        setExporting(false);
        setMessage("Không có thay đổi cần xuất.");
        const plan = await api.getSegmentEditPlan(jobId);
        applyPlan(plan);
        return;
      }
      exportAwaitingRef.current = true;
      exportCapturedVersionRef.current = planVersion;
      setMessage("Đang xuất phân đoạn…");
    } catch (cause) {
      setExporting(false);
      const code = cause instanceof ApiError ? cause.code : undefined;
      if (isSegmentEditConflictCode(code)) {
        setConflict(true);
        setMessage(cause instanceof Error ? cause.message : "Xung đột khi xuất.");
      } else if (isSegmentExportInProgressCode(code)) {
        setMessage(cause instanceof Error ? cause.message : "Đang có tiến trình xuất khác.");
      } else {
        setMessage(cause instanceof Error ? cause.message : "Xuất thất bại.");
      }
    }
  }

  async function handleReloadLatest() {
    await loadPlan();
    setMessage("Đã tải phiên bản mới nhất.");
  }

  function updateRow(clientId: string, patch: Partial<LocalSegmentRow>) {
    setRows((prev) => prev.map((row) => (row.client_id === clientId ? { ...row, ...patch } : row)));
  }

  function handleTimingChange(clientId: string, field: "start_ms" | "end_ms", raw: string) {
    const ms = parseSecondsInputToMs(raw);
    if (ms == null) return;
    updateRow(clientId, { [field]: ms });
  }

  if (loading) {
    return <p className="seg-editor__loading">Đang tải phân đoạn…</p>;
  }

  if (loadError) {
    return (
      <div style={{ display: "grid", gap: 10 }}>
        <p className="seg-editor__error">{loadError}</p>
        <button type="button" className="smoke-button" onClick={() => void loadPlan()}>
          Thử lại
        </button>
      </div>
    );
  }

  return (
    <div className="seg-editor">
      <div className="seg-editor__toolbar">
        <span className={`seg-editor__status${status === "dirty" ? " seg-editor__status--dirty" : ""}`}>
          {statusDisplayText(status)} · v{planVersion}
          {appliedPlanVersion !== planVersion ? ` (đã áp dụng v${appliedPlanVersion})` : ""}
        </span>
        <div className="seg-editor__actions">
          <button
            type="button"
            className="seg-editor__btn seg-editor__btn--primary"
            disabled={!saveEnabled}
            onClick={() => void handleSave()}
          >
            <Save size={14} />
            {saving ? "Đang lưu…" : "Lưu"}
          </button>
          <button
            type="button"
            className="seg-editor__btn seg-editor__btn--success"
            disabled={!exportEnabled}
            onClick={() => void handleExport()}
            title={dirty ? "Lưu thay đổi trước khi xuất" : undefined}
          >
            <Upload size={14} />
            {exporting ? "Đang xuất…" : "Xuất"}
          </button>
          <button
            type="button"
            className={`seg-editor__btn seg-editor__btn--warning${rerunCount > 0 ? " is-active" : ""}`}
            disabled={rerunCount === 0}
            onClick={() => setShowChangesModal(true)}
            title="Xem chi tiết các thay đổi sẽ chạy lại"
          >
            <ListChecks size={14} />
            {rerunCount} chạy lại
          </button>
        </div>
      </div>

      {dirty && !exportEnabled && (
        <p className="seg-editor__hint">Lưu thay đổi trước khi xuất</p>
      )}

      {conflict && (
        <div className="seg-editor__conflict">
          <span>
            Phiên bản trên máy chủ đã đổi. Giữ bản nháp cục bộ hoặc tải lại.
          </span>
          <button
            type="button"
            className="smoke-button"
            onClick={() => void handleReloadLatest()}
            style={{ minWidth: 0, display: "inline-flex", gap: 6, alignItems: "center" }}
          >
            <RefreshCw size={14} />
            Tải phiên bản mới nhất
          </button>
        </div>
      )}

      {message && <p className="seg-editor__message">{message}</p>}

      {rows.map((row, order) => {
        const preview = previewModeForRow(row, baseline);
        const selected = selectedClientId === row.client_id;
        return (
          <div
            key={row.client_id}
            className={`seg-editor__card${selected ? " seg-editor__card--selected" : ""}`}
            onClick={() => setSelectedClientId(row.client_id)}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
              <span className="seg-editor__card-index">#{order + 1}</span>
              <div className="seg-editor__card-actions">
                <button
                  type="button"
                  className="smoke-button"
                  disabled={order === 0 || saving || exporting}
                  onClick={(e) => {
                    e.stopPropagation();
                    setRows(moveRow(rows, row.client_id, "up"));
                  }}
                  aria-label={`Move segment ${order + 1} up`}
                >
                  <ArrowUp size={14} />
                </button>
                <button
                  type="button"
                  className="smoke-button"
                  disabled={order === rows.length - 1 || saving || exporting}
                  onClick={(e) => {
                    e.stopPropagation();
                    setRows(moveRow(rows, row.client_id, "down"));
                  }}
                  aria-label={`Move segment ${order + 1} down`}
                >
                  <ArrowDown size={14} />
                </button>
                <button
                  type="button"
                  className="smoke-button"
                  disabled={saving || exporting}
                  onClick={(e) => {
                    e.stopPropagation();
                    setRows((prev) => insertRowAfter(prev, row.client_id));
                  }}
                  aria-label="Thêm đoạn sau"
                >
                  <Plus size={14} />
                </button>
                <button
                  type="button"
                  className="smoke-button smoke-button--danger"
                  disabled={saving || exporting || rows.length <= 1}
                  onClick={(e) => {
                    e.stopPropagation();
                    setRows(removeRow(rows, row.client_id));
                  }}
                  aria-label={`Delete segment ${order + 1}`}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>

            <div className="seg-editor__field-row">
              <label className="seg-editor__field">
                Start (s)
                <input
                  className="seg-editor__input"
                  value={msToSecondsInput(row.start_ms)}
                  disabled={saving || exporting}
                  onChange={(e) => handleTimingChange(row.client_id, "start_ms", e.target.value)}
                />
              </label>
              <label className="seg-editor__field">
                End (s)
                <input
                  className="seg-editor__input"
                  value={msToSecondsInput(row.end_ms)}
                  disabled={saving || exporting}
                  onChange={(e) => handleTimingChange(row.client_id, "end_ms", e.target.value)}
                />
              </label>
            </div>

            {row.source_text && (
              <div className="seg-editor__source">Gốc: {row.source_text}</div>
            )}

            <label className="seg-editor__field">
              Bản dịch (nói)
              <textarea
                className="seg-editor__textarea"
                value={row.spoken_text}
                disabled={saving || exporting}
                onChange={(e) => updateRow(row.client_id, { spoken_text: e.target.value })}
                rows={2}
              />
            </label>

            {preview.mode === "available" && (
              <audio
                controls
                src={`http://127.0.0.1:8765/api/jobs/${jobId}/segments/${preview.source_segment_index}/wav`}
                style={{ height: 26, width: "100%" }}
              />
            )}
            {preview.mode === "timing_only" && (
              <>
                <audio
                  controls
                  src={`http://127.0.0.1:8765/api/jobs/${jobId}/segments/${preview.source_segment_index}/wav`}
                  style={{ height: 26, width: "100%" }}
                />
                <span className="seg-editor__note">{preview.note}</span>
              </>
            )}
            {preview.mode === "needs_tts" && (
              <span className="seg-editor__note">{preview.note}</span>
            )}
          </div>
        );
      })}

      {showChangesModal && (
        <div className="seg-editor__overlay" onClick={() => setShowChangesModal(false)}>
          <div className="seg-editor__modal" onClick={(e) => e.stopPropagation()}>
            <div className="seg-editor__modal-head">
              <strong style={{ fontSize: 15 }}>
                Chi tiết thay đổi · {rerunCount} đoạn chạy lại
              </strong>
              <button
                type="button"
                className="seg-editor__modal-close"
                onClick={() => setShowChangesModal(false)}
                aria-label="Đóng"
              >
                <X size={16} />
              </button>
            </div>

            {changeDetails.length === 0 ? (
              <p className="seg-editor__muted">Chưa có thay đổi nào.</p>
            ) : (
              <div style={{ display: "grid", gap: 10 }}>
                {changeDetails.map((detail) => (
                  <div key={detail.segment_id} className="seg-editor__change-item">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
                      <span className="seg-editor__change-title">
                        {detail.order != null ? `Đoạn #${detail.order}` : "Đoạn đã xóa"}
                      </span>
                      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                        {detail.kinds.map((kind) => (
                          <span key={kind} className="seg-editor__tag">
                            {kind}
                          </span>
                        ))}
                      </div>
                    </div>
                    {detail.rerun.length > 0 ? (
                      <span className="seg-editor__rerun">
                        Sẽ chạy lại: {detail.rerun.join(" · ")}
                      </span>
                    ) : (
                      <span className="seg-editor__muted">Không cần chạy lại audio</span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
