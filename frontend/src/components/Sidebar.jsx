import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";

const STATUS_META = {
  pending:              { label: "대기",     dot: "#5858a0",   pulse: false },
  detecting:            { label: "분석 중",  dot: "#3b82f6",   pulse: true  },
  generating_guideline: { label: "가이드라인", dot: "#8b5cf6", pulse: true  },
  awaiting_selection:   { label: "선택 대기", dot: "#f59e0b",  pulse: false },
  masking:              { label: "마스킹",   dot: "#06b6d4",   pulse: true  },
  done:                 { label: "완료",     dot: "#10b981",   pulse: false },
  failed:               { label: "오류",     dot: "#ef4444",   pulse: false },
};

const SCENE_LABEL = {
  meeting: "회의", lecture: "강의", interview: "인터뷰", public: "공공", vehicle: "자동차", other: "기타",
};

export default function Sidebar({ activeJobId, activeStep, onSelectJob, onNewJob, onDeleteJob }) {
  const [jobs, setJobs] = useState([]);
  const [deletingId, setDeletingId] = useState(null);
  const inputRef = useRef();

  useEffect(() => {
    let alive = true;
    async function poll() {
      try {
        const data = await api.listJobs();
        if (alive) setJobs(data);
      } catch {}
    }
    poll();
    const t = setInterval(poll, 2000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  async function handleDelete(job) {
    const ok = window.confirm(
      `${job.job_id.slice(0, 8)} 작업을 삭제하시겠습니까?\n` +
      `DB와 저장된 영상/프레임이 모두 영구 삭제됩니다.`
    );
    if (!ok) return;
    setDeletingId(job.job_id);
    try {
      await api.deleteJob(job.job_id);
      setJobs((prev) => prev.filter((j) => j.job_id !== job.job_id));
      if (job.job_id === activeJobId) onDeleteJob?.(job.job_id);
    } catch (err) {
      const detail = err?.response?.data?.detail ?? "삭제에 실패했습니다.";
      window.alert(detail);
    } finally {
      setDeletingId(null);
    }
  }

  const active  = jobs.filter((j) => !["done", "failed"].includes(j.status));
  const done    = jobs.filter((j) => j.status === "done");
  const failed  = jobs.filter((j) => j.status === "failed");

  return (
    <aside
      className="flex flex-col h-screen overflow-hidden flex-shrink-0"
      style={{ width: 260, background: "var(--surface)", borderRight: "1px solid var(--border)" }}
    >
      {/* Logo */}
      <div className="px-5 pt-6 pb-4" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="flex items-center gap-2.5 mb-0.5">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold"
            style={{ background: "linear-gradient(135deg, #7c3aed, #06b6d4)" }}
          >
            S
          </div>
          <span className="font-bold text-base" style={{ color: "var(--text)" }}>SafeVlog3</span>
          <span
            className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full"
            style={{ background: "rgba(124,58,237,0.2)", color: "#a78bfa", border: "1px solid rgba(124,58,237,0.3)" }}
          >
            BETA
          </span>
        </div>
        <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>Privacy Masking AI</p>
      </div>

      {/* Upload button */}
      <div className="px-4 py-3">
        <button
          className="grad-btn w-full rounded-xl py-2.5 text-sm font-semibold text-white flex items-center justify-center gap-2"
          onClick={() => inputRef.current?.click()}
        >
          <span className="text-base leading-none">+</span>
          새 영상 업로드
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) { e.target.value = ""; onNewJob(f); }
          }}
        />
      </div>

      {/* Job list */}
      <div className="flex-1 overflow-y-auto px-3 pb-4 space-y-4">

        {jobs.length === 0 && (
          <p className="text-xs text-center px-2 pt-6" style={{ color: "var(--muted)" }}>
            아직 처리한 영상이 없습니다
          </p>
        )}

        {active.length > 0 && (
          <JobGroup label="진행 중" jobs={active} activeJobId={activeJobId}
            onSelect={onSelectJob} onDelete={handleDelete} deletingId={deletingId} />
        )}
        {done.length > 0 && (
          <JobGroup label="완료" jobs={done} activeJobId={activeJobId}
            onSelect={onSelectJob} onDelete={handleDelete} deletingId={deletingId} />
        )}
        {failed.length > 0 && (
          <JobGroup label="오류" jobs={failed} activeJobId={activeJobId}
            onSelect={onSelectJob} onDelete={handleDelete} deletingId={deletingId} />
        )}
      </div>

      {/* Footer */}
      <div className="px-5 py-4" style={{ borderTop: "1px solid var(--border)" }}>
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          {jobs.length}개 작업 · {active.length}개 진행 중
        </p>
      </div>
    </aside>
  );
}

function JobGroup({ label, jobs, activeJobId, onSelect, onDelete, deletingId }) {
  return (
    <div>
      <p
        className="text-[10px] font-semibold uppercase tracking-widest px-2 mb-1.5"
        style={{ color: "var(--muted)" }}
      >
        {label}
      </p>
      <div className="space-y-1">
        {jobs.map((job) => (
          <JobItem
            key={job.job_id}
            job={job}
            isActive={job.job_id === activeJobId}
            onSelect={onSelect}
            onDelete={onDelete}
            isDeleting={deletingId === job.job_id}
          />
        ))}
      </div>
    </div>
  );
}

const IN_PROGRESS_STATUSES = ["detecting", "generating_guideline", "masking"];

function JobItem({ job, isActive, onSelect, onDelete, isDeleting }) {
  const meta = STATUS_META[job.status] ?? STATUS_META.pending;
  const clickable = ["awaiting_selection", "done", "detecting", "masking", "generating_guideline"].includes(job.status);
  const deletable = !IN_PROGRESS_STATUSES.includes(job.status) && !isDeleting;
  const [hovered, setHovered] = useState(false);

  return (
    <div
      className="group relative animate-slide-in-left"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        onClick={() => clickable && onSelect(job)}
        disabled={!clickable}
        className="w-full text-left rounded-xl px-3 py-2.5 transition-all"
        style={isActive ? {
          background: "linear-gradient(var(--card), var(--card)) padding-box, linear-gradient(135deg, #7c3aed, #3b82f6, #06b6d4) border-box",
          border: "1px solid transparent",
        } : {
          background: hovered ? "rgba(255,255,255,0.03)" : "transparent",
          border: "1px solid transparent",
        }}
      >
        <div className="flex items-center gap-2.5">
          {/* Status dot */}
          <span
            className={`w-2 h-2 rounded-full flex-shrink-0 ${meta.pulse ? "animate-glow-pulse" : ""}`}
            style={{ background: meta.dot, boxShadow: meta.pulse ? `0 0 6px ${meta.dot}` : "none" }}
          />

          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between gap-1">
              <span
                className="font-mono text-xs truncate"
                style={{ color: isActive ? "var(--text)" : "rgba(221,221,245,0.7)" }}
              >
                {job.job_id.slice(0, 8)}
              </span>
              {job.scene_type && (
                <span className="text-[10px] flex-shrink-0" style={{ color: "var(--muted)" }}>
                  {SCENE_LABEL[job.scene_type] ?? job.scene_type}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-[10px] font-medium" style={{ color: meta.dot }}>
                {meta.label}
              </span>
              {(job.face_count > 0 || job.pii_count > 0) && (
                <span className="text-[10px]" style={{ color: "var(--muted)" }}>
                  {[
                    job.face_count > 0 && `👤 ${job.face_count}`,
                    job.pii_count > 0  && `🔒 ${job.pii_count}`,
                  ].filter(Boolean).join("  ")}
                </span>
              )}
            </div>
          </div>

          {job.status === "awaiting_selection" && !hovered && (
            <span className="text-[10px] font-bold flex-shrink-0" style={{ color: "#f59e0b" }}>→</span>
          )}
        </div>
      </button>

      {hovered && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            if (deletable) onDelete?.(job);
          }}
          disabled={!deletable}
          title={
            deletable
              ? "작업 삭제 (DB · 영상 파일 모두)"
              : isDeleting
                ? "삭제 중..."
                : "진행 중인 작업은 삭제할 수 없습니다"
          }
          className="absolute top-1/2 right-2 -translate-y-1/2 rounded-md p-1 transition-colors"
          style={{
            background: "rgba(0,0,0,0.4)",
            color: deletable ? "#f87171" : "rgba(221,221,245,0.3)",
            cursor: deletable ? "pointer" : "not-allowed",
          }}
          onMouseEnter={(e) => {
            if (deletable) e.currentTarget.style.background = "rgba(239,68,68,0.2)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "rgba(0,0,0,0.4)";
          }}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="3 6 5 6 21 6" />
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
            <path d="M10 11v6M14 11v6" />
            <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
          </svg>
        </button>
      )}
    </div>
  );
}
