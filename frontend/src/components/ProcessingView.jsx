import { useEffect, useRef } from "react";
import { useJobLog } from "../hooks/useJobLog";

const STEPS = [
  { key: "uploading",            label: "업로드" },
  { key: "detecting",            label: "AI 분석" },
  { key: "generating_guideline", label: "가이드라인" },
  { key: "selecting",            label: "검토" },
  { key: "masking",              label: "마스킹" },
  { key: "done",                 label: "완료" },
];

const STEP_ORDER = STEPS.map((s) => s.key);

const STEP_DESC = {
  uploading:            "영상을 서버로 전송하고 있습니다",
  detecting:            "GPT-4o가 씬을 분석하고 SAM3 + InsightFace로 인물을 탐지합니다",
  generating_guideline: "탐지 결과를 바탕으로 편집 가이드라인을 생성하고 있습니다",
  masking:              "SAM3 픽셀 마스크로 프레임별 마스킹을 진행합니다",
};

export default function ProcessingView({ jobId, step }) {
  const logs = useJobLog(jobId, true);
  const bottomRef = useRef();
  const currentIdx = STEP_ORDER.indexOf(step);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className="flex flex-col h-full p-8 animate-fade-in">

      {/* Step indicator */}
      <div className="mb-10">
        <h2 className="text-xl font-bold mb-1" style={{ color: "var(--text)" }}>
          처리 중
        </h2>
        <p className="text-sm mb-8" style={{ color: "var(--muted)" }}>
          {STEP_DESC[step] ?? "처리를 진행하고 있습니다"}
        </p>

        <div className="flex items-center gap-0">
          {STEPS.map((s, i) => {
            const isDone    = i < currentIdx;
            const isActive  = i === currentIdx;
            const isPending = i > currentIdx;
            return (
              <div key={s.key} className="flex items-center" style={{ flex: i < STEPS.length - 1 ? 1 : 0 }}>
                <div className="flex flex-col items-center gap-1.5">
                  {/* Circle */}
                  <div
                    className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all"
                    style={
                      isDone ? {
                        background: "linear-gradient(135deg, #7c3aed, #06b6d4)",
                        color: "#fff",
                      } : isActive ? {
                        background: "transparent",
                        border: "2px solid #7c3aed",
                        color: "#a78bfa",
                        boxShadow: "0 0 12px rgba(124,58,237,0.4)",
                      } : {
                        background: "rgba(255,255,255,0.04)",
                        border: "1px solid var(--border)",
                        color: "var(--muted)",
                      }
                    }
                  >
                    {isDone ? "✓" : i + 1}
                    {isActive && (
                      <span
                        className="absolute w-8 h-8 rounded-full animate-ping"
                        style={{ background: "rgba(124,58,237,0.2)" }}
                      />
                    )}
                  </div>

                  {/* Label */}
                  <span
                    className="text-[10px] font-medium whitespace-nowrap"
                    style={{ color: isDone || isActive ? "var(--text)" : "var(--muted)" }}
                  >
                    {s.label}
                  </span>
                </div>

                {/* Connector */}
                {i < STEPS.length - 1 && (
                  <div
                    className="h-px flex-1 mx-2 mb-5"
                    style={{
                      background: i < currentIdx
                        ? "linear-gradient(90deg, #7c3aed, #3b82f6)"
                        : "rgba(255,255,255,0.07)",
                    }}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Live log */}
      <div className="flex-1 min-h-0">
        <div className="flex items-center gap-2 mb-3">
          <span
            className="w-1.5 h-1.5 rounded-full animate-glow-pulse"
            style={{ background: "#3b82f6", boxShadow: "0 0 6px #3b82f6" }}
          />
          <span className="text-xs font-medium" style={{ color: "var(--muted)" }}>실시간 로그</span>
        </div>

        <div
          className="glass rounded-2xl p-5 h-64 overflow-y-auto font-mono text-xs space-y-1.5"
          style={{ background: "rgba(0,0,0,0.3)" }}
        >
          {logs.length === 0 && (
            <span style={{ color: "var(--muted)" }}>처리를 시작하고 있습니다...</span>
          )}
          {logs.map((log, i) => (
            <div
              key={i}
              className="flex gap-2 animate-fade-in"
              style={{ color: log.step === "error" ? "#ef4444" : log.event === "done" ? "#10b981" : "rgba(221,221,245,0.8)" }}
            >
              <span style={{ color: "var(--muted)", flexShrink: 0 }}>
                {log.step === "error" ? "✗" : log.event === "done" ? "✓" : "›"}
              </span>
              <span>{log.message}</span>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  );
}
