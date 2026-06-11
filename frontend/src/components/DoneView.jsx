import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";

const PII_LABEL = {
  document: "문서", screen: "화면", nameplate: "명패", id_card: "신분증", license_plate: "번호판", brand_logo: "상품 로고/상표",
};
const SCENE_LABEL = {
  meeting: "회의", lecture: "강의", interview: "인터뷰", public: "공공장소", vehicle: "자동차/도로", other: "기타",
};

const TABS = ["비교", "보고서"];

export default function DoneView({ jobId, onNewJob }) {
  const [report, setReport]         = useState(null);
  const [tab, setTab]               = useState("비교");
  const [savingProfile, setSaving]  = useState(false);
  const [profileName, setProfileName] = useState("");
  const [showSaveForm, setShowSave] = useState(false);
  const [saved, setSaved]           = useState(false);
  const inputRef = useRef();

  useEffect(() => {
    api.getReport(jobId).then(setReport).catch(() => {});
  }, [jobId]);

  async function handleSaveProfile() {
    if (!profileName.trim() || !report) return;
    setSaving(true);
    try {
      await api.saveProfile(
        jobId,
        profileName.trim(),
        report.protected_face_cluster_ids ?? [],
        report.masked_pii_types ?? [],
      );
      setSaved(true);
      setShowSave(false);
      setProfileName("");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex flex-col h-full animate-fade-in">

      {/* Top bar */}
      <div
        className="flex items-center justify-between px-8 py-4 flex-shrink-0"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: "#10b981", boxShadow: "0 0 6px #10b981" }}
            />
            <span className="font-bold" style={{ color: "var(--text)" }}>처리 완료</span>
          </div>

          {/* Tabs */}
          <div
            className="flex rounded-xl p-0.5 gap-0.5"
            style={{ background: "var(--card)", border: "1px solid var(--border)" }}
          >
            {TABS.map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className="px-4 py-1.5 rounded-lg text-xs font-semibold transition-all"
                style={tab === t ? {
                  background: "linear-gradient(135deg, #7c3aed, #3b82f6)",
                  color: "#fff",
                } : {
                  color: "var(--muted)",
                }}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Profile save */}
          {!report?.skipped && (
            showSaveForm ? (
              <div className="flex items-center gap-2 animate-fade-in">
                <input
                  ref={inputRef}
                  value={profileName}
                  onChange={(e) => setProfileName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSaveProfile()}
                  placeholder="프로필 이름"
                  className="rounded-lg px-3 py-2 text-xs outline-none"
                  style={{
                    background: "var(--card)",
                    border: "1px solid rgba(124,58,237,0.4)",
                    color: "var(--text)",
                    width: 140,
                  }}
                  autoFocus
                />
                <button
                  onClick={handleSaveProfile}
                  disabled={!profileName.trim() || savingProfile}
                  className="rounded-lg px-3 py-2 text-xs font-semibold transition-all"
                  style={{
                    background: profileName.trim() ? "rgba(124,58,237,0.8)" : "rgba(124,58,237,0.2)",
                    color: "#fff",
                  }}
                >
                  {savingProfile ? "저장 중..." : "저장"}
                </button>
                <button
                  onClick={() => setShowSave(false)}
                  className="text-xs px-2"
                  style={{ color: "var(--muted)" }}
                >
                  취소
                </button>
              </div>
            ) : (
              <button
                onClick={() => { setShowSave(true); setTimeout(() => inputRef.current?.focus(), 50); }}
                className="rounded-xl px-4 py-2 text-xs font-semibold transition-all"
                style={{
                  background: saved ? "rgba(16,185,129,0.1)" : "rgba(124,58,237,0.1)",
                  border: `1px solid ${saved ? "rgba(16,185,129,0.3)" : "rgba(124,58,237,0.25)"}`,
                  color: saved ? "#10b981" : "#a78bfa",
                }}
              >
                {saved ? "✓ 프로필 저장됨" : "프로필로 저장"}
              </button>
            )
          )}

          <a
            data-testid="download-output-link"
            href={api.downloadUrl(jobId)}
            download="output.mp4"
            className="rounded-xl px-4 py-2 text-xs font-semibold transition-all"
            style={{ background: "rgba(255,255,255,0.05)", color: "var(--text)", border: "1px solid var(--border)" }}
            onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--border-hover)"; }}
            onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border)"; }}
          >
            결과 영상 다운로드
          </a>
          <button
            onClick={() => {
              const input = document.createElement("input");
              input.type = "file"; input.accept = "video/*";
              input.onchange = (e) => { const f = e.target.files?.[0]; if (f) onNewJob(f); };
              input.click();
            }}
            className="grad-btn rounded-xl px-4 py-2 text-xs font-semibold text-white"
          >
            + 새 영상
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-auto">
        {tab === "비교" && <CompareTab jobId={jobId} report={report} />}
        {tab === "보고서" && <ReportTab report={report} />}
      </div>
    </div>
  );
}

/* ── 비교 탭: 원본 vs 결과 나란히 ─────────────────────────────── */
function CompareTab({ jobId, report }) {
  const skipped = report?.skipped;
  const showColoredPreview = Boolean(report?.colored_mask_enabled);
  const resultVideoSrc = showColoredPreview ? api.maskPreviewUrl(jobId) : api.downloadUrl(jobId);
  return (
    <div className="flex flex-col gap-4 p-8 h-full">
      {skipped && (
        <div
          className="rounded-xl px-4 py-3 text-sm flex items-center gap-2 flex-shrink-0"
          style={{ background: "rgba(88,88,160,0.12)", border: "1px solid rgba(88,88,160,0.25)", color: "#8888cc" }}
        >
          <span>ℹ</span>
          편집 스킵 — 원본 영상이 그대로 출력되었습니다. 별도의 마스킹이 적용되지 않았습니다.
        </div>
      )}
      <div className="flex gap-5 flex-1 min-h-0">
        <VideoPane label="원본" src={api.originalUrl(jobId)} badge="ORIGINAL" badgeColor="#5858a0" />
        {!skipped && (
          <VideoPane
            label={showColoredPreview ? "Colored mask preview" : "Masked result"}
            src={resultVideoSrc}
            badge={showColoredPreview ? "MASK PREVIEW" : "MASKED"}
            badgeColor="#10b981"
            glow
            testId={showColoredPreview ? "colored-mask-preview-video" : "masked-output-video"}
          />
        )}
      </div>
      {!skipped && <MaskLegend report={report} />}
    </div>
  );
}

function MaskLegend({ report }) {
  if (!report?.colored_mask_enabled || !report?.mask_colors) {
    return null;
  }
  return (
    <div
      className="rounded-xl px-4 py-3 text-xs flex flex-wrap items-center gap-3 flex-shrink-0"
      style={{ background: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.22)", color: "var(--muted)" }}
    >
      <span className="font-semibold" style={{ color: "#10b981" }}>컬러 마스크 미리보기</span>
      {Object.entries(report.mask_colors).map(([type, color]) => (
        <span key={type} className="flex items-center gap-1.5">
          <span
            className="w-2.5 h-2.5 rounded-full"
            style={{ background: color, boxShadow: `0 0 8px ${color}` }}
          />
          {type === "face" ? "얼굴" : (PII_LABEL[type] ?? type)}
        </span>
      ))}
      <span style={{ color: "rgba(221,221,245,0.55)" }}>
        다운로드 영상은 기존 블러/모자이크를 유지하고, 이 미리보기에서만 마스크 영역을 색으로 표시합니다.
      </span>
    </div>
  );
}

function VideoPane({ label, src, badge, badgeColor, glow, testId }) {
  return (
    <div
      className="flex-1 flex flex-col rounded-2xl overflow-hidden"
      style={{
        background: "var(--card)",
        border: glow ? "1px solid rgba(16,185,129,0.25)" : "1px solid var(--border)",
        boxShadow: glow ? "0 0 30px rgba(16,185,129,0.08)" : "none",
      }}
    >
      <div
        className="flex items-center justify-between px-4 py-3 flex-shrink-0"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <span className="text-sm font-semibold" style={{ color: "var(--text)" }}>{label}</span>
        <span
          className="text-[10px] font-bold px-2 py-0.5 rounded-full"
          style={{ background: `${badgeColor}22`, color: badgeColor, border: `1px solid ${badgeColor}44` }}
        >
          {badge}
        </span>
      </div>
      <div className="flex-1 flex items-center justify-center p-3" style={{ background: "#000" }}>
        <video
          data-testid={testId}
          src={src}
          controls
          className="max-h-full max-w-full rounded-lg"
          style={{ maxHeight: "calc(100vh - 260px)" }}
        />
      </div>
    </div>
  );
}

/* ── 보고서 탭 ─────────────────────────────────────────────────── */
function ReportTab({ report }) {
  if (!report) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-sm" style={{ color: "var(--muted)" }}>보고서를 불러오는 중...</p>
      </div>
    );
  }

  const isSkipped      = report.skipped === true;
  const protectedCount = report.protected_face_cluster_ids?.length ?? 0;
  const sceneAnalysis  = report.scene_analysis ?? {};
  const risk           = report.risk_assessment ?? {};
  const detection      = report.detection_summary ?? {};
  const checklist      = report.review_checklist ?? [];
  const guidelines     = report.guideline_items ?? [];
  const maskedPiiTypes = (report.masked_pii_types ?? []).map((t) => PII_LABEL[t] ?? t);
  const detectionPiiTypes = report.detection_pii_types ?? report.expected_pii ?? [];
  const detectionPiiLabels = detectionPiiTypes.map((t) => PII_LABEL[t] ?? t);
  const deterministicPiiLabels = (report.deterministic_pii_types_added ?? []).map((t) => PII_LABEL[t] ?? t);
  const detectionPiiSub = deterministicPiiLabels.length > 0
    ? `자동 추가: ${deterministicPiiLabels.join(", ")}`
    : (report.detection_pii_types ? "AI 예측 + 자동 탐지 범위" : "AI 사전 예측 항목");
  const totalPeople    = report.total_people_detected ?? 0;
  const blurredPeople  = totalPeople - protectedCount;
  const selectedPiiCategoryCount = report.selected_pii_category_count ?? maskedPiiTypes.length;
  const selectedPiiObjectCount = report.selected_pii_object_count ?? report.masked_pii_object_ids?.length ?? 0;
  const totalPiiCandidateCount = report.total_pii_candidates_detected ?? 0;
  const sam3ModeLabel = report.sam3_mode_label ?? (report.sam3_mode === "precision" ? "정밀" : "보통");
  const sam3ModeSub = report.sam3_mode_description ?? (
    report.sam3_mode === "precision"
      ? "모든 프레임 segmentation"
      : "3프레임 간격 segmentation + 보간"
  );

  const cards = isSkipped ? [
    {
      label: "씬 유형",
      value: SCENE_LABEL[report.scene_type] ?? report.scene_type ?? "—",
      sub: "GPT-4o 분석 결과",
      color: "#8b5cf6",
    },
    {
      label: "감지된 인물",
      value: `${totalPeople}명`,
      sub: "편집 없이 통과",
      color: "#5858a0",
    },
    {
      label: "편집 여부",
      value: "스킵",
      sub: "원본 영상 그대로 출력",
      color: "#5858a0",
    },
  ] : [
    {
      label: "씬 유형",
      value: SCENE_LABEL[report.scene_type] ?? report.scene_type ?? "—",
      sub: "GPT-4o 분석 결과",
      color: "#8b5cf6",
    },
    {
      label: "감지된 인물",
      value: `${totalPeople}명`,
      sub: `보호 ${protectedCount}명 · 블러 ${blurredPeople}명`,
      color: "#3b82f6",
    },
    {
      label: "마스킹된 PII 종류",
      value: maskedPiiTypes.length > 0 ? maskedPiiTypes.join(", ") : "없음",
      sub: `${maskedPiiTypes.length}종 처리됨`,
      color: "#f59e0b",
    },
    {
      label: "탐지 대상 PII 유형",
      value: detectionPiiLabels.length > 0 ? detectionPiiLabels.join(", ") : "없음",
      sub: detectionPiiSub,
      color: "#06b6d4",
    },
    {
      label: "선택된 PII 후보",
      value: selectedPiiObjectCount > 0 ? `${selectedPiiObjectCount}개` : `${selectedPiiCategoryCount}종`,
      sub: selectedPiiObjectCount > 0
        ? `후보 예시 ${totalPiiCandidateCount}개 중 개별 선택`
        : `후보 예시 ${totalPiiCandidateCount}개 기준`,
      color: "#f97316",
    },
    {
      label: "SAM3 마스킹 모드",
      value: sam3ModeLabel,
      sub: sam3ModeSub,
      color: "#60a5fa",
    },
    {
      label: "컬러 마스크 미리보기",
      value: report.colored_mask_enabled ? "적용됨" : "미적용",
      sub: "완료 화면에서 변경 영역 표시",
      color: "#10b981",
    },
  ];

  return (
    <div className="p-8 space-y-6 animate-slide-up">
      {/* Executive summary */}
      {report.executive_summary && (
        <section
          className="rounded-2xl p-6"
          style={{ background: "var(--card)", border: "1px solid var(--border)" }}
        >
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <p className="text-xs font-semibold mb-2" style={{ color: "var(--muted)" }}>
                PRIVACY PROCESSING REPORT
              </p>
              <p className="text-sm mt-1 leading-relaxed max-w-3xl" style={{ color: "var(--muted)" }}>
                {report.executive_summary}
              </p>
            </div>
            {risk.level && (
              <div
                className="rounded-xl px-4 py-3 text-right flex-shrink-0"
                style={{ background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.22)" }}
              >
                <p className="text-[11px] font-semibold" style={{ color: "var(--muted)" }}>위험도</p>
                <p className="text-xl font-bold" style={{ color: "#f59e0b" }}>{risk.level}</p>
                <p className="text-xs" style={{ color: "var(--muted)" }}>{risk.score ?? 0}/100</p>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-4">
        {cards.map((c) => (
          <div
            key={c.label}
            className="rounded-2xl p-5"
            style={{ background: "var(--card)", border: "1px solid var(--border)" }}
          >
            <p className="text-xs mb-2" style={{ color: "var(--muted)" }}>{c.label}</p>
            <p className="text-xl font-bold mb-0.5" style={{ color: c.color }}>{c.value}</p>
            <p className="text-xs" style={{ color: "var(--muted)" }}>{c.sub}</p>
          </div>
        ))}
      </div>

      {/* Scene analysis + risk reasons */}
      {(sceneAnalysis.frame_count || risk.reasons?.length > 0) && (
        <div className="grid grid-cols-2 gap-4">
          <ReportSection title="씬 분석 상세">
            <div className="space-y-3">
              <InfoRow label="분석 씬" value={sceneAnalysis.scene_label ?? SCENE_LABEL[report.scene_type] ?? report.scene_type ?? "—"} />
              <InfoRow label="분석 프레임" value={`${sceneAnalysis.frame_count ?? 0}개`} />
              {sceneAnalysis.confidence != null && (
                <InfoRow label="신뢰도" value={`${Math.round(sceneAnalysis.confidence * 100)}%`} />
              )}
              {(sceneAnalysis.expected_pii_labels ?? []).length > 0 && (
                <TagList label="예상 PII" items={sceneAnalysis.expected_pii_labels} />
              )}
              {(sceneAnalysis.top_privacy_risks ?? []).length > 0 && (
                <VoteList title="주요 리스크" items={sceneAnalysis.top_privacy_risks} />
              )}
              {(sceneAnalysis.recommended_focus ?? []).length > 0 && (
                <VoteList title="검토 포인트" items={sceneAnalysis.recommended_focus} />
              )}
            </div>
          </ReportSection>

          <ReportSection title="위험 평가 근거">
            <div className="space-y-2">
              {(risk.reasons ?? []).map((reason, index) => (
                <p
                  key={`${reason}-${index}`}
                  className="text-sm leading-relaxed rounded-xl px-3 py-2"
                  style={{ color: "var(--muted)", background: "rgba(255,255,255,0.035)" }}
                >
                  {reason}
                </p>
              ))}
            </div>
          </ReportSection>
        </div>
      )}

      {/* Detection + checklist */}
      {((detection.face_clusters?.length > 0 || detection.pii_candidates?.length > 0) || checklist.length > 0) && (
        <div className="grid grid-cols-2 gap-4">
          <ReportSection title="감지 항목별 처리">
            <div className="space-y-3">
              <CompactTable
                empty="감지된 인물이 없습니다"
                rows={(detection.face_clusters ?? []).map((face) => ({
                  key: face.cluster_id,
                  name: face.display_name,
                  meta: `${face.appearances}회 등장`,
                  action: face.planned_action,
                  active: face.protected,
                }))}
              />
              <CompactTable
                empty="감지된 비얼굴 PII가 없습니다"
                rows={(detection.pii_candidates ?? []).map((pii) => ({
                  key: pii.object_id,
                  name: pii.label,
                  meta: `신뢰도 ${Math.round((pii.confidence ?? 0) * 100)}%`,
                  action: pii.planned_action,
                  active: pii.selected_for_masking,
                }))}
              />
            </div>
          </ReportSection>

          <ReportSection title="최종 검토 체크리스트">
            <div className="space-y-2">
              {checklist.map((item) => (
                <div
                  key={item.title}
                  className="rounded-xl px-3 py-3"
                  style={{ background: "rgba(255,255,255,0.035)", border: "1px solid var(--border)" }}
                >
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm font-semibold" style={{ color: "var(--text)" }}>{item.title}</p>
                    <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ color: "#f59e0b", background: "rgba(245,158,11,0.12)" }}>
                      {item.status}
                    </span>
                  </div>
                  <p className="text-xs mt-1 leading-relaxed" style={{ color: "var(--muted)" }}>{item.detail}</p>
                </div>
              ))}
            </div>
          </ReportSection>
        </div>
      )}

      {/* Guidelines */}
      {guidelines.length > 0 && (
        <ReportSection title="AI 편집 가이드라인">
          <div className="grid grid-cols-2 gap-2">
            {guidelines.map((item, index) => (
              <p
                key={`${item.message}-${index}`}
                className="text-sm rounded-xl px-3 py-2"
                style={{
                  color: item.level === "warning" ? "#fbbf24" : "var(--muted)",
                  background: item.level === "warning" ? "rgba(245,158,11,0.08)" : "rgba(255,255,255,0.035)",
                  border: "1px solid var(--border)",
                }}
              >
                {item.message}
              </p>
            ))}
          </div>
        </ReportSection>
      )}

      {/* Raw JSON */}
      <div
        className="rounded-2xl overflow-hidden"
        style={{ border: "1px solid var(--border)" }}
      >
        <div
          className="px-5 py-3 flex items-center justify-between"
          style={{ borderBottom: "1px solid var(--border)", background: "var(--card)" }}
        >
          <span className="text-xs font-semibold" style={{ color: "var(--muted)" }}>RAW REPORT</span>
          <button
            className="text-xs px-3 py-1 rounded-lg transition-colors"
            style={{ background: "rgba(255,255,255,0.05)", color: "var(--muted)" }}
            onClick={() => {
              const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
              const a = document.createElement("a");
              a.href = URL.createObjectURL(blob);
              a.download = "report.json";
              a.click();
            }}
          >
            JSON 다운로드
          </button>
        </div>
        <pre
          className="p-5 text-xs overflow-auto"
          style={{ background: "rgba(0,0,0,0.3)", color: "rgba(221,221,245,0.6)", maxHeight: 280 }}
        >
          {JSON.stringify(report, null, 2)}
        </pre>
      </div>
    </div>
  );
}

function ReportSection({ title, children }) {
  return (
    <section
      className="rounded-2xl p-5"
      style={{ background: "var(--card)", border: "1px solid var(--border)" }}
    >
      <h3 className="text-sm font-bold mb-4" style={{ color: "var(--text)" }}>{title}</h3>
      {children}
    </section>
  );
}

function InfoRow({ label, value }) {
  return (
    <div className="flex items-center justify-between gap-3 text-sm">
      <span style={{ color: "var(--muted)" }}>{label}</span>
      <span className="font-semibold text-right" style={{ color: "var(--text)" }}>{value}</span>
    </div>
  );
}

function TagList({ label, items }) {
  return (
    <div>
      <p className="text-xs mb-2" style={{ color: "var(--muted)" }}>{label}</p>
      <div className="flex flex-wrap gap-1.5">
        {(items ?? []).length === 0 ? (
          <span className="text-xs" style={{ color: "var(--muted)" }}>없음</span>
        ) : (
          items.map((item) => (
            <span
              key={item}
              className="text-xs px-2 py-1 rounded-full"
              style={{ background: "rgba(59,130,246,0.12)", color: "#93c5fd", border: "1px solid rgba(59,130,246,0.24)" }}
            >
              {item}
            </span>
          ))
        )}
      </div>
    </div>
  );
}

function VoteList({ title, items }) {
  if (!items?.length) return null;
  return (
    <div>
      <p className="text-xs mb-2" style={{ color: "var(--muted)" }}>{title}</p>
      <div className="space-y-1.5">
        {items.map((item, index) => (
          <div key={`${item.message}-${index}`} className="flex items-start justify-between gap-3 text-xs">
            <span className="leading-relaxed" style={{ color: "var(--text)" }}>{item.message}</span>
            <span className="flex-shrink-0" style={{ color: "var(--muted)" }}>{item.votes}표</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CompactTable({ rows, empty }) {
  if (!rows?.length) {
    return <p className="text-sm" style={{ color: "var(--muted)" }}>{empty}</p>;
  }
  return (
    <div className="space-y-2">
      {rows.map((row) => (
        <div
          key={row.key}
          className="grid grid-cols-[1fr_auto] gap-3 rounded-xl px-3 py-2"
          style={{ background: "rgba(255,255,255,0.035)", border: "1px solid var(--border)" }}
        >
          <div className="min-w-0">
            <p className="text-sm font-semibold truncate" style={{ color: "var(--text)" }}>{row.name}</p>
            <p className="text-xs" style={{ color: "var(--muted)" }}>{row.meta}</p>
          </div>
          <span
            className="text-[11px] self-center px-2 py-1 rounded-full"
            style={{
              color: row.active ? "#10b981" : "#f59e0b",
              background: row.active ? "rgba(16,185,129,0.1)" : "rgba(245,158,11,0.1)",
            }}
          >
            {row.action}
          </span>
        </div>
      ))}
    </div>
  );
}
