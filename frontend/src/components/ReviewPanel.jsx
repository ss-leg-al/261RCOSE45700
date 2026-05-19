import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";

const PII_LABEL = {
  document:  "문서",
  screen:    "화면/스크린",
  nameplate: "명패",
  id_card:   "신분증",
};

const SCENE_LABEL = {
  meeting: "회의", lecture: "강의", interview: "인터뷰", public: "공공장소", other: "기타",
};

export default function ReviewPanel({ candidates, guideline, onSubmit, onSkip, jobId }) {
  const { scene_type, face_clusters, pii_candidates } = candidates;

  if (face_clusters.length === 0 && pii_candidates.length === 0) {
    return <NothingDetectedView sceneType={scene_type} onDone={onSkip} />;
  }

  const [protectedFaces, setProtectedFaces] = useState(new Set());
  const piiTypes = [...new Set(pii_candidates.map((p) => p.pii_type))];
  const [maskedTypes, setMaskedTypes] = useState(new Set(piiTypes));

  const [profiles, setProfiles]               = useState([]);
  const [profileOpen, setProfileOpen]         = useState(false);
  const [applyingProfile, setApplyingProfile] = useState(false);
  const [profileResult, setProfileResult]     = useState(null); // null | ApplyProfileResponse
  const [profileFaces, setProfileFaces]       = useState(new Set()); // cluster IDs matched by profile
  const dropdownRef = useRef();

  useEffect(() => {
    api.listProfiles().then(setProfiles).catch(() => {});
  }, []);

  useEffect(() => {
    function handleClick(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setProfileOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  async function handleApplyProfile(profileId) {
    setApplyingProfile(true);
    setProfileOpen(false);
    try {
      const result = await api.applyProfile(jobId, profileId);
      setProtectedFaces(new Set(result.protected_face_cluster_ids));
      setMaskedTypes(new Set(result.masked_pii_types));
      setProfileFaces(new Set(result.protected_face_cluster_ids));
      setProfileResult(result);
    } finally {
      setApplyingProfile(false);
    }
  }

  function toggleFace(id) {
    setProtectedFaces((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function togglePiiType(type) {
    setMaskedTypes((prev) => {
      const next = new Set(prev);
      next.has(type) ? next.delete(type) : next.add(type);
      return next;
    });
  }

  const warnings = guideline.filter((g) => g.level === "warning");
  const infos    = guideline.filter((g) => g.level === "info");

  return (
    <div className="flex flex-col h-full p-8 gap-6 animate-fade-in">

      {/* Header */}
      <div className="flex items-start justify-between flex-shrink-0">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span
              className="text-[11px] font-semibold px-2 py-0.5 rounded-full"
              style={{ background: "rgba(245,158,11,0.15)", color: "#f59e0b", border: "1px solid rgba(245,158,11,0.3)" }}
            >
              검토 필요
            </span>
            {scene_type && (
              <span
                className="text-[11px] font-semibold px-2 py-0.5 rounded-full"
                style={{ background: "rgba(255,255,255,0.05)", color: "var(--muted)", border: "1px solid var(--border)" }}
              >
                {SCENE_LABEL[scene_type] ?? scene_type}
              </span>
            )}
          </div>
          <h2 className="text-xl font-bold" style={{ color: "var(--text)" }}>마스킹 검토 및 선택</h2>
          <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
            가이드라인을 확인하고 마스킹 대상을 선택하세요
          </p>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0" ref={dropdownRef}>
          {/* Profile apply dropdown */}
          <div className="relative">
            <button
              onClick={() => setProfileOpen((v) => !v)}
              disabled={applyingProfile}
              className="rounded-xl px-4 py-2.5 text-sm font-medium flex items-center gap-1.5 transition-all"
              style={{
                background: "rgba(124,58,237,0.1)",
                border: "1px solid rgba(124,58,237,0.25)",
                color: "#a78bfa",
              }}
            >
              {applyingProfile ? "적용 중..." : "프로필 적용"}
              <span className="text-xs">{profileOpen ? "▲" : "▼"}</span>
            </button>

            {profileOpen && (
              <div
                className="absolute right-0 top-full mt-1 z-20 rounded-xl overflow-hidden animate-slide-up"
                style={{
                  minWidth: 200,
                  background: "var(--card)",
                  border: "1px solid rgba(124,58,237,0.3)",
                  boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
                }}
              >
                {profiles.length === 0 ? (
                  <div className="px-4 py-3 text-xs" style={{ color: "var(--muted)" }}>
                    저장된 프로필 없음
                  </div>
                ) : (
                  profiles.map((p) => (
                    <button
                      key={p.profile_id}
                      onClick={() => handleApplyProfile(p.profile_id)}
                      className="w-full text-left px-4 py-3 text-sm transition-colors"
                      style={{ color: "var(--text)", borderBottom: "1px solid var(--border)" }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(124,58,237,0.1)"; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                    >
                      <p className="font-medium">{p.name}</p>
                      <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>
                        얼굴 {p.face_count}명 보호 ·{" "}
                        {p.masked_pii_types.length > 0 ? p.masked_pii_types.join(", ") : "PII 없음"}
                      </p>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>
          {/* Pass: 검토 없이 전체 기본값으로 진행 */}
          <button
            onClick={onSkip}
            className="rounded-xl px-4 py-2.5 text-sm font-medium transition-all"
            style={{
              background: "rgba(255,255,255,0.04)",
              border: "1px solid var(--border)",
              color: "var(--muted)",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = "var(--border-hover)";
              e.currentTarget.style.color = "var(--text)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = "var(--border)";
              e.currentTarget.style.color = "var(--muted)";
            }}
            title="편집 없이 원본 그대로 완료 처리"
          >
            편집 스킵
          </button>

          <button
            onClick={() => onSubmit([...protectedFaces], [...maskedTypes])}
            className="grad-btn rounded-xl px-6 py-2.5 text-sm font-bold text-white flex items-center gap-2"
          >
            마스킹 시작 →
          </button>
        </div>
      </div>

      {/* Profile apply result banner */}
      {profileResult && (
        <div
          className="flex-shrink-0 rounded-xl px-4 py-3 flex items-start gap-3 animate-slide-up"
          style={{
            background: profileResult.matched_face_count === 0
              ? "rgba(245,158,11,0.08)"
              : "rgba(16,185,129,0.08)",
            border: `1px solid ${profileResult.matched_face_count === 0
              ? "rgba(245,158,11,0.25)"
              : "rgba(16,185,129,0.25)"}`,
          }}
        >
          <span className="text-base flex-shrink-0 mt-0.5">
            {profileResult.matched_face_count === 0 ? "⚠" : "✓"}
          </span>
          <div className="flex-1 min-w-0">
            {profileResult.matched_face_count === 0 ? (
              <p className="text-sm font-semibold" style={{ color: "#f59e0b" }}>
                프로필 얼굴 미일치 — 이 영상에서 저장된 얼굴을 찾지 못했습니다
              </p>
            ) : (
              <p className="text-sm font-semibold" style={{ color: "#10b981" }}>
                프로필 적용 완료 —{" "}
                {profileResult.matched_face_count}/{profileResult.profile_face_count}명 얼굴 일치
              </p>
            )}
            {profileResult.unmatched_pii_types.length > 0 && (
              <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>
                프로필 PII 중 이 영상에 없는 항목:{" "}
                <span style={{ color: "#f59e0b" }}>
                  {profileResult.unmatched_pii_types.map((t) => PII_LABEL[t] ?? t).join(", ")}
                </span>
              </p>
            )}
          </div>
          <button
            onClick={() => setProfileResult(null)}
            className="flex-shrink-0 text-xs px-2 py-0.5 rounded"
            style={{ color: "var(--muted)" }}
          >
            ×
          </button>
        </div>
      )}

      {/* Main 2-column layout */}
      <div className="flex-1 min-h-0 flex gap-5">

        {/* LEFT — Guideline */}
        <div
          className="flex flex-col rounded-2xl overflow-hidden flex-shrink-0"
          style={{ width: 300, background: "var(--card)", border: "1px solid var(--border)" }}
        >
          <div
            className="px-5 py-4 flex-shrink-0"
            style={{ borderBottom: "1px solid var(--border)" }}
          >
            <div className="flex items-center gap-2">
              <span className="text-base">📋</span>
              <span className="font-semibold text-sm" style={{ color: "var(--text)" }}>편집 가이드라인</span>
            </div>
            <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>
              AI가 감지한 주의 사항입니다
            </p>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            {guideline.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full gap-2 py-8">
                <span className="text-2xl">✅</span>
                <p className="text-xs text-center" style={{ color: "var(--muted)" }}>
                  특이사항이 없습니다
                </p>
              </div>
            )}

            {warnings.map((item, i) => (
              <GuidelineItem key={`w${i}`} item={item} />
            ))}
            {infos.map((item, i) => (
              <GuidelineItem key={`i${i}`} item={item} />
            ))}
          </div>

          {guideline.length > 0 && (
            <div
              className="px-4 py-3 flex-shrink-0 flex gap-3 text-xs"
              style={{ borderTop: "1px solid var(--border)", color: "var(--muted)" }}
            >
              {warnings.length > 0 && (
                <span className="flex items-center gap-1">
                  <span style={{ color: "#f59e0b" }}>⚠</span> {warnings.length}개 경고
                </span>
              )}
              {infos.length > 0 && (
                <span className="flex items-center gap-1">
                  <span style={{ color: "#3b82f6" }}>ℹ</span> {infos.length}개 안내
                </span>
              )}
            </div>
          )}
        </div>

        {/* RIGHT — Selection */}
        <div className="flex-1 min-w-0 overflow-y-auto space-y-5">

          {/* Face clusters */}
          {face_clusters.length > 0 && (
            <section
              className="rounded-2xl p-5"
              style={{ background: "var(--card)", border: "1px solid var(--border)" }}
            >
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="font-semibold text-sm" style={{ color: "var(--text)" }}>
                    👤 얼굴 — {face_clusters.length}명 감지
                  </h3>
                  <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>
                    보호할 인물을 클릭하세요. 나머지는 자동 블러됩니다.
                  </p>
                </div>
                <span className="text-xs" style={{ color: "var(--muted)" }}>
                  {protectedFaces.size === 0
                    ? "전체 블러"
                    : `${protectedFaces.size}명 보호`}
                </span>
              </div>

              <div className="flex flex-wrap gap-3">
                {face_clusters.map((fc) => {
                  const isProtected  = protectedFaces.has(fc.cluster_id);
                  const fromProfile  = profileFaces.has(fc.cluster_id);
                  return (
                    <button
                      key={fc.cluster_id}
                      onClick={() => toggleFace(fc.cluster_id)}
                      className="relative rounded-xl overflow-hidden transition-all"
                      style={{
                        width: 76,
                        border: isProtected
                          ? "2px solid transparent"
                          : "2px solid var(--border)",
                        ...(isProtected && {
                          background: "linear-gradient(var(--card), var(--card)) padding-box, linear-gradient(135deg, #7c3aed, #06b6d4) border-box",
                          boxShadow: "0 0 16px rgba(124,58,237,0.3)",
                        }),
                      }}
                    >
                      <img
                        src={fc.thumbnail_url}
                        alt=""
                        style={{ width: 76, height: 76, objectFit: "cover", display: "block" }}
                      />
                      {isProtected && (
                        <div
                          className="absolute inset-0 flex items-center justify-center"
                          style={{ background: "rgba(124,58,237,0.3)" }}
                        >
                          <span className="text-white text-lg font-bold drop-shadow">✓</span>
                        </div>
                      )}
                      {fromProfile && (
                        <div
                          className="absolute top-1 right-1 text-[9px] font-bold px-1 py-0.5 rounded"
                          style={{
                            background: "rgba(16,185,129,0.85)",
                            color: "#fff",
                            lineHeight: 1,
                          }}
                        >
                          🔖
                        </div>
                      )}
                      <div
                        className="absolute bottom-0 inset-x-0 text-[10px] text-center py-0.5"
                        style={{ background: "rgba(0,0,0,0.6)", color: "rgba(221,221,245,0.7)" }}
                      >
                        {fc.count}회
                      </div>
                    </button>
                  );
                })}
              </div>
            </section>
          )}

          {/* Non-face PII */}
          {pii_candidates.length > 0 && (
            <section
              className="rounded-2xl p-5"
              style={{ background: "var(--card)", border: "1px solid var(--border)" }}
            >
              <div className="mb-4">
                <h3 className="font-semibold text-sm" style={{ color: "var(--text)" }}>
                  🔒 기타 개인정보
                </h3>
                <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>
                  마스킹할 항목을 선택하세요. 기본값은 전체 마스킹입니다.
                </p>
              </div>

              <div className="space-y-3">
                {piiTypes.map((type) => {
                  const items  = pii_candidates.filter((p) => p.pii_type === type);
                  const masked = maskedTypes.has(type);
                  return (
                    <div
                      key={type}
                      className="rounded-xl p-3 transition-all"
                      style={{
                        border: masked
                          ? "1px solid rgba(245,158,11,0.25)"
                          : "1px solid var(--border)",
                        background: masked
                          ? "rgba(245,158,11,0.06)"
                          : "rgba(255,255,255,0.01)",
                      }}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium" style={{ color: "var(--text)" }}>
                          {PII_LABEL[type] ?? type}
                          <span className="ml-1.5 text-xs" style={{ color: "var(--muted)" }}>
                            {items.length}개
                          </span>
                        </span>
                        <button
                          onClick={() => togglePiiType(type)}
                          className="text-xs px-3 py-1 rounded-full font-semibold transition-all"
                          style={masked ? {
                            background: "linear-gradient(135deg, #d97706, #f59e0b)",
                            color: "#fff",
                          } : {
                            background: "rgba(255,255,255,0.05)",
                            color: "var(--muted)",
                            border: "1px solid var(--border)",
                          }}
                        >
                          {masked ? "마스킹 ON" : "마스킹 OFF"}
                        </button>
                      </div>

                      <div className="flex gap-2 flex-wrap">
                        {items.map((p) => (
                          <img
                            key={p.object_id}
                            src={p.thumbnail_url}
                            alt=""
                            className="rounded-lg object-cover"
                            style={{
                              height: 52,
                              width: 80,
                              opacity: masked ? 1 : 0.3,
                              border: masked ? "1px solid rgba(245,158,11,0.4)" : "1px solid var(--border)",
                            }}
                          />
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {face_clusters.length === 0 && pii_candidates.length === 0 && (
            <div
              className="rounded-2xl p-10 text-center"
              style={{ background: "var(--card)", border: "1px solid var(--border)" }}
            >
              <p className="font-medium mb-1" style={{ color: "var(--text)" }}>감지된 PII가 없습니다</p>
              <p className="text-sm" style={{ color: "var(--muted)" }}>원본 영상이 그대로 저장됩니다</p>
            </div>
          )}

          {/* Submit footer summary */}
          <div
            className="glass rounded-2xl px-5 py-4 flex items-center justify-between"
            style={{ background: "rgba(124,58,237,0.06)", border: "1px solid rgba(124,58,237,0.2)" }}
          >
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              얼굴&nbsp;
              <span style={{ color: "var(--text)" }}>
                {face_clusters.length - protectedFaces.size}/{face_clusters.length}
              </span>
              &nbsp;블러 ·&nbsp;PII&nbsp;
              <span style={{ color: "var(--text)" }}>{maskedTypes.size}</span>
              종 마스킹
            </p>
            <button
              onClick={() => onSubmit([...protectedFaces], [...maskedTypes])}
              className="grad-btn rounded-xl px-6 py-2 text-sm font-bold text-white flex items-center gap-2"
            >
              마스킹 시작 →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function NothingDetectedView({ sceneType, onDone }) {
  return (
    <div className="flex flex-col h-full items-center justify-center gap-6 p-8 animate-fade-in">
      <div
        className="rounded-2xl p-10 flex flex-col items-center gap-4 max-w-md w-full text-center"
        style={{ background: "var(--card)", border: "1px solid var(--border)" }}
      >
        <div
          className="w-16 h-16 rounded-full flex items-center justify-center"
          style={{ background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.3)" }}
        >
          <span className="text-3xl">🛡</span>
        </div>

        <div>
          <p className="text-lg font-bold mb-1" style={{ color: "var(--text)" }}>
            감지된 개인정보 없음
          </p>
          <p className="text-sm" style={{ color: "var(--muted)" }}>
            이 영상에서 마스킹이 필요한 얼굴이나 개인정보가 감지되지 않았습니다.
            <br />원본 영상을 그대로 결과로 저장합니다.
          </p>
        </div>

        {sceneType && (
          <div
            className="text-xs px-3 py-1.5 rounded-full"
            style={{
              background: "rgba(255,255,255,0.05)",
              border: "1px solid var(--border)",
              color: "var(--muted)",
            }}
          >
            씬 유형: {SCENE_LABEL[sceneType] ?? sceneType}
          </div>
        )}

        <button
          onClick={onDone}
          className="grad-btn rounded-xl px-8 py-3 text-sm font-bold text-white w-full mt-2"
        >
          완료 처리
        </button>
      </div>
    </div>
  );
}

function GuidelineItem({ item }) {
  const isWarning = item.level === "warning";
  const CATEGORY = { face: "얼굴", pii: "PII", scene: "씬" };
  return (
    <div
      className="rounded-xl px-3.5 py-3 animate-slide-up"
      style={{
        background: isWarning ? "rgba(245,158,11,0.07)" : "rgba(59,130,246,0.07)",
        border: `1px solid ${isWarning ? "rgba(245,158,11,0.2)" : "rgba(59,130,246,0.2)"}`,
      }}
    >
      <div className="flex items-start gap-2.5">
        <span className="text-sm flex-shrink-0 mt-px" style={{ color: isWarning ? "#f59e0b" : "#3b82f6" }}>
          {isWarning ? "⚠" : "ℹ"}
        </span>
        <div>
          <span
            className="text-[10px] font-bold uppercase tracking-wider mr-1.5"
            style={{ color: isWarning ? "#f59e0b" : "#3b82f6" }}
          >
            {CATEGORY[item.category] ?? item.category}
          </span>
          <span className="text-xs" style={{ color: isWarning ? "#fef3c7" : "rgba(221,221,245,0.85)" }}>
            {item.message}
          </span>
        </div>
      </div>
    </div>
  );
}
