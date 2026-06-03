import { useState } from "react";

const PII_LABEL = {
  document:  "문서",
  screen:    "화면/스크린",
  nameplate: "명패",
  id_card:   "신분증",
  license_plate: "번호판",
  brand_logo: "상품 로고/상표",
};

export default function CandidateSelector({ candidates, onSubmit }) {
  const { scene_type, face_clusters, pii_candidates } = candidates;

  // Faces: click to PROTECT (keep unblurred). Default = all blurred.
  const [protectedFaces, setProtectedFaces] = useState(new Set());

  // Non-face PII: grouped by type. Default = all masked.
  const piiTypes = [...new Set(pii_candidates.map((p) => p.pii_type))];
  const [maskedTypes, setMaskedTypes] = useState(new Set(piiTypes));

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

  function handleSubmit() {
    onSubmit([...protectedFaces], [...maskedTypes]);
  }

  const sceneLabel = {
    meeting: "회의", lecture: "강의", interview: "인터뷰",
    public: "공공장소", other: "기타",
  }[scene_type] ?? scene_type;

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-6">
      <div className="w-full max-w-2xl space-y-8">

        {/* Header */}
        <div>
          <div className="text-xs text-gray-500 mb-1">씬 분석 결과: {sceneLabel}</div>
          <h2 className="text-white text-xl font-bold">마스킹 대상 선택</h2>
          <p className="text-gray-400 text-sm mt-1">
            얼굴은 <span className="text-blue-400">보호할 인물</span>을 클릭하세요.
            나머지는 자동으로 블러됩니다.
          </p>
        </div>

        {/* Face section */}
        {face_clusters.length > 0 && (
          <section>
            <h3 className="text-gray-300 font-semibold text-sm mb-3">
              얼굴 — {face_clusters.length}명 감지
            </h3>
            <div className="grid grid-cols-4 sm:grid-cols-5 gap-3">
              {face_clusters.map((fc) => {
                const protected_ = protectedFaces.has(fc.cluster_id);
                return (
                  <button
                    key={fc.cluster_id}
                    onClick={() => toggleFace(fc.cluster_id)}
                    className={`relative rounded-xl overflow-hidden border-2 transition-all ${
                      protected_
                        ? "border-blue-500 ring-2 ring-blue-500/40"
                        : "border-gray-700 hover:border-gray-500"
                    }`}
                  >
                    <img
                      src={fc.thumbnail_url}
                      alt={`인물 ${fc.cluster_id}`}
                      className="w-full aspect-square object-cover"
                    />
                    {protected_ && (
                      <div className="absolute inset-0 bg-blue-500/20 flex items-center justify-center">
                        <span className="text-white text-xl font-bold drop-shadow">✓</span>
                      </div>
                    )}
                    <div className="absolute bottom-0 inset-x-0 bg-black/60 text-xs text-gray-300 text-center py-0.5">
                      {fc.count}회
                    </div>
                  </button>
                );
              })}
            </div>
            <p className="text-gray-600 text-xs mt-2">
              {protectedFaces.size === 0
                ? "선택 없음 — 모든 얼굴 블러"
                : `${protectedFaces.size}명 보호, 나머지 블러`}
            </p>
          </section>
        )}

        {/* Non-face PII section */}
        {pii_candidates.length > 0 && (
          <section>
            <h3 className="text-gray-300 font-semibold text-sm mb-3">
              기타 개인정보 — 마스킹할 항목 선택
            </h3>
            <div className="space-y-4">
              {piiTypes.map((type) => {
                const items = pii_candidates.filter((p) => p.pii_type === type);
                const masked = maskedTypes.has(type);
                return (
                  <div key={type} className={`rounded-xl border p-3 transition-colors ${
                    masked ? "border-orange-600/50 bg-orange-950/20" : "border-gray-800 bg-gray-900"
                  }`}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-200 text-sm font-medium">
                        {PII_LABEL[type] ?? type} ({items.length}개)
                      </span>
                      <button
                        onClick={() => togglePiiType(type)}
                        className={`text-xs px-3 py-1 rounded-full font-semibold transition-colors ${
                          masked
                            ? "bg-orange-600 text-white"
                            : "bg-gray-700 text-gray-400"
                        }`}
                      >
                        {masked ? "마스킹 ON" : "마스킹 OFF"}
                      </button>
                    </div>
                    <div className="flex gap-2 flex-wrap">
                      {items.map((p) => (
                        <img
                          key={p.object_id}
                          src={p.thumbnail_url}
                          alt={p.pii_type}
                          className={`h-20 w-32 object-cover rounded-lg border ${
                            masked ? "border-orange-600/40" : "border-gray-700 opacity-40"
                          }`}
                        />
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* Empty state */}
        {face_clusters.length === 0 && pii_candidates.length === 0 && (
          <div className="bg-gray-900 rounded-xl p-8 text-center">
            <p className="text-gray-400">감지된 PII가 없습니다.</p>
            <p className="text-gray-600 text-sm mt-1">그대로 진행하면 원본 영상이 저장됩니다.</p>
          </div>
        )}

        {/* Submit */}
        <div className="flex items-center justify-between pt-2 border-t border-gray-800">
          <p className="text-gray-600 text-xs">
            얼굴 {face_clusters.length - protectedFaces.size}/{face_clusters.length} 블러 ·
            PII {maskedTypes.size}종 마스킹
          </p>
          <button
            onClick={handleSubmit}
            className="bg-blue-600 hover:bg-blue-500 text-white font-semibold px-6 py-2 rounded-lg transition-colors"
          >
            마스킹 시작
          </button>
        </div>
      </div>
    </div>
  );
}
