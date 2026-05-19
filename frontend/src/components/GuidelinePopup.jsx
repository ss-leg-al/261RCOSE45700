const LEVEL_STYLE = {
  warning: {
    icon: "⚠",
    border: "border-yellow-600/40",
    bg: "bg-yellow-950/30",
    icon_color: "text-yellow-400",
    text_color: "text-yellow-100",
  },
  info: {
    icon: "ℹ",
    border: "border-blue-600/30",
    bg: "bg-blue-950/20",
    icon_color: "text-blue-400",
    text_color: "text-gray-200",
  },
};

const CATEGORY_LABEL = {
  face: "얼굴",
  pii: "PII",
  scene: "씬",
};

export default function GuidelinePopup({ items, onConfirm }) {
  const warnings = items.filter((i) => i.level === "warning");
  const infos = items.filter((i) => i.level === "info");
  const ordered = [...warnings, ...infos];

  return (
    /* Backdrop */
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm px-4">
      <div className="w-full max-w-lg bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl overflow-hidden">

        {/* Header */}
        <div className="px-6 pt-6 pb-4 border-b border-gray-800">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-yellow-400 text-lg">📋</span>
            <h2 className="text-white font-bold text-lg">편집 가이드라인</h2>
          </div>
          <p className="text-gray-400 text-sm">
            마스킹 전 확인이 필요한 항목입니다. 검토 후 선택 화면으로 이동하세요.
          </p>
        </div>

        {/* Items */}
        <div className="px-6 py-4 space-y-2 max-h-72 overflow-y-auto">
          {ordered.length === 0 ? (
            <p className="text-gray-500 text-sm text-center py-4">특이사항 없음</p>
          ) : (
            ordered.map((item, i) => {
              const style = LEVEL_STYLE[item.level] ?? LEVEL_STYLE.info;
              return (
                <div
                  key={i}
                  className={`flex items-start gap-3 rounded-xl border px-4 py-3 ${style.border} ${style.bg}`}
                >
                  <span className={`text-base leading-none mt-0.5 flex-shrink-0 ${style.icon_color}`}>
                    {style.icon}
                  </span>
                  <div className="flex-1 min-w-0">
                    <span className={`text-xs font-semibold uppercase tracking-wide ${style.icon_color} mr-2`}>
                      {CATEGORY_LABEL[item.category] ?? item.category}
                    </span>
                    <span className={`text-sm ${style.text_color}`}>{item.message}</span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-800 flex items-center justify-between">
          <span className="text-gray-600 text-xs">
            {warnings.length > 0 && `경고 ${warnings.length}개`}
            {warnings.length > 0 && infos.length > 0 && " · "}
            {infos.length > 0 && `안내 ${infos.length}개`}
          </span>
          <button
            onClick={onConfirm}
            className="bg-blue-600 hover:bg-blue-500 text-white font-semibold px-6 py-2 rounded-lg transition-colors"
          >
            확인 후 선택하기 →
          </button>
        </div>
      </div>
    </div>
  );
}
