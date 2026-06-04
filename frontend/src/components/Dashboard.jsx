import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";

const STATUS_LABEL = {
  pending:              { text: "대기 중",       color: "text-gray-400",  dot: "bg-gray-500" },
  detecting:            { text: "탐지 중",       color: "text-blue-400",  dot: "bg-blue-500 animate-pulse" },
  generating_guideline: { text: "가이드라인 생성", color: "text-purple-400", dot: "bg-purple-500 animate-pulse" },
  awaiting_selection:   { text: "선택 대기",     color: "text-yellow-400", dot: "bg-yellow-500" },
  masking:              { text: "마스킹 중",     color: "text-orange-400", dot: "bg-orange-500 animate-pulse" },
  done:                 { text: "완료",          color: "text-green-400",  dot: "bg-green-500" },
  failed:               { text: "오류",          color: "text-red-400",   dot: "bg-red-500" },
};

const SCENE_LABEL = {
  meeting: "회의", lecture: "강의", interview: "인터뷰", public: "공공장소", vehicle: "자동차/도로", other: "기타",
};

export default function Dashboard({ onSelectJob, onNewJob }) {
  const [jobs, setJobs] = useState([]);
  const inputRef = useRef();

  async function refresh() {
    try {
      const data = await api.listJobs();
      setJobs(data);
    } catch {
      // server not yet ready
    }
  }

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  }, []);

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (file) {
      e.target.value = "";
      onNewJob(file);
    }
  }

  const activeJobs = jobs.filter((j) => !["done", "failed"].includes(j.status));
  const finishedJobs = jobs.filter((j) => ["done", "failed"].includes(j.status));

  return (
    <div className="min-h-screen bg-gray-950 p-8">
      {/* Header */}
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-white text-2xl font-bold">SafeVlog3</h1>
            <p className="text-gray-500 text-sm mt-0.5">프라이버시 마스킹 대시보드</p>
          </div>
          <button
            onClick={() => inputRef.current?.click()}
            className="bg-blue-600 hover:bg-blue-500 text-white font-semibold px-5 py-2.5 rounded-xl transition-colors flex items-center gap-2"
          >
            <span className="text-lg leading-none">+</span>
            새 영상 업로드
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={handleFileChange}
          />
        </div>

        {/* Empty state */}
        {jobs.length === 0 && (
          <div className="border border-dashed border-gray-700 rounded-2xl p-16 text-center">
            <p className="text-gray-500 text-lg mb-2">처리한 영상이 없습니다</p>
            <p className="text-gray-600 text-sm">위 버튼으로 영상을 업로드하세요</p>
          </div>
        )}

        {/* Active jobs */}
        {activeJobs.length > 0 && (
          <section className="mb-8">
            <h2 className="text-gray-400 text-xs font-semibold uppercase tracking-wider mb-3">진행 중</h2>
            <div className="space-y-2">
              {activeJobs.map((job) => (
                <JobCard key={job.job_id} job={job} onSelect={onSelectJob} />
              ))}
            </div>
          </section>
        )}

        {/* Finished jobs */}
        {finishedJobs.length > 0 && (
          <section>
            <h2 className="text-gray-400 text-xs font-semibold uppercase tracking-wider mb-3">완료</h2>
            <div className="space-y-2">
              {finishedJobs.map((job) => (
                <JobCard key={job.job_id} job={job} onSelect={onSelectJob} />
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

function JobCard({ job, onSelect }) {
  const s = STATUS_LABEL[job.status] ?? STATUS_LABEL.pending;
  const isActionable = ["awaiting_selection", "done"].includes(job.status);

  return (
    <div
      className={`bg-gray-900 border rounded-xl px-5 py-4 flex items-center gap-4 transition-colors ${
        isActionable
          ? "border-gray-700 hover:border-gray-500 cursor-pointer"
          : "border-gray-800"
      }`}
      onClick={() => isActionable && onSelect(job)}
    >
      {/* Status dot */}
      <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${s.dot}`} />

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3">
          <span className="text-white font-mono text-sm">{job.job_id.slice(0, 8)}</span>
          {job.scene_type && (
            <span className="text-gray-600 text-xs">
              {SCENE_LABEL[job.scene_type] ?? job.scene_type}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 mt-0.5">
          <span className={`text-xs font-medium ${s.color}`}>{s.text}</span>
          {(job.face_count > 0 || job.pii_count > 0) && (
            <span className="text-gray-600 text-xs">
              {job.face_count > 0 && `얼굴 ${job.face_count}명`}
              {job.face_count > 0 && job.pii_count > 0 && " · "}
              {job.pii_count > 0 && `PII ${job.pii_count}종`}
            </span>
          )}
        </div>
      </div>

      {/* Action hint */}
      {job.status === "awaiting_selection" && (
        <span className="text-yellow-500 text-xs font-medium flex-shrink-0">선택 필요 →</span>
      )}
      {job.status === "done" && (
        <span className="text-green-500 text-xs font-medium flex-shrink-0">다운로드 →</span>
      )}
    </div>
  );
}
