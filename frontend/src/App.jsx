import { useState } from "react";
import { api } from "./api/client";
import { useJobStatus } from "./hooks/useJobStatus";
import Sidebar from "./components/Sidebar";
import EmptyView from "./components/EmptyView";
import ProcessingView from "./components/ProcessingView";
import ReviewPanel from "./components/ReviewPanel";
import DoneView from "./components/DoneView";

const INITIAL = {
  step: "idle",        // idle | uploading | detecting | generating_guideline | selecting | masking | done | failed
  jobId: null,
  candidates: null,
  guideline: [],
  error: null,
};

export default function App() {
  const [state, setState] = useState(INITIAL);

  useJobStatus(state.jobId, async ({ status, error }) => {
    if (
      status === "awaiting_selection" &&
      ["detecting", "generating_guideline"].includes(state.step)
    ) {
      const [candidates, guidelineData] = await Promise.all([
        api.getCandidates(state.jobId),
        api.getGuideline(state.jobId),
      ]);
      setState((s) => ({
        ...s,
        step: "selecting",
        candidates,
        guideline: guidelineData.items ?? [],
      }));
    } else if (status === "generating_guideline" && state.step === "detecting") {
      setState((s) => ({ ...s, step: "generating_guideline" }));
    } else if (status === "masking" && state.step !== "masking") {
      setState((s) => ({ ...s, step: "masking" }));
    } else if (status === "done") {
      setState((s) => ({ ...s, step: "done" }));
    } else if (status === "failed") {
      setState((s) => ({ ...s, step: "failed", error }));
    }
  });

  async function handleNewJob(file) {
    setState({ ...INITIAL, step: "uploading" });
    try {
      const { job_id } = await api.createJob(file);
      setState((s) => ({ ...s, step: "detecting", jobId: job_id }));
    } catch {
      setState({ ...INITIAL, step: "failed", error: "업로드에 실패했습니다." });
    }
  }

  function handleSelectJob(job) {
    if (job.status === "awaiting_selection") {
      (async () => {
        const [candidates, guidelineData] = await Promise.all([
          api.getCandidates(job.job_id),
          api.getGuideline(job.job_id),
        ]);
        setState({
          ...INITIAL,
          step: "selecting",
          jobId: job.job_id,
          candidates,
          guideline: guidelineData.items ?? [],
        });
      })();
    } else if (job.status === "done") {
      setState({ ...INITIAL, step: "done", jobId: job.job_id });
    } else if (["detecting", "masking", "generating_guideline"].includes(job.status)) {
      setState({ ...INITIAL, step: job.status, jobId: job.job_id });
    }
  }

  async function handleSkip() {
    await api.skipJob(state.jobId);
    setState((s) => ({ ...s, step: "done" }));
  }

  async function handleSelection(protectedFaceIds, maskedPiiTypes, maskedPiiObjectIds = [], sam3Mode = "normal") {
    setState((s) => ({ ...s, step: "masking" }));
    await api.submitSelection(state.jobId, protectedFaceIds, maskedPiiTypes, maskedPiiObjectIds, sam3Mode);
  }

  const { step, jobId, candidates, guideline, error } = state;
  const isProcessing = ["detecting", "generating_guideline", "masking"].includes(step);

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--bg)" }}>
      {/* Sidebar */}
      <Sidebar
        activeJobId={jobId}
        activeStep={step}
        onSelectJob={handleSelectJob}
        onNewJob={handleNewJob}
      />

      {/* Center panel */}
      <main className="flex-1 overflow-auto grid-bg">
        {(step === "idle" || step === "uploading") && (
          <EmptyView uploading={step === "uploading"} onUpload={handleNewJob} />
        )}

        {isProcessing && (
          <ProcessingView jobId={jobId} step={step} />
        )}

        {step === "selecting" && (
          <ReviewPanel
            jobId={jobId}
            candidates={candidates}
            guideline={guideline}
            onSubmit={handleSelection}
            onSkip={handleSkip}
          />
        )}

        {step === "done" && (
          <DoneView jobId={jobId} onNewJob={handleNewJob} />
        )}

        {step === "failed" && (
          <div className="flex h-full items-center justify-center p-8 animate-fade-in">
            <div className="glass rounded-2xl p-10 text-center max-w-md">
              <div className="text-4xl mb-4">⚠</div>
              <p className="text-red-400 font-semibold text-lg mb-2">오류가 발생했습니다</p>
              <p className="text-sm mb-6" style={{ color: "var(--muted)" }}>{error}</p>
              <button
                onClick={() => setState(INITIAL)}
                className="glass rounded-xl px-6 py-2 text-sm font-medium transition-colors hover:text-white"
                style={{ color: "var(--muted)" }}
              >
                대시보드로 돌아가기
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
