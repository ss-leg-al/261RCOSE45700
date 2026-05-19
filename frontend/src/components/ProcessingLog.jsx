import { useEffect, useRef } from "react";
import { useJobLog } from "../hooks/useJobLog";

export default function ProcessingLog({ jobId, title }) {
  const logs = useJobLog(jobId, true);
  const bottomRef = useRef();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-6">
      <div className="w-full max-w-xl">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-4 h-4 rounded-full bg-blue-500 animate-pulse" />
          <h2 className="text-white font-semibold text-lg">{title}</h2>
        </div>

        <div className="bg-gray-900 rounded-xl p-4 h-72 overflow-y-auto font-mono text-sm space-y-2">
          {logs.length === 0 && (
            <p className="text-gray-600">처리를 시작하고 있습니다...</p>
          )}
          {logs.map((log, i) => (
            <div key={i} className={`flex gap-2 ${log.step === "error" ? "text-red-400" : "text-gray-300"}`}>
              <span className="text-gray-600 shrink-0">
                {log.step === "error" ? "✗" : "›"}
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
