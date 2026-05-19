import { useEffect, useState } from "react";

export function useJobLog(jobId, active) {
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    if (!jobId || !active) return;
    setLogs([]);
    const es = new EventSource(`/api/jobs/${jobId}/stream`);
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setLogs((prev) => [...prev, data]);
        if (data.event === "done" || data.step === "error") es.close();
      } catch {}
    };
    es.onerror = () => es.close();
    return () => es.close();
  }, [jobId, active]);

  return logs;
}
