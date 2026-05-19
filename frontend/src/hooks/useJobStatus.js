import { useEffect, useRef } from "react";
import { api } from "../api/client";

export function useJobStatus(jobId, onStatusChange) {
  const ref = useRef(onStatusChange);
  ref.current = onStatusChange;

  useEffect(() => {
    if (!jobId) return;
    const timer = setInterval(async () => {
      try {
        const data = await api.getStatus(jobId);
        ref.current(data);
        if (["done", "failed"].includes(data.status)) {
          clearInterval(timer);
        }
      } catch {
        // network blip — keep polling
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [jobId]);
}
