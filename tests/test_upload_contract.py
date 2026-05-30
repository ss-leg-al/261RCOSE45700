from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main
from backend.agent.job_store import get_store, reset_store


class UploadContractTests(unittest.TestCase):
    def test_video_upload_endpoint_accepts_multipart_and_persists_job(self) -> None:
        job_id = "unit-upload-job"
        reset_store(job_id)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_upload = main.settings.UPLOAD_DIR
            main.settings.UPLOAD_DIR = str(root / "uploads")
            called: list[str] = []

            def fake_detection_phase(received_job_id: str) -> None:
                called.append(received_job_id)

            try:
                client = TestClient(main.app)
                with patch.object(main.uuid, "uuid4", return_value=job_id), \
                     patch.object(main, "run_detection_phase", side_effect=fake_detection_phase):
                    response = client.post(
                        "/api/jobs",
                        files={"file": ("sample.mp4", b"fake video bytes", "video/mp4")},
                    )

                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json(), {"job_id": job_id})
                self.assertEqual(called, [job_id])

                store = get_store(job_id)
                self.assertTrue(store.video_path)
                uploaded = Path(store.video_path)
                self.assertEqual(uploaded.name, "input.mp4")
                self.assertEqual(uploaded.read_bytes(), b"fake video bytes")
            finally:
                main.settings.UPLOAD_DIR = old_upload
                reset_store(job_id)

    def test_vite_dev_proxy_targets_backend_ipv4_loopback(self) -> None:
        """Guard against localhost resolving to ::1 while uvicorn binds 127.0.0.1."""
        vite_config = Path("frontend/vite.config.js").read_text(encoding="utf-8")
        self.assertIn('target: "http://127.0.0.1:8000"', vite_config)
        self.assertNotIn('"http://localhost:8000"', vite_config)


if __name__ == "__main__":
    unittest.main()
