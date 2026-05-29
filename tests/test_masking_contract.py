from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import BackgroundTasks

from backend.agent.job_store import PIICandidate, get_store, reset_store
from backend.agent.tools.masker import apply_polygon_mask
from backend.main import submit_selection
from backend.schemas import SelectionRequest
from backend.agent import pipeline


class MaskingContractTests(unittest.TestCase):
    def test_selection_is_category_level_only(self) -> None:
        job_id = "unit-selection-contract"
        reset_store(job_id)
        store = get_store(job_id)
        store.status = "awaiting_selection"
        store.pii_candidates = [
            PIICandidate(0, "document", "doc0.jpg", 0.9),
            PIICandidate(1, "document", "doc1.jpg", 0.8),
            PIICandidate(2, "screen", "screen0.jpg", 0.7),
        ]

        body = SelectionRequest(
            protected_face_cluster_ids=[],
            masked_pii_types=["document"],
            masked_pii_object_ids=[0],  # ignored by the public schema contract
        )
        self.assertFalse(hasattr(body, "masked_pii_object_ids"))

        submit_selection(job_id, body, BackgroundTasks())

        self.assertEqual(store.masked_pii_types, ["document"])
        self.assertFalse(hasattr(store, "masked_pii_object_ids"))

    def test_per_frame_masking_does_not_reuse_stale_polygon(self) -> None:
        job_id = "unit-no-stale-mask"
        reset_store(job_id)
        store = get_store(job_id)
        store.status = "awaiting_selection"
        store.video_path = "input.mp4"
        store.expected_pii = ["document"]
        store.masked_pii_types = ["document"]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_upload = pipeline.settings.UPLOAD_DIR
            old_output = pipeline.settings.OUTPUT_DIR
            old_overlay = pipeline.settings.DEBUG_MASK_OVERLAY
            pipeline.settings.UPLOAD_DIR = str(root / "uploads")
            pipeline.settings.OUTPUT_DIR = str(root / "outputs")
            pipeline.settings.DEBUG_MASK_OVERLAY = False
            try:
                def fake_extract_frames(_video_path, frames_dir, fps=None):
                    frames_dir = Path(frames_dir)
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    frames = []
                    for index in range(2):
                        frame = np.full((32, 32, 3), 255, dtype=np.uint8)
                        path = frames_dir / f"frame_{index:04d}.jpg"
                        cv2.imwrite(str(path), frame)
                        frames.append(path)
                    return frames

                def fake_detect_pii(image_path, _types, _threshold):
                    if Path(image_path).name == "frame_0000.jpg":
                        return [{
                            "type": "document",
                            "polygon": [[8, 8], [24, 8], [24, 24], [8, 24]],
                            "bbox_xyxy": [8, 8, 24, 24],
                            "confidence": 0.99,
                            "mask_strategy": "blackbox",
                        }]
                    return []

                def fake_compose(_frames_dir, out_path, fps):
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_path).write_bytes(b"fake video")

                with patch.object(pipeline, "get_video_fps", return_value=1.0), \
                     patch.object(pipeline, "extract_frames", side_effect=fake_extract_frames), \
                     patch.object(pipeline, "compose_video", side_effect=fake_compose), \
                     patch.object(pipeline, "sam3_available", return_value=True), \
                     patch.object(pipeline, "detect_pii", side_effect=fake_detect_pii):
                    pipeline.run_masking_phase(job_id)

                self.assertEqual(store.status, "done")
                masked_dir = Path(store.masked_frames_dir)
                first = cv2.imread(str(masked_dir / "frame_0000.jpg"))
                second = cv2.imread(str(masked_dir / "frame_0001.jpg"))
                self.assertLess(int(first[16, 16].mean()), 32)
                self.assertGreater(int(second[16, 16].mean()), 240)
                self.assertFalse(store.report["temporal_mask_cache_enabled"])
                self.assertFalse(store.report["debug_mask_overlay_enabled"])
                self.assertEqual(store.report["selected_pii_category_count"], 1)
                self.assertNotIn("masked_pii_object_ids", store.report)
            finally:
                pipeline.settings.UPLOAD_DIR = old_upload
                pipeline.settings.OUTPUT_DIR = old_output
                pipeline.settings.DEBUG_MASK_OVERLAY = old_overlay

    def test_debug_overlay_is_opt_in(self) -> None:
        polygon = [[2, 2], [12, 2], [12, 12], [2, 12]]
        plain = apply_polygon_mask(
            np.full((16, 16, 3), 255, dtype=np.uint8),
            polygon,
            "blackbox",
            overlay_color=None,
        )
        colored = apply_polygon_mask(
            np.full((16, 16, 3), 255, dtype=np.uint8),
            polygon,
            "blackbox",
            overlay_color=(0, 128, 255),
        )

        self.assertTrue(np.array_equal(plain[6, 6], np.array([0, 0, 0], dtype=np.uint8)))
        self.assertGreater(int(colored[6, 6].sum()), 0)


if __name__ == "__main__":
    unittest.main()
