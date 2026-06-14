from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from backend.agent.job_store import PIICandidate, get_store, reset_store
from backend.agent.tools.masker import apply_binary_mask, apply_polygon_mask
from backend.agent.tools import sam3_engine
from backend.main import app, submit_selection
from backend.schemas import SelectionRequest
from backend.agent import pipeline


class MaskingContractTests(unittest.TestCase):
    def test_selection_accepts_individual_pii_object_ids(self) -> None:
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
            masked_pii_object_ids=[0, 2],
        )

        submit_selection(job_id, body, BackgroundTasks())

        self.assertEqual(store.masked_pii_types, ["document"])
        self.assertEqual(store.masked_pii_object_ids, [0, 2])
        self.assertEqual(store.sam3_mode, "normal")

    def test_selection_accepts_sam3_mode_aliases(self) -> None:
        job_id = "unit-selection-sam3-mode"
        reset_store(job_id)
        store = get_store(job_id)
        store.status = "awaiting_selection"

        body = SelectionRequest.model_validate({
            "protected_face_cluster_ids": [],
            "masked_pii_types": ["document"],
            "masked_pii_object_ids": [],
            "sam3-mode": "정밀",
        })

        submit_selection(job_id, body, BackgroundTasks())

        self.assertEqual(store.sam3_mode, "precision")

    def test_brand_logo_uses_product_logo_prompts_and_ambient_fill(self) -> None:
        prompts = sam3_engine._TEXT_PROMPTS["brand_logo"]

        self.assertEqual(sam3_engine.MASK_STRATEGY["brand_logo"], "ambient_fill")
        self.assertIn("product logo", prompts)
        self.assertIn("product trademark", prompts)
        self.assertNotIn("company logo", prompts)
        self.assertNotIn("branded sign", prompts)

    def test_detection_phase_discovers_brand_logo_candidates_by_default(self) -> None:
        job_id = "unit-brand-logo-detection"
        reset_store(job_id)
        store = get_store(job_id)
        store.video_path = "input.mp4"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_upload = pipeline.settings.UPLOAD_DIR
            old_output = pipeline.settings.OUTPUT_DIR
            old_logo_enabled = pipeline.settings.BRAND_LOGO_DETECTION_ENABLED
            pipeline.settings.UPLOAD_DIR = str(root / "uploads")
            pipeline.settings.OUTPUT_DIR = str(root / "outputs")
            pipeline.settings.BRAND_LOGO_DETECTION_ENABLED = True
            try:
                def fake_extract_frames(_video_path, frames_dir, fps=None):
                    frames_dir = Path(frames_dir)
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    frame = np.full((32, 32, 3), 255, dtype=np.uint8)
                    path = frames_dir / "frame_0000.jpg"
                    cv2.imwrite(str(path), frame)
                    return [path]

                detect_calls: list[list[str]] = []

                def fake_detect_pii(_image_path, pii_types, _threshold, include_binary_mask=False):
                    detect_calls.append(list(pii_types))
                    return [{
                        "type": "brand_logo",
                        "polygon": [[8, 8], [24, 8], [24, 24], [8, 24]],
                        "bbox_xyxy": [8, 8, 24, 24],
                        "confidence": 0.91,
                        "mask_strategy": "ambient_fill",
                    }]

                with patch.object(pipeline, "get_video_fps", return_value=1.0), \
                     patch.object(pipeline, "extract_frames", side_effect=fake_extract_frames), \
                     patch.object(pipeline, "analyze_scene", return_value=("other", [])), \
                     patch.object(pipeline, "sam3_available", return_value=True), \
                     patch.object(pipeline, "detect_pii", side_effect=fake_detect_pii), \
                     patch.object(pipeline, "detect_faces", return_value=[]), \
                     patch.object(pipeline, "generate_guideline", return_value=[]):
                    pipeline.run_detection_phase(job_id)

                self.assertEqual(store.status, "awaiting_selection")
                self.assertEqual(store.expected_pii, [])
                self.assertTrue(detect_calls)
                self.assertIn("brand_logo", detect_calls[0])
                self.assertEqual(store.detection_pii_types, ["brand_logo"])
                self.assertEqual(store.deterministic_pii_types_added, ["brand_logo"])
                self.assertEqual(len(store.pii_candidates), 1)
                self.assertEqual(store.pii_candidates[0].pii_type, "brand_logo")
                self.assertEqual(store.pii_candidates[0].mask_strategy, "ambient_fill")
            finally:
                pipeline.settings.UPLOAD_DIR = old_upload
                pipeline.settings.OUTPUT_DIR = old_output
                pipeline.settings.BRAND_LOGO_DETECTION_ENABLED = old_logo_enabled

    def test_detection_phase_preserves_empty_detection_metadata_when_logo_disabled(self) -> None:
        job_id = "unit-brand-logo-disabled"
        reset_store(job_id)
        store = get_store(job_id)
        store.video_path = "input.mp4"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_upload = pipeline.settings.UPLOAD_DIR
            old_output = pipeline.settings.OUTPUT_DIR
            old_logo_enabled = pipeline.settings.BRAND_LOGO_DETECTION_ENABLED
            pipeline.settings.UPLOAD_DIR = str(root / "uploads")
            pipeline.settings.OUTPUT_DIR = str(root / "outputs")
            pipeline.settings.BRAND_LOGO_DETECTION_ENABLED = False
            try:
                def fake_extract_frames(_video_path, frames_dir, fps=None):
                    frames_dir = Path(frames_dir)
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    frame = np.full((32, 32, 3), 255, dtype=np.uint8)
                    path = frames_dir / "frame_0000.jpg"
                    cv2.imwrite(str(path), frame)
                    return [path]

                with patch.object(pipeline, "get_video_fps", return_value=1.0), \
                     patch.object(pipeline, "extract_frames", side_effect=fake_extract_frames), \
                     patch.object(pipeline, "analyze_scene", return_value=("other", [])), \
                     patch.object(pipeline, "sam3_available", return_value=True), \
                     patch.object(pipeline, "detect_pii") as detect_pii_mock, \
                     patch.object(pipeline, "detect_faces", return_value=[]), \
                     patch.object(pipeline, "generate_guideline", return_value=[]):
                    pipeline.run_detection_phase(job_id)

                detect_pii_mock.assert_not_called()
                self.assertEqual(store.status, "awaiting_selection")
                self.assertEqual(store.expected_pii, [])
                self.assertEqual(store.detection_pii_types, [])
                self.assertEqual(store.deterministic_pii_types_added, [])
                self.assertEqual(store.pii_candidates, [])
            finally:
                pipeline.settings.UPLOAD_DIR = old_upload
                pipeline.settings.OUTPUT_DIR = old_output
                pipeline.settings.BRAND_LOGO_DETECTION_ENABLED = old_logo_enabled

    def test_keyframe_tail_detection_does_not_reuse_stale_polygon(self) -> None:
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
            old_interpolation = pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED
            old_interval = pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL
            old_dilate = pipeline.settings.SAM3_MASK_DILATE_PX
            pipeline.settings.UPLOAD_DIR = str(root / "uploads")
            pipeline.settings.OUTPUT_DIR = str(root / "outputs")
            pipeline.settings.DEBUG_MASK_OVERLAY = False
            pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = True
            pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL = 3
            pipeline.settings.SAM3_MASK_DILATE_PX = 0
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

                def fake_detect_pii(image_path, _types, _threshold, include_binary_mask=False):
                    if Path(image_path).name == "frame_0000.jpg":
                        binary_mask = np.zeros((32, 32), dtype=np.uint8)
                        binary_mask[8:24, 8:24] = 255
                        return [{
                            "type": "document",
                            "polygon": [[8, 8], [24, 8], [24, 24], [8, 24]],
                            "bbox_xyxy": [8, 8, 24, 24],
                            "confidence": 0.99,
                            "mask_strategy": "blackbox",
                            **({"binary_mask": binary_mask} if include_binary_mask else {}),
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
                self.assertTrue(store.report["temporal_mask_interpolation_enabled"])
                self.assertEqual(store.report["masking_mode"], "lowres_keyframe_interpolation")
                self.assertFalse(store.report["debug_mask_overlay_enabled"])
                self.assertEqual(store.report["selected_pii_category_count"], 1)
                self.assertEqual(store.report["masked_pii_object_ids"], [])
            finally:
                pipeline.settings.UPLOAD_DIR = old_upload
                pipeline.settings.OUTPUT_DIR = old_output
                pipeline.settings.DEBUG_MASK_OVERLAY = old_overlay
                pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = old_interpolation
                pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL = old_interval
                pipeline.settings.SAM3_MASK_DILATE_PX = old_dilate

    def test_keyframe_interpolation_masks_middle_frames_without_sam3_calls(self) -> None:
        job_id = "unit-keyframe-interpolation"
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
            old_interpolation = pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED
            old_interval = pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL
            old_dilate = pipeline.settings.SAM3_MASK_DILATE_PX
            pipeline.settings.UPLOAD_DIR = str(root / "uploads")
            pipeline.settings.OUTPUT_DIR = str(root / "outputs")
            pipeline.settings.DEBUG_MASK_OVERLAY = False
            pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = True
            pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL = 3
            pipeline.settings.SAM3_MASK_DILATE_PX = 0
            try:
                def fake_extract_frames(_video_path, frames_dir, fps=None):
                    frames_dir = Path(frames_dir)
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    frames = []
                    for index in range(4):
                        frame = np.full((32, 32, 3), 255, dtype=np.uint8)
                        path = frames_dir / f"frame_{index:04d}.jpg"
                        cv2.imwrite(str(path), frame)
                        frames.append(path)
                    return frames

                detect_calls: list[str] = []

                def fake_detect_pii(image_path, _types, _threshold, include_binary_mask=False):
                    name = Path(image_path).name
                    detect_calls.append(name)
                    if name == "frame_0000.jpg":
                        bbox = [8, 8, 16, 16]
                    elif name == "frame_0003.jpg":
                        bbox = [11, 11, 19, 19]
                    else:
                        raise AssertionError(f"SAM3 should not run on intermediate frame: {name}")

                    binary_mask = np.zeros((32, 32), dtype=np.uint8)
                    x1, y1, x2, y2 = bbox
                    binary_mask[y1:y2, x1:x2] = 255
                    return [{
                        "type": "document",
                        "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        "bbox_xyxy": bbox,
                        "confidence": 0.99,
                        "mask_strategy": "blackbox",
                        **({"binary_mask": binary_mask} if include_binary_mask else {}),
                    }]

                def fake_compose(_frames_dir, out_path, fps):
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_path).write_bytes(b"fake video")

                with patch.object(pipeline, "get_video_fps", return_value=1.0), \
                     patch.object(pipeline, "extract_frames", side_effect=fake_extract_frames), \
                     patch.object(pipeline, "compose_video", side_effect=fake_compose), \
                     patch.object(pipeline, "sam3_available", return_value=True), \
                     patch.object(pipeline, "detect_pii", side_effect=fake_detect_pii):
                    pipeline.run_masking_phase(job_id)

                self.assertEqual(detect_calls, ["frame_0000.jpg", "frame_0003.jpg"])
                masked_dir = Path(store.masked_frames_dir)
                middle_1 = cv2.imread(str(masked_dir / "frame_0001.jpg"))
                middle_2 = cv2.imread(str(masked_dir / "frame_0002.jpg"))
                self.assertLess(int(middle_1[13, 13].mean()), 32)
                self.assertLess(int(middle_2[14, 14].mean()), 32)
                self.assertEqual(store.report["sam3_mask_keyframes_detected"], 2)
                self.assertEqual(store.report["sam3_mask_keyframe_interval"], 3)
            finally:
                pipeline.settings.UPLOAD_DIR = old_upload
                pipeline.settings.OUTPUT_DIR = old_output
                pipeline.settings.DEBUG_MASK_OVERLAY = old_overlay
                pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = old_interpolation
                pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL = old_interval
                pipeline.settings.SAM3_MASK_DILATE_PX = old_dilate

    def test_precision_sam3_mode_segments_every_frame(self) -> None:
        job_id = "unit-precision-sam3-mode"
        reset_store(job_id)
        store = get_store(job_id)
        store.status = "awaiting_selection"
        store.video_path = "input.mp4"
        store.expected_pii = ["document"]
        store.masked_pii_types = ["document"]
        store.sam3_mode = "precision"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_upload = pipeline.settings.UPLOAD_DIR
            old_output = pipeline.settings.OUTPUT_DIR
            old_overlay = pipeline.settings.DEBUG_MASK_OVERLAY
            old_interpolation = pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED
            old_interval = pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL
            old_dilate = pipeline.settings.SAM3_MASK_DILATE_PX
            pipeline.settings.UPLOAD_DIR = str(root / "uploads")
            pipeline.settings.OUTPUT_DIR = str(root / "outputs")
            pipeline.settings.DEBUG_MASK_OVERLAY = False
            pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = True
            pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL = 3
            pipeline.settings.SAM3_MASK_DILATE_PX = 0
            try:
                def fake_extract_frames(_video_path, frames_dir, fps=None):
                    frames_dir = Path(frames_dir)
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    frames = []
                    for index in range(4):
                        frame = np.full((32, 32, 3), 255, dtype=np.uint8)
                        path = frames_dir / f"frame_{index:04d}.jpg"
                        cv2.imwrite(str(path), frame)
                        frames.append(path)
                    return frames

                detect_calls: list[str] = []

                def fake_detect_pii(image_path, _types, _threshold, include_binary_mask=False):
                    name = Path(image_path).name
                    detect_calls.append(name)
                    binary_mask = np.zeros((32, 32), dtype=np.uint8)
                    binary_mask[8:16, 8:16] = 255
                    return [{
                        "type": "document",
                        "polygon": [[8, 8], [16, 8], [16, 16], [8, 16]],
                        "bbox_xyxy": [8, 8, 16, 16],
                        "confidence": 0.99,
                        "mask_strategy": "blackbox",
                        **({"binary_mask": binary_mask} if include_binary_mask else {}),
                    }]

                def fake_compose(_frames_dir, out_path, fps):
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_path).write_bytes(b"fake video")

                with patch.object(pipeline, "get_video_fps", return_value=1.0), \
                     patch.object(pipeline, "extract_frames", side_effect=fake_extract_frames), \
                     patch.object(pipeline, "compose_video", side_effect=fake_compose), \
                     patch.object(pipeline, "sam3_available", return_value=True), \
                     patch.object(pipeline, "detect_pii", side_effect=fake_detect_pii):
                    pipeline.run_masking_phase(job_id)

                self.assertEqual(detect_calls, [
                    "frame_0000.jpg",
                    "frame_0001.jpg",
                    "frame_0002.jpg",
                    "frame_0003.jpg",
                ])
                self.assertEqual(store.report["sam3_mode"], "precision")
                self.assertEqual(store.report["sam3_mode_label"], "정밀")
                self.assertFalse(store.report["temporal_mask_interpolation_enabled"])
                self.assertEqual(store.report["sam3_mask_keyframe_interval"], 1)
                self.assertEqual(store.report["sam3_mask_keyframes_detected"], 4)
            finally:
                pipeline.settings.UPLOAD_DIR = old_upload
                pipeline.settings.OUTPUT_DIR = old_output
                pipeline.settings.DEBUG_MASK_OVERLAY = old_overlay
                pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = old_interpolation
                pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL = old_interval
                pipeline.settings.SAM3_MASK_DILATE_PX = old_dilate

    def test_individual_pii_selection_masks_only_selected_candidate(self) -> None:
        job_id = "unit-individual-pii-selection"
        reset_store(job_id)
        store = get_store(job_id)
        store.status = "awaiting_selection"
        store.video_path = "input.mp4"
        store.expected_pii = ["document"]
        store.masked_pii_types = ["document"]
        store.masked_pii_object_ids = [0]
        store.pii_candidates = [
            PIICandidate(0, "document", "left.jpg", 0.95, frame_index=0, bbox_xyxy=[4, 8, 12, 20]),
            PIICandidate(1, "document", "right.jpg", 0.95, frame_index=0, bbox_xyxy=[20, 8, 28, 20]),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_upload = pipeline.settings.UPLOAD_DIR
            old_output = pipeline.settings.OUTPUT_DIR
            old_overlay = pipeline.settings.DEBUG_MASK_OVERLAY
            old_interpolation = pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED
            old_interval = pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL
            old_dilate = pipeline.settings.SAM3_MASK_DILATE_PX
            pipeline.settings.UPLOAD_DIR = str(root / "uploads")
            pipeline.settings.OUTPUT_DIR = str(root / "outputs")
            pipeline.settings.DEBUG_MASK_OVERLAY = False
            pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = True
            pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL = 3
            pipeline.settings.SAM3_MASK_DILATE_PX = 0
            try:
                def fake_extract_frames(_video_path, frames_dir, fps=None):
                    frames_dir = Path(frames_dir)
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    frames = []
                    for index in range(4):
                        frame = np.full((32, 32, 3), 255, dtype=np.uint8)
                        path = frames_dir / f"frame_{index:04d}.jpg"
                        cv2.imwrite(str(path), frame)
                        frames.append(path)
                    return frames

                detect_calls: list[str] = []

                def fake_detect_pii(image_path, _types, _threshold, include_binary_mask=False):
                    detect_calls.append(Path(image_path).name)
                    detections = []
                    for bbox in ([4, 8, 12, 20], [20, 8, 28, 20]):
                        binary_mask = np.zeros((32, 32), dtype=np.uint8)
                        x1, y1, x2, y2 = bbox
                        binary_mask[y1:y2, x1:x2] = 255
                        detections.append({
                            "type": "document",
                            "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                            "bbox_xyxy": bbox,
                            "confidence": 0.99,
                            "mask_strategy": "blackbox",
                            **({"binary_mask": binary_mask} if include_binary_mask else {}),
                        })
                    return detections

                def fake_compose(_frames_dir, out_path, fps):
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_path).write_bytes(b"fake video")

                with patch.object(pipeline, "get_video_fps", return_value=1.0), \
                     patch.object(pipeline, "extract_frames", side_effect=fake_extract_frames), \
                     patch.object(pipeline, "compose_video", side_effect=fake_compose), \
                     patch.object(pipeline, "sam3_available", return_value=True), \
                     patch.object(pipeline, "detect_pii", side_effect=fake_detect_pii):
                    pipeline.run_masking_phase(job_id)

                self.assertEqual(detect_calls, ["frame_0000.jpg", "frame_0003.jpg"])
                masked_dir = Path(store.masked_frames_dir)
                first = cv2.imread(str(masked_dir / "frame_0000.jpg"))
                middle = cv2.imread(str(masked_dir / "frame_0001.jpg"))
                self.assertLess(int(first[10, 6].mean()), 32)
                self.assertGreater(int(first[10, 24].mean()), 240)
                self.assertLess(int(middle[10, 6].mean()), 32)
                self.assertGreater(int(middle[10, 24].mean()), 240)
                self.assertEqual(store.report["masked_pii_object_ids"], [0])
                self.assertEqual(store.report["selected_pii_object_count"], 1)
            finally:
                pipeline.settings.UPLOAD_DIR = old_upload
                pipeline.settings.OUTPUT_DIR = old_output
                pipeline.settings.DEBUG_MASK_OVERLAY = old_overlay
                pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = old_interpolation
                pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL = old_interval
                pipeline.settings.SAM3_MASK_DILATE_PX = old_dilate

    def test_completed_job_download_stays_private_and_preview_uses_colored_mask(self) -> None:
        job_id = "unit-completed-colored-preview"
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
            old_interpolation = pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED
            old_interval = pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL
            old_dilate = pipeline.settings.SAM3_MASK_DILATE_PX
            pipeline.settings.UPLOAD_DIR = str(root / "uploads")
            pipeline.settings.OUTPUT_DIR = str(root / "outputs")
            pipeline.settings.DEBUG_MASK_OVERLAY = True
            pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = True
            pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL = 3
            pipeline.settings.SAM3_MASK_DILATE_PX = 0
            try:
                def fake_extract_frames(_video_path, frames_dir, fps=None):
                    frames_dir = Path(frames_dir)
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    frame = np.full((32, 32, 3), 255, dtype=np.uint8)
                    path = frames_dir / "frame_0000.jpg"
                    cv2.imwrite(str(path), frame)
                    return [path]

                def fake_detect_pii(image_path, _types, _threshold, include_binary_mask=False):
                    binary_mask = np.zeros((32, 32), dtype=np.uint8)
                    binary_mask[8:24, 8:24] = 255
                    return [{
                        "type": "document",
                        "polygon": [[8, 8], [24, 8], [24, 24], [8, 24]],
                        "bbox_xyxy": [8, 8, 24, 24],
                        "confidence": 0.99,
                        "mask_strategy": "blackbox",
                        **({"binary_mask": binary_mask} if include_binary_mask else {}),
                    }]

                def fake_compose(frames_dir, out_path, fps):
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    if Path(frames_dir).name == "mask_preview_frames":
                        Path(out_path).write_bytes(b"colored preview video")
                    else:
                        Path(out_path).write_bytes(b"private download video")

                with patch.object(pipeline, "get_video_fps", return_value=1.0), \
                     patch.object(pipeline, "extract_frames", side_effect=fake_extract_frames), \
                     patch.object(pipeline, "compose_video", side_effect=fake_compose), \
                     patch.object(pipeline, "sam3_available", return_value=True), \
                     patch.object(pipeline, "detect_pii", side_effect=fake_detect_pii):
                    pipeline.run_masking_phase(job_id)

                self.assertEqual(store.status, "done")
                self.assertTrue(store.report["colored_mask_enabled"])
                self.assertTrue(store.report["colored_mask_preview_enabled"])
                self.assertFalse(store.report["debug_mask_overlay_enabled"])
                self.assertEqual(store.report["mask_colors"]["document"], "#f59e0b")
                self.assertEqual(Path(store.output_video_path).read_bytes(), b"private download video")
                self.assertEqual(Path(store.mask_preview_video_path).read_bytes(), b"colored preview video")

                download_frame = cv2.imread(str(Path(store.masked_frames_dir) / "frame_0000.jpg"))
                preview = cv2.imread(str(Path(store.mask_preview_frames_dir) / "frame_0000.jpg"))
                download_pixel = download_frame[16, 16].astype(int)
                preview_pixel = preview[16, 16].astype(int)
                unmasked_pixel = preview[4, 4].astype(int)

                # The actual downloadable output remains the normal privacy
                # transform, while the completed-state preview video shows a
                # colored mask so the user can see what area was affected.
                self.assertTrue(np.array_equal(download_pixel, np.array([0, 0, 0])))
                self.assertGreater(int(preview_pixel.sum()), 40)
                self.assertGreater(preview_pixel[2], preview_pixel[1])
                self.assertGreater(preview_pixel[1], preview_pixel[0])
                self.assertGreater(int(unmasked_pixel.mean()), 240)

                client = TestClient(app)
                download = client.get(f"/api/jobs/{job_id}/download")
                mask_preview = client.get(f"/api/jobs/{job_id}/mask-preview")
                self.assertEqual(download.status_code, 200)
                self.assertEqual(mask_preview.status_code, 200)
                self.assertEqual(download.content, b"private download video")
                self.assertEqual(mask_preview.content, b"colored preview video")
            finally:
                pipeline.settings.UPLOAD_DIR = old_upload
                pipeline.settings.OUTPUT_DIR = old_output
                pipeline.settings.DEBUG_MASK_OVERLAY = old_overlay
                pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = old_interpolation
                pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL = old_interval
                pipeline.settings.SAM3_MASK_DILATE_PX = old_dilate

    def test_brand_logo_selection_masks_and_reports_colored_preview(self) -> None:
        job_id = "unit-brand-logo-masking"
        reset_store(job_id)
        store = get_store(job_id)
        store.status = "awaiting_selection"
        store.video_path = "input.mp4"
        store.expected_pii = []
        store.detection_pii_types = ["brand_logo"]
        store.deterministic_pii_types_added = ["brand_logo"]
        store.masked_pii_types = ["brand_logo"]
        store.masked_pii_object_ids = [0]
        store.pii_candidates = [
            PIICandidate(
                0,
                "brand_logo",
                "logo.jpg",
                0.95,
                frame_index=0,
                bbox_xyxy=[8, 8, 24, 24],
                mask_strategy="ambient_fill",
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_upload = pipeline.settings.UPLOAD_DIR
            old_output = pipeline.settings.OUTPUT_DIR
            old_overlay = pipeline.settings.DEBUG_MASK_OVERLAY
            old_interpolation = pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED
            old_interval = pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL
            old_dilate = pipeline.settings.SAM3_MASK_DILATE_PX
            pipeline.settings.UPLOAD_DIR = str(root / "uploads")
            pipeline.settings.OUTPUT_DIR = str(root / "outputs")
            pipeline.settings.DEBUG_MASK_OVERLAY = False
            pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = True
            pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL = 3
            pipeline.settings.SAM3_MASK_DILATE_PX = 0
            try:
                def fake_extract_frames(_video_path, frames_dir, fps=None):
                    frames_dir = Path(frames_dir)
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    frame = np.full((32, 32, 3), 240, dtype=np.uint8)
                    frame[8:24, 8:24] = np.array([0, 0, 255], dtype=np.uint8)
                    path = frames_dir / "frame_0000.jpg"
                    cv2.imwrite(str(path), frame)
                    return [path]

                detect_types: list[list[str]] = []

                def fake_detect_pii(_image_path, pii_types, _threshold, include_binary_mask=False):
                    detect_types.append(list(pii_types))
                    binary_mask = np.zeros((32, 32), dtype=np.uint8)
                    binary_mask[8:24, 8:24] = 255
                    return [{
                        "type": "brand_logo",
                        "polygon": [[8, 8], [24, 8], [24, 24], [8, 24]],
                        "bbox_xyxy": [8, 8, 24, 24],
                        "confidence": 0.99,
                        "mask_strategy": "ambient_fill",
                        **({"binary_mask": binary_mask} if include_binary_mask else {}),
                    }]

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
                self.assertEqual(detect_types, [["brand_logo"]])
                self.assertEqual(store.report["masked_pii_types"], ["brand_logo"])
                self.assertEqual(store.report["masked_pii_object_ids"], [0])
                self.assertEqual(store.report["mask_colors"]["brand_logo"], "#ec4899")
                self.assertEqual(store.report["detection_pii_types"], ["brand_logo"])
                self.assertEqual(store.report["deterministic_pii_types_added"], ["brand_logo"])
                self.assertEqual(store.report["total_pii_masked"], 1)

                masked = cv2.imread(str(Path(store.masked_frames_dir) / "frame_0000.jpg"))
                preview = cv2.imread(str(Path(store.mask_preview_frames_dir) / "frame_0000.jpg"))
                self.assertGreater(int(masked[16, 16].mean()), 180)
                self.assertFalse(np.array_equal(masked[16, 16], np.array([0, 0, 255], dtype=np.uint8)))
                self.assertGreater(int(preview[16, 16].sum()), 40)
            finally:
                pipeline.settings.UPLOAD_DIR = old_upload
                pipeline.settings.OUTPUT_DIR = old_output
                pipeline.settings.DEBUG_MASK_OVERLAY = old_overlay
                pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = old_interpolation
                pipeline.settings.SAM3_MASK_KEYFRAME_INTERVAL = old_interval
                pipeline.settings.SAM3_MASK_DILATE_PX = old_dilate

    def test_report_preserves_recorded_empty_detection_metadata(self) -> None:
        job_id = "unit-empty-detection-metadata-report"
        reset_store(job_id)
        store = get_store(job_id)
        store.status = "awaiting_selection"
        store.video_path = "input.mp4"
        store.expected_pii = []
        store.detection_pii_types = []
        store.deterministic_pii_types_added = []
        store.masked_pii_types = []
        store.masked_pii_object_ids = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_upload = pipeline.settings.UPLOAD_DIR
            old_output = pipeline.settings.OUTPUT_DIR
            old_interpolation = pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED
            old_logo_enabled = pipeline.settings.BRAND_LOGO_DETECTION_ENABLED
            pipeline.settings.UPLOAD_DIR = str(root / "uploads")
            pipeline.settings.OUTPUT_DIR = str(root / "outputs")
            pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = True
            pipeline.settings.BRAND_LOGO_DETECTION_ENABLED = True
            try:
                def fake_extract_frames(_video_path, frames_dir, fps=None):
                    frames_dir = Path(frames_dir)
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    frame = np.full((16, 16, 3), 255, dtype=np.uint8)
                    path = frames_dir / "frame_0000.jpg"
                    cv2.imwrite(str(path), frame)
                    return [path]

                def fake_compose(_frames_dir, out_path, fps):
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_path).write_bytes(b"fake video")

                with patch.object(pipeline, "get_video_fps", return_value=1.0), \
                     patch.object(pipeline, "extract_frames", side_effect=fake_extract_frames), \
                     patch.object(pipeline, "compose_video", side_effect=fake_compose), \
                     patch.object(pipeline, "sam3_available", return_value=True), \
                     patch.object(pipeline, "detect_pii") as detect_pii_mock:
                    pipeline.run_masking_phase(job_id)

                detect_pii_mock.assert_not_called()
                self.assertEqual(store.status, "done")
                self.assertEqual(store.report["detection_pii_types"], [])
                self.assertEqual(store.report["deterministic_pii_types_added"], [])
                self.assertFalse(store.report["colored_mask_enabled"])
            finally:
                pipeline.settings.UPLOAD_DIR = old_upload
                pipeline.settings.OUTPUT_DIR = old_output
                pipeline.settings.SAM3_MASK_INTERPOLATION_ENABLED = old_interpolation
                pipeline.settings.BRAND_LOGO_DETECTION_ENABLED = old_logo_enabled

    def test_ambient_fill_masks_with_surrounding_color(self) -> None:
        img = np.full((32, 32, 3), 230, dtype=np.uint8)
        img[10:22, 10:22] = np.array([0, 0, 255], dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:22, 10:22] = 255

        masked = apply_binary_mask(img.copy(), mask, "ambient_fill")

        self.assertGreater(int(masked[16, 16].mean()), 180)
        self.assertFalse(np.array_equal(masked[16, 16], np.array([0, 0, 255], dtype=np.uint8)))
        self.assertFalse(np.array_equal(masked[16, 16], np.array([0, 0, 0], dtype=np.uint8)))

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
