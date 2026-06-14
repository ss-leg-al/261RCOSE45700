
from __future__ import annotations

import unittest
from pathlib import Path


class DoneViewContractTests(unittest.TestCase):
    def test_done_view_previews_colored_mask_but_downloads_private_output(self) -> None:
        done_view = Path("frontend/src/components/DoneView.jsx").read_text(encoding="utf-8")
        api_client = Path("frontend/src/api/client.js").read_text(encoding="utf-8")

        self.assertIn("maskPreviewUrl", api_client)
        self.assertIn('`/api/jobs/${jobId}/mask-preview`', api_client)

        self.assertIn("const showColoredPreview = Boolean(report?.colored_mask_enabled)", done_view)
        self.assertIn(
            "const resultVideoSrc = showColoredPreview ? api.maskPreviewUrl(jobId) : api.downloadUrl(jobId)",
            done_view,
        )
        self.assertIn('data-testid="download-output-link"', done_view)
        self.assertIn("href={api.downloadUrl(jobId)}", done_view)
        self.assertIn('"colored-mask-preview-video"', done_view)
        self.assertIn('"masked-output-video"', done_view)


if __name__ == "__main__":
    unittest.main()
