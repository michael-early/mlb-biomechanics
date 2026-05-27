import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_biomechanics import paths
from mlb_biomechanics.pipeline import build_mvp


class PipelineTest(unittest.TestCase):
    def test_build_mvp_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(paths, "RAW_DIR", root / "data" / "raw"), patch(
                "mlb_biomechanics.pipeline.PROCESSED_DIR", root / "data" / "processed"
            ), patch("mlb_biomechanics.pipeline.REPORTS_DIR", root / "reports"), patch(
                "mlb_biomechanics.report.REPORTS_DIR", root / "reports", create=True
            ):
                outputs = build_mvp()

            for output in outputs.values():
                self.assertTrue(os.path.exists(output), output)


if __name__ == "__main__":
    unittest.main()

