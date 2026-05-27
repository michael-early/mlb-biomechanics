import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_biomechanics.metrics import METRIC_FEATURES, add_biomechanical_metrics
from mlb_biomechanics.sample_data import make_sample_biomechanics


class BiomechanicalMetricsTest(unittest.TestCase):
    def test_adds_expected_metrics_without_missing_values(self):
        df = make_sample_biomechanics(n_sessions=8, pitches_per_session=3)
        metrics = add_biomechanical_metrics(df)

        for feature in METRIC_FEATURES:
            self.assertIn(feature, metrics.columns)
            self.assertFalse(metrics[feature].isna().any(), feature)

    def test_release_consistency_rewards_stable_session_rows(self):
        df = make_sample_biomechanics(n_sessions=4, pitches_per_session=4)
        metrics = add_biomechanical_metrics(df)

        self.assertLess(metrics["release_consistency_score"].min(), metrics["release_consistency_score"].max())


if __name__ == "__main__":
    unittest.main()

