import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_biomechanics.metrics import (
    METRIC_FEATURES,
    BiomechanicalMetricTransformer,
    add_biomechanical_metrics,
)
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

    def test_metric_transformer_uses_training_scale_for_test_rows(self):
        train = make_sample_biomechanics(n_sessions=8, pitches_per_session=3)
        test = make_sample_biomechanics(n_sessions=2, pitches_per_session=3)
        test["max_pelvis_rotational_velo"] = test["max_pelvis_rotational_velo"] + 10000

        transformer = BiomechanicalMetricTransformer().fit(train)
        transformed_test = transformer.transform(test)
        full_data_metrics = add_biomechanical_metrics(test)

        self.assertIn("sequencing_efficiency", transformed_test.columns)
        self.assertNotAlmostEqual(
            transformed_test["sequencing_efficiency"].mean(),
            full_data_metrics["sequencing_efficiency"].mean(),
        )


if __name__ == "__main__":
    unittest.main()
