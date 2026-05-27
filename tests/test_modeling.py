import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_biomechanics.metrics import add_biomechanical_metrics
from mlb_biomechanics.modeling import evaluate_velocity_model, statcast_performance_bridge
from mlb_biomechanics.sample_data import make_sample_biomechanics, make_sample_statcast


class ModelingTest(unittest.TestCase):
    def test_velocity_model_beats_baseline_on_sample_data(self):
        df = add_biomechanical_metrics(make_sample_biomechanics(n_sessions=40, pitches_per_session=4))
        results = evaluate_velocity_model(df)

        self.assertGreater(results["test_rows"], 0)
        self.assertLess(results["ridge"]["rmse"], results["baseline"]["rmse"])
        self.assertIn("permutation_importance", results)

    def test_statcast_bridge_returns_pitch_type_summary_and_correlations(self):
        results = statcast_performance_bridge(make_sample_statcast(n_pitches=300))

        self.assertGreater(len(results["pitch_type_summary"]), 0)
        self.assertIn("whiff", results["trait_outcome_correlations"])


if __name__ == "__main__":
    unittest.main()

