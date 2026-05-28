import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_biomechanics.metrics import add_biomechanical_metrics
from mlb_biomechanics.modeling import (
    bootstrap_metric_intervals,
    compare_feature_sets,
    evaluate_velocity_model,
    grouped_kfold_splits,
    metric_ablation,
    residual_diagnostics,
    repeated_grouped_kfold_splits,
    select_ridge_alpha,
    statcast_performance_bridge,
)
from mlb_biomechanics.sample_data import make_sample_biomechanics, make_sample_statcast


class ModelingTest(unittest.TestCase):
    def test_velocity_model_beats_baseline_on_sample_data(self):
        df = add_biomechanical_metrics(make_sample_biomechanics(n_sessions=40, pitches_per_session=4))
        results = evaluate_velocity_model(df, include_extended=False)

        self.assertGreater(results["test_rows"], 0)
        self.assertLess(results["ridge"]["rmse"], results["baseline"]["rmse"])
        self.assertIn("permutation_importance", results)

    def test_statcast_bridge_returns_pitch_type_summary_and_correlations(self):
        results = statcast_performance_bridge(make_sample_statcast(n_pitches=300))

        self.assertGreater(len(results["pitch_type_summary"]), 0)
        self.assertIn("whiff", results["trait_outcome_correlations"])
        self.assertIn("sample_summary", results)

    def test_grouped_kfold_does_not_leak_sessions(self):
        df = add_biomechanical_metrics(make_sample_biomechanics(n_sessions=12, pitches_per_session=3))
        splits = grouped_kfold_splits(df, n_splits=4)

        self.assertEqual(len(splits), 4)
        for train, test in splits:
            train_sessions = set(train["session"])
            test_sessions = set(test["session"])
            self.assertTrue(train_sessions.isdisjoint(test_sessions))
            self.assertGreater(len(test_sessions), 0)

    def test_repeated_grouped_kfold_does_not_leak_sessions(self):
        df = add_biomechanical_metrics(make_sample_biomechanics(n_sessions=12, pitches_per_session=3))
        splits = repeated_grouped_kfold_splits(df, n_splits=4, n_repeats=2, seed=3)

        self.assertEqual(len(splits), 8)
        for repeat, fold, train, test in splits:
            self.assertGreater(repeat, 0)
            self.assertGreater(fold, 0)
            self.assertTrue(set(train["session"]).isdisjoint(set(test["session"])))

    def test_ridge_alpha_selection_returns_valid_alpha_and_table(self):
        df = add_biomechanical_metrics(make_sample_biomechanics(n_sessions=16, pitches_per_session=3))
        best_alpha, table = select_ridge_alpha(df, alphas=[0.0, 1.0, 10.0], n_splits=4)

        self.assertIn(best_alpha, [0.0, 1.0, 10.0])
        self.assertIn("selected_alpha", table.columns)
        self.assertGreater(len(table), 0)

    def test_bootstrap_intervals_are_ordered(self):
        df = add_biomechanical_metrics(make_sample_biomechanics(n_sessions=30, pitches_per_session=4))
        results = evaluate_velocity_model(df, include_extended=False)
        intervals = bootstrap_metric_intervals(results["predictions"], n_bootstrap=50)

        self.assertEqual(set(intervals["metric"]), {"rmse", "mae", "r2"})
        self.assertTrue((intervals["ci_lower"] <= intervals["estimate"]).all())
        self.assertTrue((intervals["estimate"] <= intervals["ci_upper"]).all())
        self.assertEqual(set(intervals["bootstrap_unit"]), {"session"})

    def test_residual_diagnostics_assigns_velocity_bands(self):
        df = add_biomechanical_metrics(make_sample_biomechanics(n_sessions=30, pitches_per_session=4))
        results = evaluate_velocity_model(df, include_extended=False)
        diagnostics = residual_diagnostics(results["predictions"])
        band_rows = diagnostics[diagnostics["diagnostic_group"] == "velocity_band"]

        self.assertGreater(len(band_rows), 0)
        self.assertFalse(band_rows["velocity_band"].isna().any())

    def test_feature_set_comparison_and_ablation_return_summaries(self):
        df = make_sample_biomechanics(n_sessions=16, pitches_per_session=3)
        feature_sets = compare_feature_sets(df, n_splits=4, n_repeats=1)
        ablation = metric_ablation(df, n_splits=4, n_repeats=1)

        self.assertIn("feature_set", feature_sets.columns)
        self.assertIn("rmse_delta_vs_best", feature_sets.columns)
        self.assertIn("rmse_delta_vs_all", ablation.columns)
        self.assertGreater(len(ablation), 1)

    def test_statcast_bridge_uses_rate_denominators(self):
        results = statcast_performance_bridge(make_sample_statcast(n_pitches=300))
        summary = results["pitch_type_summary"]
        manifest = results["sample_manifest"]

        self.assertIn("swings", summary.columns)
        self.assertIn("out_of_zone_pitches", summary.columns)
        self.assertIn("batted_balls", summary.columns)
        self.assertIn("rate_denominators", set(manifest["field"]))


if __name__ == "__main__":
    unittest.main()
