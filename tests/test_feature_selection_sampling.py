import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
from joblib import parallel_backend

from stock_market.analytics.training import feature_engineering as fe


def labeled_frame():
    return pd.DataFrame(
        [
            {
                "indicator_date": pd.Timestamp(
                    datetime(year, 1, 1) + timedelta(days=i)
                ),
                "stock_id": i,
                "label": label,
                "feature": year * 1000 + label * 100 + i,
            }
            for year in (2023, 2024)
            for label in (0, 1, 2)
            for i in range(20)
        ]
    )


class FeatureSelectionSamplingTests(unittest.TestCase):
    def test_stratified_sample_is_exact_and_reproducible(self):
        data = labeled_frame()
        sample = fe.sample_feature_selection_data(data, 0.25, cv=3)
        repeated = fe.sample_feature_selection_data(data, 0.25, cv=3)

        self.assertEqual(len(sample), 30)
        pd.testing.assert_frame_equal(sample, repeated)
        groups = sample.groupby([sample["indicator_date"].dt.year, "label"]).size()
        self.assertEqual(groups.tolist(), [5] * 6)
        self.assertTrue(sample.index.isin(data.index).all())

    def test_fraction_rounding_and_full_sample(self):
        data = labeled_frame()
        self.assertEqual(len(fe.sample_feature_selection_data(data, 0.26, 3)), 31)
        pd.testing.assert_frame_equal(
            fe.sample_feature_selection_data(data, 1.0, 3), data
        )

    def test_invalid_fractions_and_too_small_sample(self):
        for fraction in (0, -0.1, 1.1, np.nan, np.inf):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "sample_fraction"):
                    fe.sample_feature_selection_data(labeled_frame(), fraction, 3)
        with self.assertRaisesRegex(ValueError, "at least 3 rows"):
            fe.sample_feature_selection_data(labeled_frame(), 0.01, 3)

    def test_sample_runs_through_shap_rfe(self):
        sample = fe.sample_feature_selection_data(labeled_frame(), 0.25, cv=2)
        features = sample[["feature"]].copy()
        features["second"] = sample["stock_id"]
        model = fe.xgb_model(n_estimator=2, device="cpu").set_params(n_jobs=1)
        with parallel_backend("threading", n_jobs=2):
            rows = fe.shap_rfe(model, features, sample["label"], min_features=1, cv=2)
        self.assertEqual([row[0] for row in rows], [2, 1])


class FeatureSelectionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_select_features_samples_after_preprocessing_and_saves_metadata(self):
        data = labeled_frame()
        engine = MagicMock()
        engine.dispose = AsyncMock()
        engine.connect.return_value.__aenter__ = AsyncMock(return_value=object())
        engine.connect.return_value.__aexit__ = AsyncMock(return_value=False)
        date = datetime(2023, 1, 1)

        with (
            patch.object(fe, "init_async_db", return_value=engine),
            patch.object(fe, "xgb_model", return_value=object()),
            patch.object(fe, "get_indicators_data", new=AsyncMock(return_value=data)),
            patch.object(fe, "process_data_training", return_value=data) as prepare,
            patch.object(fe, "NUMERIC_FEATURE_COLUMNS", ["feature"]),
            patch.object(fe, "shap_rfe", return_value=[(1, 0.5, ["feature"])]) as rfe,
            patch.object(fe, "save_selected_features") as save,
        ):
            result = await fe.select_features(
                start_date=date, end_date=date, device="cpu", sample_fraction=0.25
            )

        self.assertEqual(result, ["feature"])
        self.assertEqual(len(prepare.call_args.args[0]), 120)
        self.assertEqual(len(rfe.call_args.args[1]), 30)
        self.assertEqual(len(rfe.call_args.args[2]), 30)
        params = save.call_args.kwargs["params"]
        self.assertEqual(params["sample_fraction"], 0.25)
        self.assertEqual(params["prepared_rows"], 120)
        self.assertEqual(params["sampled_rows"], 30)
        engine.dispose.assert_awaited_once()

    async def test_training_configuration_is_forwarded(self):
        selection = AsyncMock(return_value=["feature"])
        with (
            patch.object(fe, "select_features", selection),
            patch.object(fe, "load_train_config", return_value={"feature_selection": {}}),
        ):
            await fe.resolve_features(
                config={},
                training={"feature_selection": {"sample_fraction": 0.5, "random_state": 7}},
                train_config_path=Path("train.yaml"),
                use_feature_engineering=True,
                start_date=datetime(2023, 1, 1).date(),
                end_date=datetime(2024, 1, 1).date(),
                stock_ids=[],
                device="cpu",
            )
        self.assertEqual(selection.await_args.kwargs["sample_fraction"], 0.5)
        self.assertEqual(selection.await_args.kwargs["random_state"], 7)


if __name__ == "__main__":
    unittest.main()
