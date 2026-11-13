import copy
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd

from stock_market.analytics.inference import inference
from utils.helper import DEFAULT_INFERENCE_CONFIG


class InferenceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = inference.load_model_config(DEFAULT_INFERENCE_CONFIG)
        cls.model = inference.load_model(cls.config)

    def frame(self):
        frame = pd.DataFrame(
            np.zeros((2, len(self.config["features"]))),
            columns=self.config["features"],
        )
        frame.insert(0, "indicator_date", [date(2024, 1, 2)] * 2)
        frame.insert(0, "ticker", ["AAA", "BBB"])
        frame.insert(0, "stock_id", [1, 2])
        return frame

    def test_real_model_uses_ordered_features_and_returns_probabilities(self):
        frame = self.frame()
        predictions, rejected = inference.predict_snapshot(
            self.model, self.config, frame[frame.columns[::-1]]
        )
        expected = self.model.predict_proba(frame[self.config["features"]])
        self.assertEqual(rejected, [])
        self.assertEqual(len(predictions), 2)
        for prediction, probabilities in zip(predictions, expected):
            self.assertEqual(prediction["predicted_class"], int(np.argmax(probabilities)))
            self.assertEqual(
                prediction["predicted_label"],
                self.config["label"]["classes"][prediction["predicted_class"]],
            )
            np.testing.assert_allclose(
                list(prediction["probabilities"].values()), probabilities
            )
            self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1)

    def test_bad_rows_are_reported_without_rejecting_valid_rows(self):
        frame = self.frame()
        feature = self.config["features"][0]
        frame.loc[1, feature] = np.inf
        predictions, rejected = inference.predict_snapshot(self.model, self.config, frame)
        self.assertEqual([row["stock_id"] for row in predictions], [1])
        self.assertEqual(rejected, [{"stock_id": 2, "invalid_features": [feature]}])
        frame.loc[0, feature] = np.nan
        with self.assertRaisesRegex(ValueError, "No rows have valid"):
            inference.predict_snapshot(self.model, self.config, frame)

    def test_missing_and_duplicate_rows_fail(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, "missing columns"):
            inference.predict_snapshot(self.model, self.config, frame.drop(columns=self.config["features"][0]))
        frame.loc[1, "stock_id"] = 1
        with self.assertRaisesRegex(ValueError, "one non-null row per stock"):
            inference.predict_snapshot(self.model, self.config, frame)

    def test_model_config_mismatch_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["features"].reverse()
        with self.assertRaisesRegex(ValueError, "feature names/order"):
            inference.load_model(config)
        config = copy.deepcopy(self.config)
        config["model"]["best_iteration"] += 1
        with self.assertRaisesRegex(ValueError, "best_iteration"):
            inference.load_model(config)

    def test_numeric_features_outside_default_feature_list(self):
        config = {"features": ["ma_alignment_score"], "label": self.config["label"]}
        frame = pd.DataFrame(
            {"stock_id": [1], "indicator_date": [date(2024, 1, 2)], "ma_alignment_score": ["3"]}
        )
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        predictions, rejected = inference.predict_snapshot(model, config, frame)
        self.assertEqual(predictions[0]["predicted_class"], 2)
        self.assertEqual(rejected, [])
        self.assertEqual(model.predict_proba.call_args.args[0].iloc[0, 0], 3)

    def test_cli_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.json"
            with (
                patch("sys.argv", ["inference", "--output", str(output)]),
                patch.object(inference, "run_inference", new=AsyncMock(return_value={"predictions": []})),
            ):
                inference.main()
            self.assertEqual(json.loads(output.read_text()), {"predictions": []})

    def test_cli_forwards_range(self):
        with (
            patch("sys.argv", [
                "inference", "--start-date", "2024-01-01",
                "--end-date", "2024-01-05", "--stock-ids", "1",
            ]),
            patch.object(inference, "run_inference", new=AsyncMock(return_value={})) as run,
            patch("builtins.print"),
        ):
            inference.main()
        self.assertEqual(run.await_args.args[2], [1])
        self.assertEqual(run.await_args.kwargs, {
            "start_date": "2024-01-01", "end_date": "2024-01-05"
        })


class InferenceFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_range_scores_each_available_date_and_returns_summary(self):
        config = inference.load_model_config(DEFAULT_INFERENCE_CONFIG)
        first, second = date(2024, 1, 5), date(2024, 1, 8)
        def frame(day, value):
            return pd.DataFrame([{
                **dict.fromkeys(config["features"], value),
                "stock_id": 8, "indicator_date": day,
            }])
        frames = [frame(first, np.nan), frame(second, 0.0)]
        with (
            patch.object(inference, "fetch_inference_dates", new=AsyncMock(
                return_value=[first, second]
            )) as dates,
            patch.object(inference, "fetch_inference_data", new=AsyncMock(
                side_effect=[(first, frames[0]), (second, frames[1])]
            )) as fetch,
            patch.object(inference, "persist_monitoring", new=AsyncMock()) as persist,
            patch.object(inference, "load_model", wraps=inference.load_model) as model,
        ):
            result = await inference.run_inference(
                start_date="2024-01-05", end_date="2024-01-08", stock_ids=[8]
            )
        dates.assert_awaited_once_with(first, second)
        self.assertEqual(fetch.await_count, 2)
        self.assertEqual(fetch.await_args_list[0].args, (config["features"], first, [8]))
        self.assertEqual(fetch.await_args_list[1].args, (config["features"], second, [8]))
        self.assertEqual(persist.await_count, 2)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(result["batches"], [
            {"indicator_date": "2024-01-05", "predicted_rows": 0, "rejected_rows": 1},
            {"indicator_date": "2024-01-08", "predicted_rows": 1, "rejected_rows": 0},
        ])
        self.assertEqual(result["predicted_rows"], 1)
        self.assertEqual(result["rejected_rows"], 1)
        self.assertEqual(result["training_metrics"], config["metrics"])
        json.dumps(result, allow_nan=False)

    async def test_range_validation_happens_before_loading_model(self):
        for start, end, as_of in (
            ("2024-01-01", None, None),
            (None, "2024-01-01", None),
            ("", "2024-01-01", None),
            ("2024-01-02", "2024-01-01", None),
            ("2024-01-01", "2024-01-02", "2024-01-01"),
        ):
            with self.subTest(start=start, end=end, as_of=as_of), patch.object(
                inference, "load_model_config"
            ) as load:
                with self.assertRaises(ValueError):
                    await inference.run_inference(
                        as_of=as_of, start_date=start, end_date=end
                    )
                load.assert_not_called()

    async def test_range_accepts_same_start_and_end(self):
        config = inference.load_model_config(DEFAULT_INFERENCE_CONFIG)
        day = date(2024, 1, 2)
        frame = pd.DataFrame([{
            **dict.fromkeys(config["features"], 0.0), "stock_id": 1, "indicator_date": day,
        }])
        with (
            patch.object(inference, "fetch_inference_dates", new=AsyncMock(return_value=[day])) as dates,
            patch.object(inference, "fetch_inference_data", new=AsyncMock(return_value=(day, frame))),
            patch.object(inference, "persist_monitoring", new=AsyncMock()),
        ):
            result = await inference.run_inference(start_date=day, end_date=day)
        dates.assert_awaited_once_with(day, day)
        self.assertEqual(result["batches"][0]["indicator_date"], "2024-01-02")

    async def test_all_rejected_range_persists_every_date_before_failure(self):
        config = inference.load_model_config(DEFAULT_INFERENCE_CONFIG)
        dates = [date(2024, 1, 2), date(2024, 1, 3)]
        frames = [pd.DataFrame([{
            **dict.fromkeys(config["features"], np.nan), "stock_id": 1, "indicator_date": day
        }]) for day in dates]
        with (
            patch.object(inference, "fetch_inference_dates", new=AsyncMock(return_value=dates)),
            patch.object(inference, "fetch_inference_data", new=AsyncMock(
                side_effect=list(zip(dates, frames))
            )),
            patch.object(inference, "persist_monitoring", new=AsyncMock()) as persist,
        ):
            with self.assertRaisesRegex(ValueError, "rejected rows were persisted"):
                await inference.run_inference(start_date=dates[0], end_date=dates[-1])
        self.assertEqual(persist.await_count, 2)

    async def test_range_stops_when_a_batch_write_fails(self):
        config = inference.load_model_config(DEFAULT_INFERENCE_CONFIG)
        dates = [date(2024, 1, day) for day in (2, 3, 4)]
        frames = [pd.DataFrame([{
            **dict.fromkeys(config["features"], 0.0), "stock_id": 1, "indicator_date": day
        }]) for day in dates]
        with (
            patch.object(inference, "fetch_inference_dates", new=AsyncMock(return_value=dates)),
            patch.object(inference, "fetch_inference_data", new=AsyncMock(
                side_effect=list(zip(dates, frames))
            )) as fetch,
            patch.object(inference, "persist_monitoring", new=AsyncMock(
                side_effect=[None, RuntimeError("write failed")]
            )) as persist,
        ):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                await inference.run_inference(start_date=dates[0], end_date=dates[-1])
        self.assertEqual(fetch.await_count, 2)
        self.assertEqual(persist.await_count, 2)

    async def test_fetch_range_dates_uses_inclusive_bounds_and_disposes(self):
        result = MagicMock()
        result.scalars.return_value.all.return_value = [date(2024, 1, 5), date(2024, 1, 8)]
        connection = MagicMock()
        connection.execute = AsyncMock(return_value=result)
        engine = MagicMock()
        engine.connect.return_value.__aenter__ = AsyncMock(return_value=connection)
        engine.connect.return_value.__aexit__ = AsyncMock(return_value=False)
        engine.dispose = AsyncMock()
        with patch.object(inference, "init_async_db", return_value=engine):
            dates = await inference.fetch_inference_dates(date(2024, 1, 5), date(2024, 1, 8))
        self.assertEqual(dates, [date(2024, 1, 5), date(2024, 1, 8)])
        query = connection.execute.await_args.args[0]
        self.assertIn("BETWEEN", str(query))
        self.assertIn("DISTINCT", str(query))
        engine.dispose.assert_awaited_once()

    async def test_empty_range_is_explicit(self):
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        connection = MagicMock()
        connection.execute = AsyncMock(return_value=result)
        engine = MagicMock()
        engine.connect.return_value.__aenter__ = AsyncMock(return_value=connection)
        engine.connect.return_value.__aexit__ = AsyncMock(return_value=False)
        engine.dispose = AsyncMock()
        with patch.object(inference, "init_async_db", return_value=engine):
            with self.assertRaisesRegex(ValueError, "No indicator snapshots"):
                await inference.fetch_inference_dates(date(2024, 1, 6), date(2024, 1, 7))
        engine.dispose.assert_awaited_once()

    async def test_flow_uses_metadata_snapshot_and_returns_historical_metrics(self):
        config = inference.load_model_config(DEFAULT_INFERENCE_CONFIG)
        frame = pd.DataFrame(
            np.zeros((1, len(config["features"]))), columns=config["features"]
        )
        frame["stock_id"] = 8
        frame["ticker"] = "AAA"
        frame["indicator_date"] = date(2024, 1, 2)
        with (
            patch.object(
                inference, "fetch_inference_data",
                new=AsyncMock(return_value=(date(2024, 1, 2), frame)),
            ) as fetch,
            patch.object(inference, "persist_monitoring", new=AsyncMock()) as persist,
        ):
            result = await inference.run_inference(as_of="2024-01-02", stock_ids=[8])
        fetch.assert_awaited_once_with(config["features"], date(2024, 1, 2), [8])
        self.assertEqual(persist.await_count, 1)
        self.assertEqual(result["training_metrics"], config["metrics"])
        self.assertEqual(result["model"], config["model"])
        self.assertEqual(result["indicator_date"], "2024-01-02")
        self.assertEqual(result["predictions"][0]["stock_id"], 8)
        json.dumps(result, allow_nan=False)

    async def test_fetch_latest_global_date_and_dispose(self):
        date_result = MagicMock()
        date_result.scalar_one.return_value = date(2024, 1, 2)
        rows_result = MagicMock()
        rows_result.mappings.return_value.all.return_value = [
            {"stock_id": 5, "ticker": "AAA", "indicator_date": date(2024, 1, 2), "momentum_5d": 1}
        ]
        connection = MagicMock()
        connection.execute = AsyncMock(side_effect=[date_result, rows_result])
        engine = MagicMock()
        engine.connect.return_value.__aenter__ = AsyncMock(return_value=connection)
        engine.connect.return_value.__aexit__ = AsyncMock(return_value=False)
        engine.dispose = AsyncMock()
        with patch.object(inference, "init_async_db", return_value=engine):
            snapshot, frame = await inference.fetch_inference_data(["momentum_5d"], stock_ids=[5])
        self.assertEqual(snapshot, date(2024, 1, 2))
        self.assertEqual(frame["stock_id"].tolist(), [5])
        first_sql = str(connection.execute.await_args_list[0].args[0])
        second_sql = str(connection.execute.await_args_list[1].args[0])
        self.assertIn("max(", first_sql)
        self.assertNotIn("stock_id IN", first_sql)
        self.assertIn("stock_id IN", second_sql)
        engine.dispose.assert_awaited_once()

    async def test_missing_requested_stock_is_not_silently_dropped(self):
        result = MagicMock()
        result.mappings.return_value.all.return_value = [
            {"stock_id": 1, "ticker": "AAA", "indicator_date": date(2024, 1, 2), "momentum_5d": 1}
        ]
        connection = MagicMock()
        connection.execute = AsyncMock(return_value=result)
        engine = MagicMock()
        engine.connect.return_value.__aenter__ = AsyncMock(return_value=connection)
        engine.connect.return_value.__aexit__ = AsyncMock(return_value=False)
        engine.dispose = AsyncMock()
        with patch.object(inference, "init_async_db", return_value=engine):
            with self.assertRaisesRegex(ValueError, r"Stocks missing.*\[2\]"):
                await inference.fetch_inference_data(["momentum_5d"], date(2024, 1, 2), [1, 2])
        engine.dispose.assert_awaited_once()

    async def test_empty_snapshot_and_database_errors_dispose_connection(self):
        result = MagicMock()
        result.scalar_one.return_value = None
        connection = MagicMock()
        connection.execute = AsyncMock(return_value=result)
        engine = MagicMock()
        engine.connect.return_value.__aenter__ = AsyncMock(return_value=connection)
        engine.connect.return_value.__aexit__ = AsyncMock(return_value=False)
        engine.dispose = AsyncMock()
        with patch.object(inference, "init_async_db", return_value=engine):
            with self.assertRaisesRegex(ValueError, "No indicator snapshot"):
                await inference.fetch_inference_data(["momentum_5d"])
        engine.dispose.assert_awaited_once()

    async def test_upsert_monitoring_contains_predictions_rejections_and_market_context(self):
        config = inference.load_model_config(DEFAULT_INFERENCE_CONFIG)
        frame = pd.DataFrame([
            {
                "stock_id": 1, "ticker": "AAA", "indicator_date": date(2024, 1, 2),
                "market_sector": "Energy", "close": 100, "returns": 0.02,
                "volume": 500, "momentum_5d": 0.1, "volatility_20d_ann": 0.3,
            },
            {
                "stock_id": 2, "ticker": "BBB", "indicator_date": date(2024, 1, 2),
                "market_sector": "Technology", "close": 200, "returns": -0.01,
            },
        ])
        predictions = [{
            "stock_id": 1, "predicted_class": 2, "predicted_label": "upper_barrier",
            "probabilities": {"lower_barrier": 0.1, "neither_barrier": 0.2, "upper_barrier": 0.7},
        }]
        rejected = [{"stock_id": 2, "invalid_features": ["rsi_14"]}]
        engine = MagicMock()
        connection = object()
        engine.begin.return_value.__aenter__ = AsyncMock(return_value=connection)
        engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)
        engine.dispose = AsyncMock()
        with (
            patch.object(inference, "init_async_db", return_value=engine),
            patch.object(inference, "upsert_table", new=AsyncMock()) as upsert,
        ):
            await inference.persist_monitoring(frame, config, predictions, rejected)
        self.assertIs(upsert.await_args.args[0], connection)
        self.assertIs(upsert.await_args.args[1], inference.et.InferenceMonitoring)
        records = upsert.await_args.args[2].to_dict("records")
        self.assertEqual(records[0]["status"], "predicted")
        self.assertEqual(records[0]["close"], 100)
        self.assertEqual(records[0]["daily_return"], 0.02)
        self.assertEqual(records[0]["sector"], "Energy")
        self.assertEqual(records[0]["probability_upper"], 0.7)
        self.assertEqual(records[0]["invalid_features"], [])
        self.assertEqual(records[1]["status"], "rejected")
        self.assertEqual(records[1]["invalid_features"], ["rsi_14"])
        self.assertTrue(pd.isna(records[1]["predicted_class"]))
        self.assertTrue(pd.isna(records[1]["probability_upper"]))
        engine.dispose.assert_awaited_once()

    async def test_all_rejections_are_persisted_before_failure(self):
        config = inference.load_model_config(DEFAULT_INFERENCE_CONFIG)
        frame = pd.DataFrame([{**dict.fromkeys(config["features"], np.nan),
                               "stock_id": 1, "indicator_date": date(2024, 1, 2)}])
        with (
            patch.object(inference, "fetch_inference_data", new=AsyncMock(
                return_value=(date(2024, 1, 2), frame)
            )),
            patch.object(inference, "persist_monitoring", new=AsyncMock()) as persist,
        ):
            with self.assertRaisesRegex(ValueError, "rejected rows were persisted"):
                await inference.run_inference()
        self.assertEqual(persist.await_args.args[2], [])
        self.assertEqual(persist.await_args.args[3][0]["stock_id"], 1)

    async def test_persistence_errors_are_not_hidden(self):
        config = inference.load_model_config(DEFAULT_INFERENCE_CONFIG)
        frame = pd.DataFrame([{"stock_id": 1, "indicator_date": date(2024, 1, 2)}])
        engine = MagicMock()
        engine.begin.return_value.__aenter__ = AsyncMock(return_value=object())
        engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)
        engine.dispose = AsyncMock()
        with (
            patch.object(inference, "init_async_db", return_value=engine),
            patch.object(inference, "upsert_table", new=AsyncMock(side_effect=RuntimeError("db failure"))),
        ):
            with self.assertRaisesRegex(RuntimeError, "db failure"):
                await inference.persist_monitoring(
                    frame, config, [], [{"stock_id": 1, "invalid_features": ["rsi_14"]}]
                )
        engine.dispose.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
