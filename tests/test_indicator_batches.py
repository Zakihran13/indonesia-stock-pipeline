import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd
from sqlalchemy import create_engine, select, text, update

from data.db import entities_transformed as et
from stock_market.transform import indicators
from utils.indicators_calc import build_indicator_frame


class SQLiteAsyncConnection:
    def __init__(self, connection):
        self.connection = connection

    async def execute(self, statement, *args):
        if isinstance(statement, type(text(""))):
            statement = text(str(statement).replace(" ON COMMIT DROP", ""))
        return self.connection.execute(statement, *args)


class SQLiteAsyncEngine:
    def __init__(self, engine):
        self.engine = engine
        self.disposed = False

    def begin(self):
        return self

    async def __aenter__(self):
        self.transaction = self.engine.begin()
        return SQLiteAsyncConnection(self.transaction.__enter__())

    async def __aexit__(self, *args):
        return self.transaction.__exit__(*args)

    async def dispose(self):
        self.disposed = True


class IndicatorBatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        with self.engine.begin() as conn:
            conn.exec_driver_sql("ATTACH DATABASE ':memory:' AS stock_market")
            et.Base.metadata.create_all(
                conn,
                tables=[
                    et.StockMetadata.__table__,
                    et.PriceData.__table__,
                    et.IndicatorsData.__table__,
                ],
            )

        dates = pd.bdate_range("2024-01-01", periods=275)
        metadata = [
            {"stock_id": 1, "ticker": "A", "sector": "Tech"},
            {"stock_id": 2, "ticker": "B", "sector": "Tech"},
            {"stock_id": 3, "ticker": "C", "sector": "Tech"},
            {"stock_id": 4, "ticker": "D", "sector": None},
        ]
        price_rows = []
        for stock_id in range(1, 5):
            for index, date in enumerate(dates):
                if stock_id == 4 and index % 17 == 0:
                    continue
                close = 100 + index * (0.1 if stock_id in (1, 2) else 0.05)
                price_rows.append(
                    {
                        "stock_id": stock_id,
                        "ticker": chr(64 + stock_id),
                        "trade_date": date.date(),
                        "open": close,
                        "high": close + 1,
                        "low": close - 1,
                        "close": close,
                        "volume": 1000,
                    }
                )
        self.prices = pd.DataFrame(price_rows)
        for row in metadata:
            row["created_at"] = datetime(2023, 1, 1)
        self.metadata = pd.DataFrame(metadata)
        with self.engine.begin() as conn:
            conn.execute(et.StockMetadata.__table__.insert(), metadata)
            conn.execute(et.PriceData.__table__.insert(), price_rows)
        self.start_date = dates[-3].to_pydatetime()

    def tearDown(self):
        self.engine.dispose()

    async def test_sector_join_ignores_metadata_dates(self):
        future_date = datetime(2030, 1, 1)
        metadata = self.metadata.copy()
        metadata.loc[metadata["stock_id"] == 1, "created_at"] = future_date
        metadata.loc[metadata["stock_id"] == 2, "created_at"] = pd.NaT
        with self.engine.begin() as conn:
            conn.execute(
                update(et.StockMetadata)
                .where(et.StockMetadata.stock_id == 1)
                .values(created_at=future_date)
            )
            conn.execute(
                update(et.StockMetadata)
                .where(et.StockMetadata.stock_id == 2)
                .values(created_at=None)
            )

        baseline = build_indicator_frame(
            self.prices, metadata, start_date=self.start_date
        ).set_index(["stock_id", "indicator_date"])
        last_date = self.prices["trade_date"].max()
        self.assertEqual(baseline.loc[(1, last_date), "sector"], "Tech")
        self.assertEqual(baseline.loc[(2, last_date), "sector"], "Tech")
        self.assertTrue(pd.isna(baseline.loc[(4, last_date), "sector"]))
        expected = (
            baseline.loc[(1, last_date), "momentum_20d"]
            - baseline.loc[
                [(1, last_date), (2, last_date), (3, last_date)], "momentum_20d"
            ].mean()
        )
        self.assertAlmostEqual(
            baseline.loc[(1, last_date), "sector_rel_momentum_20d"], expected
        )

        db = SQLiteAsyncEngine(self.engine)
        with patch.object(indicators, "init_async_db", return_value=db):
            await indicators.materialize_indicators(
                start_date=self.start_date, batch_size=2
            )
        with self.engine.connect() as conn:
            rows = conn.execute(select(et.IndicatorsData.__table__)).mappings().all()
        actual = pd.DataFrame(rows).set_index(["stock_id", "indicator_date"])
        for stock_id in (1, 2, 3):
            self.assertEqual(actual.loc[(stock_id, last_date), "sector"], "Tech")
            self.assertAlmostEqual(
                float(actual.loc[(stock_id, last_date), "sector_rel_momentum_20d"]),
                float(baseline.loc[(stock_id, last_date), "sector_rel_momentum_20d"]),
                places=6,
            )

    def test_empty_metadata_preserves_price_rows(self):
        result = build_indicator_frame(
            self.prices, pd.DataFrame(), start_date=self.start_date
        )
        expected = self.prices[self.prices["trade_date"] >= self.start_date.date()]
        self.assertEqual(len(result), len(expected))
        self.assertTrue(result["sector"].isna().all())

    def test_missing_stock_metadata_preserves_price_rows(self):
        metadata = self.metadata[self.metadata["stock_id"] != 3]
        result = build_indicator_frame(
            self.prices, metadata, start_date=self.start_date
        )
        self.assertEqual(set(result["stock_id"]), {1, 2, 3, 4})
        self.assertTrue(result.loc[result["stock_id"] == 3, "sector"].isna().all())

    def test_duplicate_metadata_stock_ids_are_rejected(self):
        metadata = pd.concat(
            [self.metadata, self.metadata.iloc[[0]]], ignore_index=True
        )
        with self.assertRaises(pd.errors.MergeError):
            build_indicator_frame(self.prices, metadata, start_date=self.start_date)

    async def test_batches_match_universe_calculation_and_target_only(self):
        universe = build_indicator_frame(
            self.prices, self.metadata, start_date=self.start_date
        )
        baseline = universe[universe["stock_id"].isin([1, 3])].set_index(
            ["stock_id", "indicator_date"]
        )
        with self.engine.begin() as conn:
            await indicators.insert_indicators_data(SQLiteAsyncConnection(conn), universe)
            peers_before = conn.execute(
                select(et.IndicatorsData.__table__).where(
                    et.IndicatorsData.stock_id.in_([2, 4])
                )
            ).mappings().all()
        db = SQLiteAsyncEngine(self.engine)
        real_fetch = indicators.fetch_frame
        fetched_batches = []

        async def record_fetch(conn, model, date_column, stock_ids, start_date):
            if model is et.PriceData:
                fetched_batches.append(list(stock_ids))
            return await real_fetch(conn, model, date_column, stock_ids, start_date)

        with (
            patch.object(indicators, "init_async_db", return_value=db),
            patch.object(indicators, "fetch_frame", side_effect=record_fetch),
        ):
            count = await indicators.materialize_indicators(
                stock_ids=[1, 3], start_date=self.start_date, batch_size=1
            )
        self.assertEqual(fetched_batches, [[1], [3]])
        self.assertTrue(db.disposed)
        self.assertEqual(count, len(baseline))

        with self.engine.connect() as conn:
            rows = conn.execute(
                select(et.IndicatorsData.__table__).where(
                    et.IndicatorsData.stock_id.in_([1, 3])
                )
            ).mappings().all()
            peers_after = conn.execute(
                select(et.IndicatorsData.__table__).where(
                    et.IndicatorsData.stock_id.in_([2, 4])
                )
            ).mappings().all()
        self.assertEqual(peers_after, peers_before)
        actual = pd.DataFrame(rows).set_index(["stock_id", "indicator_date"])
        self.assertEqual(set(actual.index), set(baseline.index))
        columns = baseline.columns
        for column in columns:
            for key, expected in baseline[column].items():
                value = actual.loc[key, column]
                if pd.isna(expected):
                    self.assertTrue(pd.isna(value), (key, column, value))
                elif isinstance(expected, str):
                    self.assertEqual(value, expected, (key, column))
                else:
                    self.assertAlmostEqual(
                        float(value), float(expected), places=6, msg=f"{key} {column}"
                    )

    async def test_targeted_run_without_materialized_peers(self):
        db = SQLiteAsyncEngine(self.engine)
        with patch.object(indicators, "init_async_db", return_value=db):
            await indicators.materialize_indicators(
                stock_ids=[1], start_date=self.start_date, batch_size=1
            )
        with self.engine.connect() as conn:
            rows = conn.execute(select(et.IndicatorsData.__table__)).mappings().all()
        self.assertTrue(rows)
        self.assertEqual({row["stock_id"] for row in rows}, {1})
        for row in rows:
            self.assertEqual(float(row["rank_momentum_20d"]), 1.0)
            self.assertEqual(float(row["sector_rel_momentum_20d"]), 0.0)

    async def test_targeted_run_uses_stored_peers_not_stale_targets(self):
        baseline = build_indicator_frame(
            self.prices, self.metadata, start_date=self.start_date
        )
        stored = baseline[baseline["stock_id"].isin([1, 2])]
        last_date = self.prices["trade_date"].max()
        with self.engine.begin() as conn:
            await indicators.insert_indicators_data(SQLiteAsyncConnection(conn), stored)
            for stock_id, momentum in ((1, 999), (2, 5)):
                conn.execute(
                    update(et.IndicatorsData)
                    .where(et.IndicatorsData.stock_id == stock_id)
                    .values(momentum_20d=momentum)
                )
        db = SQLiteAsyncEngine(self.engine)
        with patch.object(indicators, "init_async_db", return_value=db):
            await indicators.materialize_indicators(
                stock_ids=[1], start_date=self.start_date
            )
        with self.engine.connect() as conn:
            target = conn.execute(
                select(et.IndicatorsData.__table__).where(
                    et.IndicatorsData.stock_id == 1,
                    et.IndicatorsData.indicator_date == last_date,
                )
            ).mappings().one()
            peer = conn.execute(
                select(et.IndicatorsData.momentum_20d).where(
                    et.IndicatorsData.stock_id == 2,
                    et.IndicatorsData.indicator_date == last_date,
                )
            ).scalar_one()
        momentum = float(target["momentum_20d"])
        self.assertAlmostEqual(float(peer), 5.0)
        self.assertAlmostEqual(float(target["rank_momentum_20d"]), 0.5)
        self.assertAlmostEqual(
            float(target["sector_rel_momentum_20d"]), (momentum - 5) / 2, places=6
        )

    async def test_failure_rolls_back_all_batches(self):
        db = SQLiteAsyncEngine(self.engine)
        with (
            patch.object(indicators, "init_async_db", return_value=db),
            patch.object(
                indicators, "_update_cross_section", side_effect=RuntimeError("failed")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                await indicators.materialize_indicators(
                    start_date=self.start_date, batch_size=2
                )
        self.assertTrue(db.disposed)
        with self.engine.connect() as conn:
            self.assertEqual(conn.execute(select(et.IndicatorsData)).all(), [])

    async def test_no_prices_returns_none(self):
        db = SQLiteAsyncEngine(self.engine)
        with patch.object(indicators, "init_async_db", return_value=db):
            count = await indicators.materialize_indicators(
                start_date=datetime(2100, 1, 1), batch_size=2
            )
        self.assertIsNone(count)
        self.assertTrue(db.disposed)

    async def test_invalid_batch_size_does_not_open_database(self):
        with patch.object(indicators, "init_async_db") as init:
            with self.assertRaises(ValueError):
                await indicators.materialize_indicators(batch_size=0)
            init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
