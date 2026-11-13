"""Opt-in PostgreSQL integration test; all test DDL/data is rolled back."""

import os
import unittest
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from data.db.client import init_async_db
from data.db.entities_transformed import InferenceMonitoring
from data.db.statements import upsert_table
from stock_market.config import ROOT_DIR


@unittest.skipUnless(os.getenv("TEST_INFERENCE_POSTGRES") == "1", "PostgreSQL test is opt-in")
class MonitoringPostgresTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_upsert_and_status_transitions(self):
        load_dotenv(ROOT_DIR / ".env")
        engine = init_async_db()
        schema = "inference_test_" + uuid4().hex
        ddl = (ROOT_DIR / "data/db/schema_transformed.sql").read_text()
        marker = "CREATE TABLE IF NOT EXISTS stock_market.inference_monitoring"
        ddl = marker + ddl.split(marker, 1)[1]
        ddl = ddl.replace("stock_market.", f"{schema}.")
        statements = [statement.strip() for statement in ddl.split(";") if statement.strip()]
        try:
            async with engine.connect() as conn:
                transaction = await conn.begin()
                try:
                    await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
                    await conn.execute(text(f'CREATE TABLE "{schema}".metadata (stock_id INT PRIMARY KEY)'))
                    for statement in statements:
                        await conn.execute(text(statement))
                    await conn.execute(text(f'INSERT INTO "{schema}".metadata VALUES (1), (2)'))
                    await conn.execution_options(schema_translate_map={"stock_market": schema})
                    common = {
                        "indicator_date": pd.Timestamp("2024-01-02"),
                        "horizon": 10,
                        "model_path": "model.ubj",
                        "scored_at": pd.Timestamp("2024-01-03"),
                    }
                    records = pd.DataFrame([
                        {**common, "stock_id": 1, "status": "predicted",
                         "predicted_class": 2, "predicted_label": "upper_barrier",
                         "probability_lower": 0.1, "probability_neither": 0.2,
                         "probability_upper": 0.7, "invalid_features": [], "close": 100},
                        {**common, "stock_id": 2, "status": "rejected",
                         "predicted_class": None, "predicted_label": None,
                         "probability_lower": None, "probability_neither": None,
                         "probability_upper": None, "invalid_features": ["rsi_14"], "close": 200},
                    ])
                    await upsert_table(conn, InferenceMonitoring, records)
                    initial = (await conn.execute(text(
                        f'SELECT stock_id, status, invalid_features, close FROM "{schema}".inference_monitoring ORDER BY stock_id'
                    ))).all()
                    self.assertEqual([(r.stock_id, r.status, r.invalid_features) for r in initial],
                                     [(1, "predicted", []), (2, "rejected", ["rsi_14"])])
                    changed = records.iloc[[0]].copy()
                    changed["status"] = "rejected"
                    changed["predicted_class"] = None
                    changed["predicted_label"] = None
                    changed["probability_lower"] = None
                    changed["probability_neither"] = None
                    changed["probability_upper"] = None
                    changed["invalid_features"] = pd.Series([["momentum_5d"]], index=changed.index)
                    await upsert_table(conn, InferenceMonitoring, changed)
                    final = (await conn.execute(text(
                        f'SELECT stock_id, status, invalid_features, probability_upper FROM "{schema}".inference_monitoring ORDER BY stock_id'
                    ))).all()
                    self.assertEqual([(r.stock_id, r.status, r.invalid_features) for r in final],
                                     [(1, "rejected", ["momentum_5d"]), (2, "rejected", ["rsi_14"])])
                    self.assertIsNone(final[0].probability_upper)
                finally:
                    await transaction.rollback()
        finally:
            await engine.dispose()
