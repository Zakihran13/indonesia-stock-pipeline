import re
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from data.db import entities_transformed as et
from stock_market.config import RANK_FEATURES
from utils import indicators_calc


class IndicatorSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        frames = []
        for stock_id in (1, 2):
            close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, 300)))
            frames.append(
                pd.DataFrame(
                    {
                        "stock_id": stock_id,
                        "trade_date": pd.bdate_range("2024-01-01", periods=300),
                        "ticker": str(stock_id),
                        "open": close * 0.999,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                    }
                )
            )
        cls.prices = pd.concat(frames, ignore_index=True)
        cls.metadata = pd.DataFrame(
            {"stock_id": [1, 2], "sector": ["Tech", "Tech"]}
        )
        cls.columns = list(et.IndicatorsData.__table__.columns.keys())

    def test_sql_model_and_calculation_columns_match(self):
        schema = (
            Path(__file__).resolve().parents[1] / "data/db/schema_transformed.sql"
        ).read_text()
        table = schema.split(
            "CREATE TABLE IF NOT EXISTS stock_market.indicators_data (", 1
        )[1].split("PRIMARY KEY", 1)[0]
        sql_columns = re.findall(
            r"^\s*(\w+)\s+(?:INT|DATE|VARCHAR|NUMERIC|BOOLEAN)\s*,",
            table,
            re.MULTILINE,
        )
        self.assertEqual(sql_columns, self.columns)
        result = indicators_calc.build_indicator_frame(self.prices, self.metadata)
        self.assertEqual(list(result.columns), sql_columns)
        self.assertFalse(result.isna().all().any(), result.columns[result.isna().all()])

    def test_empty_output_has_table_columns(self):
        result = indicators_calc.build_indicator_frame(pd.DataFrame(), pd.DataFrame())
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), self.columns)

    def test_deferred_cross_section_columns_are_explicitly_null(self):
        result = indicators_calc.build_indicator_frame(
            self.prices, self.metadata, calculate_cross_section=False
        )
        deferred = [
            *RANK_FEATURES.values(),
            "sector_rel_momentum_20d",
            "sector_rel_momentum_120d",
        ]
        self.assertEqual(list(result.columns), self.columns)
        self.assertTrue(result[deferred].isna().all().all())
        self.assertFalse(
            result.drop(columns=deferred).isna().all().any(),
            result.drop(columns=deferred).columns[
                result.drop(columns=deferred).isna().all()
            ],
        )

    def test_missing_indicator_calculation_is_not_silently_padded(self):
        incomplete_ranks = {
            feature: column
            for feature, column in RANK_FEATURES.items()
            if column != "rank_momentum_20d"
        }
        with patch.object(indicators_calc, "RANK_FEATURES", incomplete_ranks):
            with self.assertRaisesRegex(KeyError, "rank_momentum_20d"):
                indicators_calc.build_indicator_frame(self.prices, self.metadata)


if __name__ == "__main__":
    unittest.main()
