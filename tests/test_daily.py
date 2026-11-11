"""Модульные тесты для суточной агрегации (src/aggregation/daily.py)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

import chatmood_metrics as cm
from src.aggregation.daily import (
    REQUIRED_DAILY_COLUMNS,
    compute_daily_aggregates,
    export_chat_daily_csv,
)
from src.data.loader import load_dev


class TestDailyAggregation(unittest.TestCase):
    """Тестирование расчета суточных агрегатов и совпадения с эталоном."""

    def test_exact_match_with_chatmood_metrics(self):
        """Проверка 100% совпадения с эталонным расчётом chatmood_metrics.py на dev."""
        dev_df = load_dev("data")
        msgs = dev_df[["message_id", "chat_id", "timestamp"]]
        lbls = dev_df[["message_id", "is_service", "sentiment", "aggression"]]

        # 1. Расчет нашей реализацией
        our_daily = compute_daily_aggregates(msgs, lbls)

        # 2. Расчет эталоном жюри
        gold_daily = cm.daily_aggregates(msgs, lbls)

        # Сравнение колонок
        for col in ["n_messages", "n_resident", "n_neg", "n_pos", "n_agg"]:
            np.testing.assert_array_equal(
                our_daily[col].to_numpy(),
                gold_daily[col].to_numpy(),
                err_msg=f"Несовпадение в колонке {col}",
            )

        for col in ["mood_index", "negative_share", "aggression_share"]:
            np.testing.assert_allclose(
                our_daily[col].to_numpy(),
                gold_daily[col].to_numpy(),
                rtol=1e-5,
                atol=1e-5,
                equal_nan=True,
                err_msg=f"Несовпадение в расчетной метрике {col}",
            )

    def test_service_only_day_nan_metrics(self):
        """Проверка дня, когда были только служебные сообщения (n_resident == 0)."""
        df = pd.DataFrame({
            "message_id": ["m1", "m2"],
            "chat_id": ["c1", "c1"],
            "timestamp": ["2026-06-15 10:00:00", "2026-06-15 11:00:00"],
            "is_service": [1, 1],
            "sentiment": ["neutral", "neutral"],
            "aggression": [0, 0],
        })
        daily = compute_daily_aggregates(df)
        self.assertEqual(len(daily), 1)
        self.assertEqual(daily.iloc[0]["n_messages"], 2)
        self.assertEqual(daily.iloc[0]["n_resident"], 0)
        self.assertTrue(np.isnan(daily.iloc[0]["mood_index"]))
        self.assertTrue(np.isnan(daily.iloc[0]["negative_share"]))
        self.assertTrue(np.isnan(daily.iloc[0]["aggression_share"]))

    def test_export_chat_daily_csv(self):
        """Проверка экспорта в CSV с разделителем ';' и корректными заголовками."""
        df = pd.DataFrame({
            "message_id": ["m1", "m2"],
            "chat_id": ["c1", "c1"],
            "timestamp": ["2026-06-15 10:00:00", "2026-06-15 11:00:00"],
            "is_service": [0, 0],
            "sentiment": ["positive", "negative"],
            "aggression": [0, 0],
        })
        daily = compute_daily_aggregates(df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "chat_daily.csv"
            export_chat_daily_csv(daily, out_file)
            self.assertTrue(out_file.is_file())

            # Читаем и проверяем схему
            read_df = pd.read_csv(out_file, sep=";")
            self.assertListEqual(list(read_df.columns), REQUIRED_DAILY_COLUMNS)
            self.assertEqual(len(read_df), 1)
            self.assertEqual(read_df.iloc[0]["n_resident"], 2)
            self.assertEqual(read_df.iloc[0]["mood_index"], 0.0)


if __name__ == "__main__":
    unittest.main()
