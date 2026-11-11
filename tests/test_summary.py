"""Модульные тесты для сводной агрегации по чатам (src/aggregation/summary.py)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

import chatmood_metrics as cm
from src.aggregation.daily import compute_daily_aggregates, export_chat_daily_csv
from src.aggregation.summary import (
    REQUIRED_SUMMARY_COLUMNS,
    compute_chat_summary,
    export_chat_summary_csv,
)
from src.data.loader import load_dev


class TestSummaryAggregation(unittest.TestCase):
    """Тестирование расчета сводных показателей по чатам."""

    def test_exact_match_with_chatmood_metrics(self):
        """Проверка 100% совпадения с эталонным расчетом cm.chat_summary на dev."""
        dev_df = load_dev("data")
        msgs = dev_df[["message_id", "chat_id", "timestamp"]]
        lbls = dev_df[["message_id", "is_service", "sentiment", "aggression"]]

        daily = compute_daily_aggregates(msgs, lbls)
        our_summ = compute_chat_summary(daily)
        gold_summ = cm.chat_summary(daily)

        # Проверка порядка колонок
        self.assertListEqual(list(our_summ.columns), REQUIRED_SUMMARY_COLUMNS)

        # Проверка совпадения числовых и категориальных колонок
        for col in ["n_resident", "trend", "peak_days"]:
            np.testing.assert_array_equal(
                our_summ[col].to_numpy(),
                gold_summ[col].to_numpy(),
                err_msg=f"Несовпадение в колонке {col}",
            )

        for col in ["mood_index", "negative_share", "aggression_share"]:
            np.testing.assert_allclose(
                our_summ[col].to_numpy(),
                gold_summ[col].to_numpy(),
                rtol=1e-5,
                atol=1e-5,
                equal_nan=True,
                err_msg=f"Несовпадение в расчетной метрике {col}",
            )

    def test_reconstruction_from_exported_daily(self):
        """Проверка восстановления счетчиков из экспортированного chat_daily.csv."""
        dev_df = load_dev("data")
        msgs = dev_df[["message_id", "chat_id", "timestamp"]]
        lbls = dev_df[["message_id", "is_service", "sentiment", "aggression"]]

        daily = compute_daily_aggregates(msgs, lbls)

        with tempfile.TemporaryDirectory() as tmp_dir:
            daily_file = Path(tmp_dir) / "chat_daily.csv"
            export_chat_daily_csv(daily, daily_file)

            # Читаем chat_daily.csv (в нем нет колонок n_neg, n_pos, n_agg)
            read_daily = pd.read_csv(daily_file, sep=";")
            summ_from_file = compute_chat_summary(read_daily)
            summ_direct = compute_chat_summary(daily)

            pd.testing.assert_frame_equal(summ_from_file, summ_direct)

    def test_empty_dataframe(self):
        """Проверка устойчивости к пустому входному DataFrame."""
        empty_daily = pd.DataFrame()
        summ = compute_chat_summary(empty_daily)
        self.assertTrue(summ.empty)
        self.assertListEqual(list(summ.columns), REQUIRED_SUMMARY_COLUMNS)

    def test_export_chat_summary_csv(self):
        """Проверка экспорта в CSV с разделителем ';' и кодировкой UTF-8-BOM."""
        dev_df = load_dev("data")
        msgs = dev_df[["message_id", "chat_id", "timestamp"]]
        lbls = dev_df[["message_id", "is_service", "sentiment", "aggression"]]

        daily = compute_daily_aggregates(msgs, lbls)
        summ = compute_chat_summary(daily)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "chat_summary.csv"
            export_chat_summary_csv(summ, out_file)
            self.assertTrue(out_file.is_file())

            read_back = pd.read_csv(out_file, sep=";")
            self.assertListEqual(list(read_back.columns), REQUIRED_SUMMARY_COLUMNS)
            self.assertEqual(len(read_back), len(summ))


if __name__ == "__main__":
    unittest.main()
