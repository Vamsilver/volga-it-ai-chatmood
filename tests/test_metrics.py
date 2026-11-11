"""Модульные тесты для расчета метрик тренда и пиковых дней (src/aggregation/metrics.py)."""
import unittest
import numpy as np
import pandas as pd

import chatmood_metrics as cm
from src.aggregation.daily import compute_daily_aggregates
from src.aggregation.metrics import (
    calculate_conflict_score,
    calculate_mood_index,
    calculate_peak_days,
    calculate_trend,
)
from src.data.loader import load_dev


class TestAggregationMetrics(unittest.TestCase):
    """Тестирование формул расчета тренда, пиков и агрегированных индексов."""

    def setUp(self):
        """Загрузка валидационных данных dev."""
        self.dev_df = load_dev("data")
        msgs = self.dev_df[["message_id", "chat_id", "timestamp"]]
        lbls = self.dev_df[["message_id", "is_service", "sentiment", "aggression"]]
        self.daily = compute_daily_aggregates(msgs, lbls)

    def test_exact_match_trend_with_chatmood_metrics(self):
        """Проверка совпадения calculate_trend с cm.trend_of на всех чатах dev."""
        for chat_id, d in self.daily.groupby("chat_id"):
            our_trend = calculate_trend(d)
            gold_trend = cm.trend_of(d)
            self.assertEqual(
                our_trend,
                gold_trend,
                f"Несовпадение тренда для чата {chat_id}: {our_trend} != {gold_trend}",
            )

    def test_exact_match_peak_days_with_chatmood_metrics(self):
        """Проверка совпадения calculate_peak_days с cm.peak_days_of на всех чатах dev."""
        for chat_id, d in self.daily.groupby("chat_id"):
            our_peaks = calculate_peak_days(d)
            gold_peaks = cm.peak_days_of(d)
            self.assertListEqual(
                our_peaks,
                gold_peaks,
                f"Несовпадение пиковых дней для чата {chat_id}: {our_peaks} != {gold_peaks}",
            )

    def test_formulas_conflict_and_mood(self):
        """Проверка формул индекса конфликта и настроения."""
        # conflict = n_neg + 2 * n_agg
        self.assertEqual(calculate_conflict_score(3, 2), 7)
        np.testing.assert_array_equal(
            calculate_conflict_score([1, 2], [3, 4]), np.array([7, 10])
        )

        # mood_index = (pos - neg) / resident
        self.assertAlmostEqual(calculate_mood_index(5, 1, 10), 0.4)
        self.assertTrue(np.isnan(calculate_mood_index(5, 1, 0)))

    def test_synthetic_trends(self):
        """Проверка поведения линейной регрессии при явном улучшении и ухудшении."""
        # Создаем 3 ISO-недели (понедельники)
        dates = ["2026-06-01", "2026-06-08", "2026-06-15"]

        # Улучшение: mood от -0.8 до +0.8
        improving_df = pd.DataFrame({
            "date": dates,
            "n_resident": [10, 10, 10],
            "n_neg": [9, 5, 1],
            "n_pos": [1, 5, 9],
            "n_agg": [0, 0, 0],
        })
        self.assertEqual(calculate_trend(improving_df), "improving")

        # Ухудшение: mood от +0.8 до -0.8
        worsening_df = pd.DataFrame({
            "date": dates,
            "n_resident": [10, 10, 10],
            "n_neg": [1, 5, 9],
            "n_pos": [9, 5, 1],
            "n_agg": [0, 0, 0],
        })
        self.assertEqual(calculate_trend(worsening_df), "worsening")

        # Менее 3 недель -> stable
        short_df = improving_df.iloc[:2]
        self.assertEqual(calculate_trend(short_df), "stable")

    def test_peak_days_edge_cases(self):
        """Проверка порогов и сортировки пиковых дней."""
        # День с n_resident < 5 игнорируется
        df_low_support = pd.DataFrame({
            "date": ["2026-06-01"],
            "n_resident": [4],
            "n_neg": [4],
            "n_agg": [2],
        })
        self.assertEqual(calculate_peak_days(df_low_support), [])

        # День с conflict == 0 игнорируется
        df_zero_conflict = pd.DataFrame({
            "date": ["2026-06-01"],
            "n_resident": [10],
            "n_neg": [0],
            "n_agg": [0],
        })
        self.assertEqual(calculate_peak_days(df_zero_conflict), [])

        # При равенстве конфликта выбирается более ранняя дата
        df_tie = pd.DataFrame({
            "date": ["2026-06-10", "2026-06-05", "2026-06-01"],
            "n_resident": [10, 10, 10],
            "n_neg": [2, 2, 2],
            "n_agg": [1, 1, 1],
        })
        peaks = calculate_peak_days(df_tie)
        self.assertEqual(len(peaks), 3)
        self.assertListEqual(peaks, ["2026-06-01", "2026-06-05", "2026-06-10"])


if __name__ == "__main__":
    unittest.main()
