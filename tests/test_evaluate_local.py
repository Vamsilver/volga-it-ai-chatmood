"""Модульные тесты для модуля локальной оценки (src/evaluate_local.py)."""
import unittest
import numpy as np
import pandas as pd

from src.evaluate_local import (
    evaluate_predictions,
    _evaluate_hard_tags,
    generate_diff_report,
)


class TestEvaluateLocal(unittest.TestCase):
    """Тестирование расчета метрик критериев 1 и 3."""

    def setUp(self):
        """Создание тестовых наборов данных."""
        self.gold_df = pd.DataFrame({
            "message_id": ["m1", "m2", "m3", "m4"],
            "is_service": [0, 0, 1, 0],
            "sentiment": ["negative", "positive", "neutral", "neutral"],
            "aggression": [1, 0, 0, 0],
            "tags": ["sarcasm", "emoji", "", "jargon"],
        })

    def test_perfect_predictions_score(self):
        """Проверка идеальных предсказаний."""
        pred_df = self.gold_df[["message_id", "is_service", "sentiment", "aggression"]].copy()
        res = evaluate_predictions(self.gold_df, pred_df)

        self.assertEqual(res["q1"], 1.0)
        self.assertEqual(res["points_c1"], 20.0)
        self.assertEqual(res["f1_sentiment_macro"], 1.0)
        self.assertEqual(res["f1_aggression"], 1.0)
        self.assertEqual(res["f1_service"], 1.0)
        self.assertEqual(res["accuracy_sentiment"], 1.0)

    def test_invariants_filter_residents(self):
        """Проверка, что тональность и агрессия оцениваются только по сообщениям жителей."""
        # Модель ошиблась на служебном сообщении (is_service=1) в sentiment
        pred_df = self.gold_df[["message_id", "is_service", "sentiment", "aggression"]].copy()
        pred_df.loc[pred_df["message_id"] == "m3", "sentiment"] = "negative"

        res = evaluate_predictions(self.gold_df, pred_df)
        # Поскольку m3 служебное, ошибка в sentiment m3 не должна влиять на f1_sentiment_macro жителей!
        self.assertEqual(res["f1_sentiment_macro"], 1.0)

    def test_diff_report_generation(self):
        """Проверка генерации отчета об ошибках."""
        pred_df = self.gold_df[["message_id", "is_service", "sentiment", "aggression"]].copy()
        # Создаем 1 ошибку
        pred_df.loc[pred_df["message_id"] == "m1", "sentiment"] = "positive"

        merged = self.gold_df.merge(pred_df, on="message_id", suffixes=("", "_p"))
        diff = generate_diff_report(merged)

        self.assertEqual(len(diff), 1)
        self.assertEqual(diff.iloc[0]["message_id"], "m1")


if __name__ == "__main__":
    unittest.main()
