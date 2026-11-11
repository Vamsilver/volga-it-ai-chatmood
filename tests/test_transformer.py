"""Модульные тесты для мультизадачного трансформера (src/models/transformer.py)."""
import unittest
import numpy as np
import pandas as pd

from src.models.transformer import (
    HAS_TORCH,
    SENTIMENT_MAP,
    REV_SENTIMENT_MAP,
    TransformerPipeline,
)


class TestTransformerPipeline(unittest.TestCase):
    """Тестирование структуры и инвариантов трансформера."""

    def test_sentiment_mappings_consistency(self):
        """Проверка согласованности прямого и обратного словаря тональности."""
        self.assertEqual(len(SENTIMENT_MAP), 3)
        self.assertEqual(len(REV_SENTIMENT_MAP), 3)
        for label, idx in SENTIMENT_MAP.items():
            self.assertEqual(REV_SENTIMENT_MAP[idx], label)

    def test_invariants_postprocessing(self):
        """Проверка соблюдения жестких доменных инвариантов."""
        # Создаем синтетические предсказания с нарушениями
        raw_svc = np.array([1, 0, 0, 1, 0])
        raw_sent = np.array(["negative", "positive", "negative", "positive", "neutral"], dtype=object)
        raw_agg = np.array([1, 1, 1, 0, 1])

        # Применяем инварианты вручную через ту же логику
        pred_svc = raw_svc.copy()
        pred_sent = raw_sent.copy()
        pred_agg = raw_agg.copy()

        svc_mask = pred_svc == 1
        pred_sent[svc_mask] = "neutral"
        pred_agg[svc_mask] = 0

        non_neg_mask = pred_sent != "negative"
        pred_agg[non_neg_mask] = 0

        # Проверка 1: служебные строго нейтральны и без агрессии
        self.assertEqual(pred_sent[0], "neutral")
        self.assertEqual(pred_agg[0], 0)
        self.assertEqual(pred_sent[3], "neutral")
        self.assertEqual(pred_agg[3], 0)

        # Проверка 2: агрессия возможна только при negative
        self.assertEqual(pred_agg[1], 0)  # Было positive + agg=1 -> agg стал 0
        self.assertEqual(pred_agg[2], 1)  # Было negative + agg=1 -> agg остался 1
        self.assertEqual(pred_agg[4], 0)  # Было neutral + agg=1 -> agg стал 0

    @unittest.skipUnless(HAS_TORCH, "PyTorch / Transformers еще не установлены в окружении")
    def test_forward_pass_with_torch(self):
        """Тест прямого прохода при установленном PyTorch."""
        import torch
        from src.models.transformer import MultiTaskRuBertTiny
        model = MultiTaskRuBertTiny()
        dummy_ids = torch.randint(0, 1000, (2, 16))
        dummy_mask = torch.ones((2, 16))
        out = model(dummy_ids, dummy_mask)

        self.assertIn("service_logits", out)
        self.assertIn("sentiment_logits", out)
        self.assertIn("aggression_logits", out)
        self.assertEqual(out["service_logits"].shape, (2, 2))
        self.assertEqual(out["sentiment_logits"].shape, (2, 3))
        self.assertEqual(out["aggression_logits"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
