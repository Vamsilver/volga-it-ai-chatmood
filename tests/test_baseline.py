"""Модульные тесты для базовой модели TF-IDF + LogReg (src/models/baseline.py)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.baseline import BaselineClassifier, cross_validate_baseline


class TestBaselineClassifier(unittest.TestCase):
    """Тестирование BaselineClassifier и соблюдения инвариантов."""

    def setUp(self):
        """Создание синтетического набора данных для быстрых тестов."""
        self.texts = [
            "Уважаемые жители, завтра отключение ГВС с 10:00 до 17:00",
            "Спасибо за быструю починку домофона!",
            "Опять лифт не работает, сколько можно терпеть?!",
            "Вы тупая и не умеете читать объявления!",
            "Кто потерял ключи во дворе?",
            "Официальное объявление УК: прием граждан в среду",
            "Прекрасный двор, спасибо дворникам!",
            "Безобразие, почему опять грязь на лестнице?!",
        ]
        self.is_service = [1, 0, 0, 0, 0, 1, 0, 0]
        self.sentiment = [
            "neutral",
            "positive",
            "negative",
            "negative",
            "neutral",
            "neutral",
            "positive",
            "negative",
        ]
        self.aggression = [0, 0, 1, 1, 0, 0, 1, 1]
        self.chat_ids = ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"]
        self.message_ids = [f"m{i}" for i in range(len(self.texts))]

    def test_fit_and_predict_shapes(self):
        """Проверка обучения и формата вывода."""
        model = BaselineClassifier(random_state=42)
        model.fit(self.texts, self.is_service, self.sentiment, self.aggression)
        self.assertTrue(model.is_fitted)

        preds = model.predict(self.texts)
        self.assertEqual(len(preds), len(self.texts))
        self.assertListEqual(
            list(preds.columns), ["is_service", "sentiment", "aggression"]
        )

    def test_invariants_enforcement(self):
        """Проверка жестких доменных инвариантов на выходе."""
        model = BaselineClassifier(random_state=42)
        model.fit(self.texts, self.is_service, self.sentiment, self.aggression)

        preds = model.predict(self.texts)

        # Инвариант 1: is_service == 1 -> sentiment = neutral, aggression = 0
        svc_preds = preds[preds["is_service"] == 1]
        if len(svc_preds) > 0:
            self.assertTrue((svc_preds["sentiment"] == "neutral").all())
            self.assertTrue((svc_preds["aggression"] == 0).all())

        # Инвариант 2: sentiment != negative -> aggression = 0
        non_neg = preds[preds["sentiment"] != "negative"]
        if len(non_neg) > 0:
            self.assertTrue((non_neg["aggression"] == 0).all())

    def test_predict_messages_schema(self):
        """Проверка соответствия схеме messages_pred.csv."""
        model = BaselineClassifier(random_state=42)
        model.fit(self.texts, self.is_service, self.sentiment, self.aggression)

        df_in = pd.DataFrame({"message_id": self.message_ids, "text": self.texts})
        df_out = model.predict_messages(df_in)

        self.assertListEqual(
            list(df_out.columns),
            ["message_id", "is_service", "sentiment", "aggression"],
        )
        self.assertEqual(len(df_out), len(df_in))

    def test_save_and_load(self):
        """Проверка сохранения и загрузки весов."""
        model = BaselineClassifier(random_state=42)
        model.fit(self.texts, self.is_service, self.sentiment, self.aggression)
        preds_orig = model.predict(self.texts)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "baseline.joblib"
            model.save(model_path)
            self.assertTrue(model_path.is_file())

            loaded_model = BaselineClassifier.load(model_path)
            self.assertTrue(loaded_model.is_fitted)

            preds_loaded = loaded_model.predict(self.texts)
            pd.testing.assert_frame_equal(preds_orig, preds_loaded)

    def test_cross_validate_baseline(self):
        """Проверка функции кросс-валидации по chat_id."""
        df = pd.DataFrame({
            "chat_id": self.chat_ids * 2,
            "text": self.texts * 2,
            "is_service": self.is_service * 2,
            "sentiment": self.sentiment * 2,
            "aggression": self.aggression * 2,
        })
        cv_res = cross_validate_baseline(df, n_splits=2, random_state=42, min_df=1)
        self.assertIn("q1_mean", cv_res)
        self.assertIn("f1_sentiment_mean", cv_res)
        self.assertIn("f1_service_mean", cv_res)
        self.assertIn("f1_aggression_mean", cv_res)

    def test_train_on_real_sample(self):
        """Проверка обучения на подвыборке реальных данных из train.csv."""
        from src.data.loader import load_train
        train_df = load_train(data_dir="data").head(150)
        model = BaselineClassifier(random_state=42, min_df=2)
        model.fit(
            train_df["text"],
            train_df["is_service"],
            train_df["sentiment"],
            train_df["aggression"],
        )
        preds = model.predict(train_df["text"])
        self.assertEqual(len(preds), 150)
        # Проверка инвариантов
        svc_mask = preds["is_service"] == 1
        if svc_mask.any():
            self.assertTrue((preds.loc[svc_mask, "sentiment"] == "neutral").all())
            self.assertTrue((preds.loc[svc_mask, "aggression"] == 0).all())


if __name__ == "__main__":
    unittest.main()
