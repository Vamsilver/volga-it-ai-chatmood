"""Модульные тесты для ансамбля моделей (src/models/ensemble.py)."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models.baseline import BaselineClassifier
from src.models.ensemble import EnsembleClassifier


class TestEnsembleClassifier(unittest.TestCase):
    """Тестирование EnsembleClassifier, хардкейсов и инвариантов."""

    def setUp(self):
        """Создание тестовой выборки."""
        self.texts = [
            "Уважаемые жители, завтра отключение ГВС с 10:00 до 17:00",
            "Спасибо за отличную работу управляющей компании! Все чисто.",
            "Опять лифт сломался, сколько можно терпеть этот беспредел?!",
            "Вы совсем рукожопые уроды, насрать на людей!",
            "Спасибо за горячую воду, которой нет уже неделю 😂😂😂",
            "Кто потерял связку ключей около 2 парадной?",
            "Официальное уведомление УК: поверка счетчиков ИПУ",
            "Безобразие, почему опять грязь на лестнице? Руко@опые уборщики!",
        ]
        self.is_service = [1, 0, 0, 0, 0, 0, 1, 0]
        self.sentiment = [
            "neutral",
            "positive",
            "negative",
            "negative",
            "negative",
            "neutral",
            "neutral",
            "negative",
        ]
        self.aggression = [0, 0, 1, 1, 0, 0, 0, 1]
        self.message_ids = [f"m{i}" for i in range(len(self.texts))]

    def test_fit_and_predict_shapes(self):
        """Проверка обучения и структуры вывода предсказаний."""
        ensemble = EnsembleClassifier()
        ensemble.fit(self.texts, self.is_service, self.sentiment, self.aggression)

        preds = ensemble.predict(self.texts)
        self.assertEqual(len(preds), len(self.texts))
        self.assertListEqual(
            list(preds.columns), ["is_service", "sentiment", "aggression"]
        )

    def test_invariants_enforcement(self):
        """Проверка безусловного соблюдения инвариантов разметки."""
        ensemble = EnsembleClassifier()
        ensemble.fit(self.texts, self.is_service, self.sentiment, self.aggression)

        preds = ensemble.predict(self.texts)

        # Инвариант 1: is_service == 1 -> neutral и aggression == 0
        svc_preds = preds[preds["is_service"] == 1]
        if len(svc_preds) > 0:
            self.assertTrue((svc_preds["sentiment"] == "neutral").all())
            self.assertTrue((svc_preds["aggression"] == 0).all())

        # Инвариант 2: sentiment != negative -> aggression == 0
        non_neg = preds[preds["sentiment"] != "negative"]
        if len(non_neg) > 0:
            self.assertTrue((non_neg["aggression"] == 0).all())

    def test_hardcases_obscene_handling(self):
        """Проверка детекции мата и усиления негатива/агрессии."""
        ensemble = EnsembleClassifier()
        ensemble.fit(self.texts, self.is_service, self.sentiment, self.aggression)

        obscene_msg = ["Вы руко@опые уроды, достали уже!"]
        probs = ensemble.predict_proba(obscene_msg)
        self.assertGreater(probs["sentiment"][0, 0], 0.3)
        self.assertGreater(probs["aggression"][0], 0.3)

    def test_predict_messages_format(self):
        """Проверка соответствия схеме сообщений с message_id."""
        ensemble = EnsembleClassifier()
        ensemble.fit(self.texts, self.is_service, self.sentiment, self.aggression)

        df_in = pd.DataFrame({"message_id": self.message_ids, "text": self.texts})
        df_out = ensemble.predict_messages(df_in)

        self.assertListEqual(
            list(df_out.columns),
            ["message_id", "is_service", "sentiment", "aggression"],
        )
        self.assertEqual(len(df_out), len(df_in))

    def test_save_and_load(self):
        """Проверка сериализации и десериализации ансамбля."""
        ensemble = EnsembleClassifier()
        ensemble.fit(self.texts, self.is_service, self.sentiment, self.aggression)
        preds_orig = ensemble.predict(self.texts)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "ensemble"
            ensemble.save(save_path)

            self.assertTrue((save_path / "ensemble_config.json").is_file())
            self.assertTrue((save_path / "baseline.joblib").is_file())

            loaded = EnsembleClassifier.load(save_path)
            preds_loaded = loaded.predict(self.texts)

            pd.testing.assert_frame_equal(preds_orig, preds_loaded)

    def test_from_pretrained_with_disk_models(self):
        """Проверка загрузки реальных обученных моделей из models/."""
        baseline_path = Path("models/baseline.joblib")
        if not baseline_path.is_file():
            self.skipTest("models/baseline.joblib отсутствует на диске")

        ensemble = EnsembleClassifier.from_pretrained(
            baseline_path=baseline_path,
            transformer_dir="models/rubert_tiny2_multitask",
        )
        self.assertIsNotNone(ensemble.baseline)

        preds = ensemble.predict(self.texts[:3])
        self.assertEqual(len(preds), 3)
        self.assertListEqual(
            list(preds.columns), ["is_service", "sentiment", "aggression"]
        )

    def test_insult_hard_override(self):
        """Проверка принудительного выставления negative и aggression=1 для прямых оскорблений."""
        ensemble = EnsembleClassifier()
        ensemble.fit(self.texts, self.is_service, self.sentiment, self.aggression)

        insult_msgs = [
            "Конченные люди, слов нет!",
            "Вы лучше включите свою голову и подумайте",
        ]
        preds = ensemble.predict(insult_msgs)
        self.assertEqual(preds.iloc[0]["sentiment"], "negative")
        self.assertEqual(preds.iloc[0]["aggression"], 1)
        self.assertEqual(preds.iloc[1]["sentiment"], "negative")
        self.assertEqual(preds.iloc[1]["aggression"], 1)

    def test_service_override(self):
        """Проверка детекции служебных рассылок и исключений обращений жителей."""
        ensemble = EnsembleClassifier()
        ensemble.fit(self.texts, self.is_service, self.sentiment, self.aggression)

        test_msgs = [
            "Счета за июнь уже доступны в приложении Госуслуги Дом. Оплачивайте вовремя!",
            "Уважаемый представитель УК! Когда наконец починят домофон в 3 подъезде?",
        ]
        preds = ensemble.predict(test_msgs)
        self.assertEqual(preds.iloc[0]["is_service"], 1)
        self.assertEqual(preds.iloc[0]["sentiment"], "neutral")
        self.assertEqual(preds.iloc[1]["is_service"], 0)

    def test_sarcasm_and_emoji_override(self):
        """Проверка оверрайдов для сарказма и одиночных эмодзи (Milestone 8.3)."""
        ensemble = EnsembleClassifier()
        ensemble.fit(self.texts, self.is_service, self.sentiment, self.aggression)

        test_msgs = [
            "🤡",
            "👍 🙏",
            "Спасибо за отопление, которого нет 2 дня",
            "Зато в подъезде тепло 😡",
            "В 3-й секции вот такая красота. Прошу принять меры.",
            "Передаю огромный привет родителям ребят, которые кидали яйца с балкона.",
            "Отписывайтесь дальше",
        ]
        preds = ensemble.predict(test_msgs)
        self.assertEqual(preds.iloc[0]["sentiment"], "negative")
        self.assertEqual(preds.iloc[1]["sentiment"], "positive")
        self.assertEqual(preds.iloc[2]["sentiment"], "negative")
        self.assertEqual(preds.iloc[3]["sentiment"], "negative")
        self.assertEqual(preds.iloc[4]["sentiment"], "negative")
        self.assertEqual(preds.iloc[5]["sentiment"], "negative")
        self.assertEqual(preds.iloc[6]["sentiment"], "negative")

    def test_strong_positive_override(self):
        """Проверка оверрайдов для сильных позитивных маркеров (Milestone 11.2)."""
        ensemble = EnsembleClassifier()
        ensemble.fit(self.texts, self.is_service, self.sentiment, self.aggression)

        test_msgs = [
            "Сочная, обалденная рыбка!",
            "Милота какая)",
            "Поздравляю всех с праздником!",
            "Спасибо неравнодушные соседи!",
        ]
        preds = ensemble.predict(test_msgs)
        for i in range(len(test_msgs)):
            self.assertEqual(preds.iloc[i]["sentiment"], "positive")
            self.assertEqual(preds.iloc[i]["aggression"], 0)


if __name__ == "__main__":
    unittest.main()
