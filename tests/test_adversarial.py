"""Adversarial stress-test suite for preprocessor and ensemble models.

Author: Challenger 1 (Model Robustness & Edge Cases)
Verifies:
1. Domain invariants: is_service == 1 -> neutral & 0 aggression; sentiment != negative -> 0 aggression.
2. Extreme inputs: empty strings, pure whitespace, huge texts, punctuation-only, special characters.
3. Emoji boundary cases: pure positive/negative emojis, punctuation variations, unmapped emojis.
4. Sarcasm boundary cases: sarcastic complaints vs polite explanations/positive gratitude.
5. Aggression boundary cases: personal insults/threats vs harsh civil complaints.
6. Context reaction propagation: isolated pairs, chat isolation, time threshold (>300s).
"""
import unittest
import numpy as np
import pandas as pd

from predict import load_prediction_model
from src.features.preprocessor import TextPreprocessor, is_context_reaction


class TestAdversarialHarness(unittest.TestCase):
    """Adversarial test cases covering extreme boundaries and stress conditions."""

    @classmethod
    def setUpClass(cls):
        """Load the production ensemble model once for the test suite."""
        cls.model = load_prediction_model("models", device="cpu")
        cls.prep = TextPreprocessor()

    def test_pipeline_invariants_under_adversarial_pressure(self):
        """Invariance check: service announcements must NEVER be non-neutral or aggressive, even with insults."""
        adversarial_texts = [
            "Уважаемые жители, вы конченные дебилы!",
            "Госуслуги Дом: аварийно-восстановительные работы, закрой рот тварь!",
            "Лифт остановлен на ремонт, рукожопые уроды!",
            "График отключения горячей воды: пошли нах*й!",
            "Уважаемые собственники, спасибо огромное ❤️🙏",
            "Плановое отключение ГВС 😂😂😂",
            "Информируем вас: отопление отключено 😡🤬",
            "Официальное уведомление УК: поверка ИПУ 💩",
            "Госуслуги Дом: счета доступны, урааа!",
        ]
        df = pd.DataFrame({
            "message_id": [f"adv_inv_{i}" for i in range(len(adversarial_texts))],
            "text": adversarial_texts,
        })
        preds = self.model.predict_messages(df)

        for i, row in preds.iterrows():
            if row["is_service"] == 1:
                self.assertEqual(row["sentiment"], "neutral", f"Service violated neutral: {adversarial_texts[i]}")
                self.assertEqual(row["aggression"], 0, f"Service violated 0 agg: {adversarial_texts[i]}")
            if row["sentiment"] != "negative":
                self.assertEqual(row["aggression"], 0, f"Non-negative violated 0 agg: {adversarial_texts[i]}")

    def test_extreme_inputs_no_crash(self):
        """Check extreme inputs: empty, whitespace, null bytes, long text, punctuation."""
        extreme_cases = [
            "",
            "   \t\r\n   ",
            ".....",
            "?!?!?!?",
            "---",
            "\x00\x00\x00",
            "A" * 15000,
            "PRIVET sosedi, lift slomalsya",
            "1234567890",
        ]
        df = pd.DataFrame({
            "message_id": [f"ext_{i}" for i in range(len(extreme_cases))],
            "text": extreme_cases,
        })
        preds = self.model.predict_messages(df)
        self.assertEqual(len(preds), len(extreme_cases))
        for row in preds.itertuples():
            self.assertIn(row.sentiment, ("negative", "neutral", "positive"))
            self.assertIn(row.aggression, (0, 1))
            self.assertIn(row.is_service, (0, 1))

    def test_emoji_sentiment_polarity(self):
        """Pure emojis should be appropriately classified as positive/negative without aggression."""
        pos_emojis = ["👍", "👍!", "👍👍👍", "🙏❤️", "🥰🥳"]
        neg_emojis = ["😡", "😡!", "😡😡", "🤬🤮", "🤦👎"]

        df_pos = pd.DataFrame({"message_id": [f"p_{i}" for i in range(len(pos_emojis))], "text": pos_emojis})
        preds_pos = self.model.predict_messages(df_pos)
        for s, a in zip(preds_pos["sentiment"], preds_pos["aggression"]):
            self.assertEqual(s, "positive")
            self.assertEqual(a, 0)

        df_neg = pd.DataFrame({"message_id": [f"n_{i}" for i in range(len(neg_emojis))], "text": neg_emojis})
        preds_neg = self.model.predict_messages(df_neg)
        for s in preds_neg["sentiment"]:
            self.assertEqual(s, "negative")

    def test_sarcasm_vs_polite_explanations(self):
        """Sarcasm detection must trigger negative without false-positive negatives on polite/helpful texts."""
        sarcastic_texts = [
            "Спасибо за горячую воду, которой нет уже неделю",
            "Спасибо за кипяток вместо холодной воды! 😂",
            "Наша УК нас бережет, как же 😂",
            "Зато у нас самый красивый сугроб во дворе 💩",
            "Наш любимый лифт опять отдыхает ))",
            "Открою вам тайну: уборщицы у нас нет",
            "Красота, мусор под окнами, примите меры",
            "Передаю привет соседям сверху, которые бросают окурки с балкона",
        ]
        df_sarc = pd.DataFrame({"message_id": [f"s_{i}" for i in range(len(sarcastic_texts))], "text": sarcastic_texts})
        preds_sarc = self.model.predict_messages(df_sarc)
        for text, sent in zip(sarcastic_texts, preds_sarc["sentiment"]):
            self.assertEqual(sent, "negative", f"Failed sarcasm detection on: {text}")

        polite_texts = [
            "Вы хоть не вздумайте включать автомат под напряжением",
            "Интересно, а почему отключили отопление?",
            "Потому что авария на теплотрассе)",
            "Подскажите, пожалуйста, до скольки работает паспортный стол?",
            "Спасибо за информацию, будем знать!",
            "Спасибо огромное за помощь с замком!",
        ]
        df_polite = pd.DataFrame({"message_id": [f"pol_{i}" for i in range(len(polite_texts))], "text": polite_texts})
        preds_polite = self.model.predict_messages(df_polite)
        for text, sent in zip(polite_texts, preds_polite["sentiment"]):
            self.assertNotEqual(sent, "negative", f"False positive negative on polite text: {text}")

    def test_personal_insults_vs_civil_complaints(self):
        """Direct insults must trigger aggression=1, while harsh civil complaints must remain aggression=0."""
        insults = [
            "Вы совсем дебилы, конченные твари!",
            "Рот закрой и молчи, быдло",
            "Слышишь меня, морду набью тебе",
            "Какие родители, такие дети, упыри",
            "Нормальный человек до такого бы не додумался, придурок",
            "Включи мозги свои, идиот",
            "Вы рукожопые уроды, насрать на вас",
        ]
        df_insults = pd.DataFrame({"message_id": [f"ins_{i}" for i in range(len(insults))], "text": insults})
        preds_insults = self.model.predict_messages(df_insults)
        for text, sent, agg in zip(insults, preds_insults["sentiment"], preds_insults["aggression"]):
            self.assertEqual(agg, 1, f"Failed aggression detection on insult: {text}")
            self.assertEqual(sent, "negative", f"Insult must be negative: {text}")

        civil_complaints = [
            "Из крана течет ржавая и грязная вода уже третий день подряд",
            "Лифт снова сломался между этажами, почините немедленно",
            "В подъезде ужасная вонь и грязь, мусоропровод забит",
            "Опять отключили отопление без предупреждения, дома жуткий холод",
            "На придомовой территории ямы и лужи, невозможно пройти с коляской",
            "Диспетчер не берет трубку, аварийная служба не реагирует на заявки",
        ]
        df_civil = pd.DataFrame({"message_id": [f"civ_{i}" for i in range(len(civil_complaints))], "text": civil_complaints})
        preds_civil = self.model.predict_messages(df_civil)
        for text, agg in zip(civil_complaints, preds_civil["aggression"]):
            self.assertEqual(agg, 0, f"False positive aggression on civil complaint: {text}")

    def test_context_reaction_propagation_isolated_pairs(self):
        """Short agreement replies should propagate parent sentiment in isolated 2-turn dialogs."""
        reactions = ["+1", "Аналогично", "Тоже", "Поддерживаю", "Да, именно", "У нас тоже", "Согласен", "Факт", "Плюсую"]

        # Negative parent
        rows_neg = []
        for idx, r in enumerate(reactions):
            c_id = f"chat_neg_{idx}"
            rows_neg.append((c_id, "2026-06-01 10:00:00", "Лифт опять сломался, безобразие!"))
            rows_neg.append((c_id, "2026-06-01 10:01:00", r))

        df_neg = pd.DataFrame({
            "message_id": [f"rn_{i}" for i in range(len(rows_neg))],
            "chat_id": [r[0] for r in rows_neg],
            "timestamp": [r[1] for r in rows_neg],
            "text": [r[2] for r in rows_neg],
        })
        preds_neg = self.model.predict_messages(df_neg)
        for idx, r in enumerate(reactions):
            c_sent = preds_neg.loc[idx * 2 + 1, "sentiment"]
            self.assertEqual(c_sent, "negative", f"Reaction '{r}' failed to inherit negative from parent")

        # Positive parent
        rows_pos = []
        for idx, r in enumerate(reactions):
            c_id = f"chat_pos_{idx}"
            rows_pos.append((c_id, "2026-06-01 10:00:00", "Спасибо УК за отличную уборку снега! Все чисто."))
            rows_pos.append((c_id, "2026-06-01 10:01:00", r))

        df_pos = pd.DataFrame({
            "message_id": [f"rp_{i}" for i in range(len(rows_pos))],
            "chat_id": [r[0] for r in rows_pos],
            "timestamp": [r[1] for r in rows_pos],
            "text": [r[2] for r in rows_pos],
        })
        preds_pos = self.model.predict_messages(df_pos)
        for idx, r in enumerate(reactions):
            c_sent = preds_pos.loc[idx * 2 + 1, "sentiment"]
            self.assertEqual(c_sent, "positive", f"Reaction '{r}' failed to inherit positive from parent")

    def test_context_time_threshold_and_chat_isolation(self):
        """Reaction propagation must respect chat isolation and expire when dt > 300s."""
        rows = [
            ("c_iso1", "2026-06-01 10:00:00", "Лифт сломался!"),
            ("c_iso1", "2026-06-01 10:10:00", "+1"),  # dt = 600s (>300s) -> should NOT propagate negative
            ("c_iso2", "2026-06-01 10:00:00", "Лифт сломался!"),
            ("c_iso3", "2026-06-01 10:00:30", "+1"),  # different chat -> should NOT propagate
        ]
        df = pd.DataFrame({
            "message_id": [f"t_{i}" for i in range(len(rows))],
            "chat_id": [r[0] for r in rows],
            "timestamp": [r[1] for r in rows],
            "text": [r[2] for r in rows],
        })
        preds = self.model.predict_messages(df)
        self.assertEqual(preds.loc[1, "sentiment"], "neutral", "dt > 300s should not propagate negative")
        self.assertEqual(preds.loc[3, "sentiment"], "neutral", "different chat should not propagate negative")


if __name__ == "__main__":
    unittest.main()
