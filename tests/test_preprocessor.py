"""Модульные тесты для препроцессора текста (src/features/preprocessor.py)."""
import unittest
import pandas as pd

from src.features.preprocessor import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):
    """Тестирование нормализации специфики доменных текстов ЖКХ."""

    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_jkh_jargon_expansion(self):
        """Проверка расшифровки сокращений ЖКХ."""
        text = "Когда дадут ГВС и ХВС? Наша УК ничего не делает!"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("горячее водоснабжение", cleaned)
        self.assertIn("холодное водоснабжение", cleaned)
        self.assertIn("управляющая компания", cleaned)

    def test_emoji_mapping(self):
        """Проверка замены эмодзи на семантические токены."""
        text = "Спасибо за уборку 🙏! А лифт опять сломался 😡🤣"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("[EMOJI_POS]", cleaned)
        self.assertIn("[EMOJI_NEG]", cleaned)
        self.assertIn("[EMOJI_LAUGH]", cleaned)

    def test_obscene_masking(self):
        """Проверка детекции обфусцированного мата."""
        text = "В Сосновом Бору все руко@опые, пипец!"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("[OBSCENE]", cleaned)

    def test_marker_isolation_and_typos(self):
        """Проверка разделения маркеров и опечаток склейки."""
        text = "[адрес].Воды нет.Завтра будет?"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("[адрес]", cleaned)
        self.assertIn("Воды нет. Завтра", cleaned)

    def test_feature_extraction(self):
        """Проверка извлечения числовых и булевых признаков."""
        df = pd.DataFrame({
            "text": [
                "Объявление: [адрес], [телефон] прием граждан",
                "СПАСИБО ОГРОМНОЕ ЗА ПОМОЩЬ!!! 🙏",
                "Опять лифт не работает 😡",
            ]
        })
        feats = self.preprocessor.extract_features(df)
        self.assertEqual(len(feats), 3)
        self.assertEqual(feats.iloc[0]["has_address"], 1)
        self.assertEqual(feats.iloc[0]["has_phone"], 1)
        self.assertGreater(feats.iloc[1]["caps_ratio"], 0.5)
        self.assertEqual(feats.iloc[1]["has_emoji_pos"], 1)
        self.assertEqual(feats.iloc[1]["has_excl_high"], 1)
        self.assertEqual(feats.iloc[2]["has_emoji_neg"], 1)

    def test_aggression_signal(self):
        """Проверка детекции сигналов агрессии и оскорблений."""
        df = pd.DataFrame({
            "text": [
                "Вы свою голову лучше включите!!!!!!!",
                "Конченные люди",
                "В Сосновом Бору все руко@опые",
                "Спокойной ночи, соседи",
            ]
        })
        feats = self.preprocessor.extract_features(df)
        self.assertEqual(feats.iloc[0]["has_aggression_signal"], 1)
        self.assertEqual(feats.iloc[1]["has_aggression_signal"], 1)
        self.assertEqual(feats.iloc[2]["has_aggression_signal"], 1)
        self.assertEqual(feats.iloc[3]["has_aggression_signal"], 0)

    def test_sarcasm_signal(self):
        """Проверка детекции сигналов сарказма и иронии."""
        df = pd.DataFrame({
            "text": [
                "Спасибо за горячую воду, которой нет уже неделю!",
                "Наша УК нас бережет, как же 😂",
                "Обычный рабочий день",
            ]
        })
        feats = self.preprocessor.extract_features(df)
        self.assertEqual(feats.iloc[0]["has_sarcasm_signal"], 1)
        self.assertEqual(feats.iloc[1]["has_sarcasm_signal"], 1)
        self.assertEqual(feats.iloc[2]["has_sarcasm_signal"], 0)

    def test_enrich_chat_context(self):
        """Проверка контекстного обогащения коротких реплик-реакций."""
        from src.features.preprocessor import enrich_chat_context

        df = pd.DataFrame({
            "chat_id": ["c1", "c1", "c1", "c1", "c2"],
            "timestamp": [
                "2026-06-01 10:00:00",
                "2026-06-01 10:01:00",  # +60s, реакция
                "2026-06-01 10:02:00",  # +60s, позитив (НЕ обогащать)
                "2026-06-01 10:40:00",  # +38m, слишком большой разрыв
                "2026-06-01 10:00:30",  # другой чат, первое сообщение
            ],
            "text": [
                "Лифт сломался, опять пешком на 15 этаж",
                "И на 9 тоже",
                "Спасибо, мастер уже пришел!",
                "Аналогично",
                "Согласен",
            ]
        })

        enriched = enrich_chat_context(df, max_time_diff_sec=300)
        self.assertEqual(len(enriched), 5)
        # 1-я реплика - без контекста
        self.assertEqual(enriched.iloc[0], "Лифт сломался, опять пешком на 15 этаж")
        # 2-я реплика - обогащена контекстом лифта
        self.assertIn("[Контекст: Лифт сломался, опять пешком на 15 этаж]", enriched.iloc[1])
        # 3-я реплика - позитив, не обогащается
        self.assertEqual(enriched.iloc[2], "Спасибо, мастер уже пришел!")
        # 4-я реплика - превышен таймаут, не обогащается
        self.assertEqual(enriched.iloc[3], "Аналогично")
        # 5-я реплика - первое сообщение в чате c2, не обогащается
        self.assertEqual(enriched.iloc[4], "Согласен")

    def test_typo_normalization(self):
        """Проверка нормализации опечаток в корнях проблемных слов (Milestone 12.2)."""
        raw = "Тут жмть невозможно, лдифт сломалсо, отопленее отключино!"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertIn("жить невозможно", cleaned)
        self.assertIn("лифт", cleaned)
        self.assertIn("сломался", cleaned)
        self.assertIn("отопление", cleaned)
        self.assertIn("отключено", cleaned)

    def test_extended_jkh_and_emoji(self):
        """Проверка расширенных словарей ЖКХ и эмодзи."""
        text = "Провели ОСС, передали в АДС и Квадо 🫶. Но в 3 парадная нарки все сломали 🤐😭"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("общее собрание собственников", cleaned)
        self.assertIn("аварийно диспетчерская служба", cleaned)
        self.assertIn("система заявок", cleaned)
        self.assertIn("подъезд", cleaned)
        self.assertIn("наркоманы", cleaned)
        self.assertIn("[EMOJI_POS]", cleaned)
        self.assertIn("[EMOJI_NEG]", cleaned)

    def test_extended_typo_normalization(self):
        """Проверка расширенных опечаток."""
        text = "Мы добролись до навнодушнего соседа, поаравтлиа замок, сабака спит."
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("добрались", cleaned)
        self.assertIn("равнодушного", cleaned)
        self.assertIn("поправила", cleaned)
        self.assertIn("собака", cleaned)

    def test_is_context_reaction_and_stuck_punctuation(self):
        """Проверка детекции реакций с прилипшей пунктуацией."""
        from src.features.preprocessor import is_context_reaction
        self.assertTrue(is_context_reaction("2парадная ,аналогично"))
        self.assertTrue(is_context_reaction("Согласен"))
        self.assertTrue(is_context_reaction("я с вами"))
        self.assertTrue(is_context_reaction("+1"))
        self.assertTrue(is_context_reaction("даааа!!"))
        self.assertFalse(is_context_reaction("Спасибо огромное за оперативную помощь!"))
        self.assertFalse(is_context_reaction("Это длинное сообщение с подробным описанием проблем в парадной"))


if __name__ == "__main__":
    unittest.main()
