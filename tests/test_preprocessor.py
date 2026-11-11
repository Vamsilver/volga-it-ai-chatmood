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


if __name__ == "__main__":
    unittest.main()
